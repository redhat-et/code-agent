"""K8s Pod environment for SWE-bench RL training.

Manages a persistent K8s Pod as an execution sandbox for one agent
rollout.  Unlike the eval pipeline's AgentWorker (which launches
self-contained K8s Jobs), this creates a long-lived Pod that accepts
commands via ``kubectl exec`` and stays alive for the full episode.

Lifecycle:
  env = SWEBenchEnvironment()
  await env.create(instance_dict)   # spin up Pod
  out, rc = await env.execute("ls") # run commands
  patch   = await env.get_patch()   # extract diff
  ok, log = await env.run_eval()    # grade the patch
  await env.destroy()               # clean up
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import re
import time
from functools import partial

from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes.stream import stream as k8s_stream
from swebench.harness.constants import DOCKER_PATCH, DOCKER_WORKDIR

from evals.swe_bench.agent_worker import resolve_instance
from evals.swe_bench.grader import grade_instance

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 2
_POD_READY_TIMEOUT = int(os.environ.get("SWE_POD_READY_TIMEOUT", "120"))
_EXEC_TIMEOUT = int(os.environ.get("SWE_EXEC_TIMEOUT", "120"))

_RC_PATTERN = re.compile(r"__RC_8372916__:(\d+)\n?")

_GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


def _init_k8s():
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()
    return k8s_client.CoreV1Api()


def _detect_namespace() -> str:
    ns_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    try:
        with open(ns_path) as f:
            return f.read().strip()
    except FileNotFoundError:
        return "default"


def _pod_name(instance_id: str) -> str:
    safe_id = instance_id.lower().replace("__", "-").replace("_", "-")
    uid = hashlib.md5(f"{instance_id}-{time.time()}".encode()).hexdigest()[:6]
    max_base = 63 - len("rl--") - len(uid)
    safe_id = safe_id[:max_base].strip("-")
    return f"rl-{safe_id}-{uid}"


class SWEBenchEnvironment:
    """Manages one K8s Pod as a sandbox for an agent episode."""

    def __init__(
        self,
        image_registry: str = "",
        namespace: str | None = None,
        service_account: str = "swe-bench-training",
    ):
        self._core_api = _init_k8s()
        self._namespace = namespace or _detect_namespace()
        self._image_registry = image_registry
        self._service_account = service_account

        self._pod_name: str | None = None
        self._test_spec = None
        self._eval_script: str = ""
        self._instance_id: str = ""

    async def create(self, instance: dict) -> None:
        """Create a Pod with the SWE-bench instance image and wait for Ready."""
        self._instance_id = instance["instance_id"]
        info = resolve_instance(instance, self._image_registry)
        self._test_spec = info.test_spec
        self._eval_script = info.eval_script

        name = _pod_name(self._instance_id)
        self._pod_name = name

        init_container = k8s_client.V1Container(
            name="fix-permissions",
            image=info.image,
            image_pull_policy="IfNotPresent",
            command=[
                "/bin/sh", "-c",
                f"chown -R 1001:0 {DOCKER_WORKDIR} && "
                f"chmod -R a+rwX {DOCKER_WORKDIR} && "
                "chown -R 1001:0 /opt/miniconda3 2>/dev/null || true && "
                "chmod -R a+rwX /opt/miniconda3 2>/dev/null || true",
            ],
            security_context=k8s_client.V1SecurityContext(
                allow_privilege_escalation=False,
                run_as_user=0,
                run_as_non_root=False,
                capabilities=k8s_client.V1Capabilities(drop=["ALL"]),
                privileged=False,
            ),
        )

        container = k8s_client.V1Container(
            name="sandbox",
            image=info.image,
            image_pull_policy="IfNotPresent",
            command=["/bin/bash", "-c", "sleep infinity"],
            working_dir=DOCKER_WORKDIR,
            security_context=k8s_client.V1SecurityContext(
                allow_privilege_escalation=False,
                run_as_non_root=True,
                run_as_user=1001,
                capabilities=k8s_client.V1Capabilities(drop=["ALL"]),
            ),
            resources=k8s_client.V1ResourceRequirements(
                requests={"cpu": "1", "memory": "2Gi"},
                limits={"cpu": "2", "memory": "4Gi", "ephemeral-storage": "4Gi"},
            ),
        )

        pod = k8s_client.V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=k8s_client.V1ObjectMeta(
                name=name,
                namespace=self._namespace,
                labels={
                    "app": "swe-bench-rl",
                    "instance-id": self._instance_id[:63],
                },
            ),
            spec=k8s_client.V1PodSpec(
                init_containers=[init_container],
                containers=[container],
                restart_policy="Never",
                service_account_name=self._service_account,
                automount_service_account_token=False,
                security_context=k8s_client.V1PodSecurityContext(
                    run_as_non_root=True,
                    fs_group=0,
                    supplemental_groups=[0],
                ),
            ),
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(
                self._core_api.create_namespaced_pod,
                namespace=self._namespace,
                body=pod,
            ),
        )
        logger.info(f"[{self._instance_id}] Created Pod {name}")
        await self._wait_ready()

    async def _wait_ready(self) -> None:
        loop = asyncio.get_event_loop()
        deadline = time.time() + _POD_READY_TIMEOUT

        while time.time() < deadline:
            pod = await loop.run_in_executor(
                None,
                partial(
                    self._core_api.read_namespaced_pod,
                    name=self._pod_name,
                    namespace=self._namespace,
                ),
            )
            if pod.status.phase == "Running":
                conditions = pod.status.conditions or []
                if any(c.type == "Ready" and c.status == "True" for c in conditions):
                    logger.info(f"[{self._instance_id}] Pod ready")
                    return
            if pod.status.phase in ("Failed", "Succeeded"):
                raise RuntimeError(
                    f"Pod {self._pod_name} ended with phase {pod.status.phase}"
                )
            await asyncio.sleep(_POLL_INTERVAL)

        raise TimeoutError(
            f"Pod {self._pod_name} not ready within {_POD_READY_TIMEOUT}s"
        )

    async def execute(
        self, command: str, timeout: int = _EXEC_TIMEOUT
    ) -> tuple[str, int]:
        """Execute a bash command in the Pod.

        Returns (combined_output, exit_code).
        """
        sentinel = "__RC_8372916__"
        # Subshell so `exit N` doesn't kill the outer shell before the sentinel.
        wrapped = f'({command}); echo "{sentinel}:$?"'

        loop = asyncio.get_event_loop()
        raw_output: str = await loop.run_in_executor(
            None,
            partial(
                k8s_stream,
                self._core_api.connect_get_namespaced_pod_exec,
                self._pod_name,
                self._namespace,
                command=["/bin/bash", "-c", wrapped],
                stdout=True,
                stderr=True,
                stdin=False,
                tty=False,
                _request_timeout=timeout,
            ),
        )

        # Stdout/stderr are interleaved, so search for the sentinel anywhere.
        rc_match = _RC_PATTERN.search(raw_output)
        if rc_match:
            exit_code = int(rc_match.group(1))
            output = (
                raw_output[: rc_match.start()] + raw_output[rc_match.end() :]
            ).strip()
        else:
            output = raw_output
            exit_code = -1

        return output, exit_code

    async def get_patch(self) -> str:
        """Extract the current diff from the repo."""
        output, _ = await self.execute(
            f"cd {DOCKER_WORKDIR} && git diff", timeout=30
        )
        return output.strip()

    async def run_eval(self) -> tuple[bool, str]:
        """Run the SWE-bench eval script and grade the result.

        Returns (resolved, eval_output).
        """
        patch = await self.get_patch()
        if not patch:
            return False, "empty patch"

        # git apply requires a trailing newline; get_patch() strips it.
        patch_with_nl = patch if patch.endswith("\n") else patch + "\n"
        b64_patch = base64.b64encode(patch_with_nl.encode()).decode()
        b64_eval = base64.b64encode(self._eval_script.encode()).decode()

        apply_chain = " || ".join(
            f'{cmd} "{DOCKER_PATCH}"' for cmd in _GIT_APPLY_CMDS
        )

        eval_cmd = " && ".join([
            f"cd {DOCKER_WORKDIR}",
            "git checkout -- . 2>/dev/null || true",
            "git clean -fd 2>/dev/null || true",
            f"echo '{b64_patch}' | base64 -d > {DOCKER_PATCH}",
            f"( {apply_chain} )",
            f"echo '{b64_eval}' | base64 -d > /tmp/eval.sh",
            "bash /tmp/eval.sh 2>&1",
        ])

        eval_output, _ = await self.execute(eval_cmd, timeout=600)

        prediction = {
            "instance_id": self._instance_id,
            "model_patch": patch,
            "model_name_or_path": "rl-agent",
        }

        try:
            result = grade_instance(
                test_spec=self._test_spec,
                prediction=prediction,
                test_output=eval_output,
            )
            return result.resolved, eval_output
        except Exception as e:
            logger.warning(f"[{self._instance_id}] Grading failed: {e}")
            return False, eval_output

    async def destroy(self) -> None:
        """Delete the Pod."""
        if not self._pod_name:
            return
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                partial(
                    self._core_api.delete_namespaced_pod,
                    name=self._pod_name,
                    namespace=self._namespace,
                    grace_period_seconds=0,
                ),
            )
            logger.debug(f"Deleted Pod {self._pod_name}")
        except k8s_client.ApiException as e:
            logger.warning(f"Failed to delete Pod {self._pod_name}: {e}")
        finally:
            self._pod_name = None
