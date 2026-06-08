"""Ray worker for running agents on SWE-bench instances via K8s Jobs.

Creates a K8s Job per instance using the correct SWE-bench container image.
The agent to run is defined by an AgentConfig YAML file, making it trivially
swappable between mini-swe-agent, OpenCode, or any other agent.

Each Job runs four phases inside the container:
  1. Install and run the agent (produces a patch)
  2. Extract the patch (preds.json or git diff)
  3. Extract the trajectory (if available)
  4. Apply the patch and run the SWE-bench eval script (test grading)

Grading happens inside the Ray worker after collecting pod logs,
using swebench's grade_instance() to parse the test output.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import ray
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from swebench.harness.constants import DOCKER_PATCH, DOCKER_WORKDIR
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec

from .agent_config import AgentConfig, render_template
from .grader import grade_instance

logger = logging.getLogger(__name__)

# Delimiters for extracting data from pod logs
PREDS_START = "<<PREDS_JSON_START_7829361>>"
PREDS_END = "<<PREDS_JSON_END_7829361>>"
PATCH_START = "<<PATCH_START_7829361>>"
PATCH_END = "<<PATCH_END_7829361>>"
EVAL_START = "<<EVAL_OUTPUT_START_7829361>>"
EVAL_END = "<<EVAL_OUTPUT_END_7829361>>"
TRAJ_START = "<<TRAJ_JSON_START_7829361>>"
TRAJ_END = "<<TRAJ_JSON_END_7829361>>"

# Heredoc delimiters for inline scripts
_PATCH_HEREDOC = "EOF_MODEL_PATCH_7829361"
_EVAL_HEREDOC = "EOF_EVAL_SCRIPT_4819253"

# Patch apply strategies (same as swebench uses)
_GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

# How often to poll K8s Job status (seconds)
_POLL_INTERVAL = 10


@dataclass
class InstanceInfo:
    """Resolved info for a SWE-bench instance."""
    image: str
    test_spec: TestSpec
    eval_script: str


# ── Helpers (module-level, used by both the Ray actor and generate_job_yaml) ──

def _job_name(instance_id: str, run_id: str) -> str:
    """Generate a unique K8s-safe Job name."""
    safe_id = instance_id.lower().replace("__", "-").replace("_", "-")
    uid = hashlib.md5(
        f"{instance_id}-{run_id}-{time.time()}".encode()
    ).hexdigest()[:6]
    max_base_len = 63 - len("swe--") - len(uid)
    safe_id = safe_id[:max_base_len].strip("-")
    return f"swe-{safe_id}-{uid}"


def resolve_instance(
    instance: dict,
    image_registry: str = "",
    swebench_namespace: str = "swebench",
) -> InstanceInfo:
    """Resolve the SWE-bench image and eval script for an instance."""
    test_spec = make_test_spec(instance, namespace=swebench_namespace)
    image = test_spec.instance_image_key
    eval_script = test_spec.eval_script

    if image_registry:
        _, _, image_name = image.partition("/")
        if image_name.endswith(":latest"):
            image_name = image_name.removesuffix(":latest") + ":v1"
        image = f"{image_registry}/{image_name}"

    return InstanceInfo(image=image, test_spec=test_spec, eval_script=eval_script)


def _build_eval_step(eval_script: str) -> str:
    """Build the shell commands to apply the patch and run the eval script."""
    apply_lines = []
    for i, cmd in enumerate(_GIT_APPLY_CMDS):
        keyword = "if" if i == 0 else "elif"
        apply_lines.append(f'{keyword} {cmd} "{DOCKER_PATCH}"; then')
        apply_lines.append('    echo ">>>>> Applied Patch"')
    apply_lines.append("else")
    apply_lines.append('    echo ">>>>> Patch Apply Failed"')
    apply_lines.append(f'    echo "{EVAL_START}"')
    apply_lines.append('    echo "PATCH_APPLY_FAILED"')
    apply_lines.append(f'    echo "{EVAL_END}"')
    apply_lines.append("    exit 0")
    apply_lines.append("fi")

    return "\n".join([
        "",
        "# ── Evaluation: apply patch and run tests ──",
        f"cd {DOCKER_WORKDIR}",
        "git checkout -- . 2>/dev/null || true",
        "git clean -fd 2>/dev/null || true",
        "",
        f"cp /tmp/model_patch.txt {DOCKER_PATCH}",
        "",
        "\n".join(apply_lines),
        "",
        f"cat > /tmp/eval.sh << '{_EVAL_HEREDOC}'",
        eval_script,
        _EVAL_HEREDOC,
        "",
        f'echo "{EVAL_START}"',
        "/bin/bash /tmp/eval.sh",
        f'echo "{EVAL_END}"',
    ])


def _build_job_command(
    agent_config: AgentConfig,
    instance: dict,
    template_vars: dict[str, str],
    eval_script: str = "",
    run_eval: bool = True,
    disable_thinking: bool = False,
) -> list[str]:
    """Build the shell command for the K8s Job container."""
    install_cmd = render_template(agent_config.install_command, **template_vars).strip()
    agent_cmd = render_template(agent_config.agent_command, **template_vars).strip()

    if disable_thinking:
        agent_cmd += ' -c \'model.model_kwargs.extra_body={"chat_template_kwargs": {"enable_thinking": false}}\''

    extraction = agent_config.patch_extraction
    if extraction.startswith("preds_json:"):
        preds_path = extraction.split(":", 1)[1]
        extract_step = "\n".join([
            f'echo "{PREDS_START}"',
            f'cat {preds_path}',
            f'echo "{PREDS_END}"',
            "",
            "# Extract raw patch from preds.json for eval step",
            f"python3 -c \"import json; d=json.load(open('{preds_path}')); "
            f"v=next(iter(d.values())); print(v.get('model_patch',''))\" "
            f"> /tmp/model_patch.txt",
        ])
    elif extraction == "git_diff":
        extract_step = "\n".join([
            f'cd {DOCKER_WORKDIR}',
            'git diff > /tmp/model_patch.txt',
            f'echo "{PATCH_START}"',
            'cat /tmp/model_patch.txt',
            f'echo "{PATCH_END}"',
        ])
    elif extraction.startswith("file:"):
        file_path = extraction.split(":", 1)[1]
        extract_step = "\n".join([
            f'cp {file_path} /tmp/model_patch.txt',
            f'echo "{PATCH_START}"',
            'cat /tmp/model_patch.txt',
            f'echo "{PATCH_END}"',
        ])
    else:
        raise ValueError(f"Unknown patch_extraction method: {extraction}")

    problem_setup = ""
    if not agent_config.needs_swebench_dataset:
        problem_setup = "\n".join([
            "cat > /tmp/problem_statement.txt << 'EOF_PROBLEM_STMT_9182736'",
            instance.get("problem_statement", ""),
            "EOF_PROBLEM_STMT_9182736",
        ])

    env_exports = []
    for key, val_template in agent_config.env.items():
        rendered_val = render_template(val_template, **template_vars)
        env_exports.append(f'export {key}="{rendered_val}"')

    eval_step = ""
    if run_eval and eval_script:
        eval_step = _build_eval_step(eval_script)

    script = "\n".join(filter(None, [
        'export HOME=/tmp',
        'export PATH="$HOME/.local/bin:$PATH"',
        f'git config --global --add safe.directory {DOCKER_WORKDIR}',
        "",
        "\n".join(env_exports) if env_exports else None,
        "",
        problem_setup if problem_setup else None,
        "",
        f"# ── Phase 1: Install agent ({agent_config.name}) ──",
        install_cmd if install_cmd else None,
        "",
        f"# ── Phase 2: Run agent ({agent_config.name}) ──",
        agent_cmd,
        "",
        "# ── Phase 3: Extract prediction ──",
        extract_step,
        "",
        "# ── Phase 3b: Extract trajectory ──",
        f'TRAJ_FILE=$(find /tmp/output -name "*.traj.json" 2>/dev/null | head -1)',
        f'if [ -n "$TRAJ_FILE" ]; then',
        f'  echo "{TRAJ_START}"',
        f'  cat "$TRAJ_FILE"',
        f'  echo "{TRAJ_END}"',
        f'fi',
        "",
        eval_step if eval_step else None,
    ]))

    return ["/bin/bash", "-c", script]


def _build_job_manifest(
    instance_id: str,
    run_id: str,
    image: str,
    command: list[str],
    namespace: str,
    agent_config: AgentConfig,
    service_account: Optional[str] = None,
) -> k8s_client.V1Job:
    """Build a K8s Job manifest for running an agent on a single instance."""
    job_name = _job_name(instance_id, run_id)
    resources = agent_config.resources

    init_container = k8s_client.V1Container(
        name="fix-permissions",
        image=image,
        image_pull_policy="IfNotPresent",
        command=["/bin/sh", "-c",
                 f"chown -R 1001:0 {DOCKER_WORKDIR} && "
                 f"chmod -R a+rwX {DOCKER_WORKDIR} && "
                 f"find {DOCKER_WORKDIR} -type f ! -perm -666 -exec chmod 666 {{}} + && "
                 "chown -R 1001:0 /opt/miniconda3 2>/dev/null || true && "
                 "chmod -R a+rwX /opt/miniconda3 2>/dev/null || true"],
        security_context=k8s_client.V1SecurityContext(
            allow_privilege_escalation=False,
            run_as_user=0,
            run_as_non_root=False,
            capabilities=k8s_client.V1Capabilities(drop=["ALL"]),
            privileged=False,
        ),
    )

    container = k8s_client.V1Container(
        name="agent",
        image=image,
        image_pull_policy="IfNotPresent",
        command=command,
        working_dir=DOCKER_WORKDIR,
        security_context=k8s_client.V1SecurityContext(
            allow_privilege_escalation=False,
            run_as_non_root=True,
            run_as_user=1001,
            capabilities=k8s_client.V1Capabilities(drop=["ALL"]),
        ),
        resources=k8s_client.V1ResourceRequirements(
            requests=resources.get("requests", {}),
            limits=resources.get("limits", {}),
        ),
    )

    pod_spec = k8s_client.V1PodSpec(
        init_containers=[init_container],
        containers=[container],
        restart_policy="Never",
        service_account_name=service_account,
        automount_service_account_token=False,
        security_context=k8s_client.V1PodSecurityContext(
            run_as_non_root=True,
            fs_group=0,
            supplemental_groups=[0],
        ),
    )

    template = k8s_client.V1PodTemplateSpec(
        metadata=k8s_client.V1ObjectMeta(
            labels={
                "app": "swe-bench-agent",
                "instance-id": instance_id[:63],
                "run-id": run_id[:63],
            },
        ),
        spec=pod_spec,
    )

    job_spec = k8s_client.V1JobSpec(
        template=template,
        backoff_limit=0,
        active_deadline_seconds=agent_config.job_timeout,
        ttl_seconds_after_finished=600,
    )

    return k8s_client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=k8s_client.V1ObjectMeta(
            name=job_name,
            namespace=namespace,
            labels={
                "app": "swe-bench-agent",
                "instance-id": instance_id[:63],
                "run-id": run_id[:63],
            },
        ),
        spec=job_spec,
    )


# ── Log parsing ──────────────────────────────────────────────────────

def extract_prediction_from_logs(
    logs: str, instance_id: str, model_name: str, patch_extraction: str,
) -> dict:
    """Extract the prediction dict from pod logs."""
    if patch_extraction.startswith("preds_json:"):
        return _extract_preds_json(logs, instance_id, model_name)
    else:
        patch = _extract_raw_patch(logs)
        return {
            "instance_id": instance_id,
            "model_patch": patch,
            "model_name_or_path": model_name,
        }


def extract_eval_output(logs: str) -> str:
    """Extract eval/test output from between EVAL_START/EVAL_END delimiters."""
    try:
        start_idx = logs.index(EVAL_START) + len(EVAL_START)
        end_idx = logs.index(EVAL_END)
        return logs[start_idx:end_idx].strip()
    except ValueError:
        return ""


def extract_trajectory(logs: str) -> str:
    """Extract trajectory JSON from between TRAJ_START/TRAJ_END delimiters."""
    try:
        start_idx = logs.index(TRAJ_START) + len(TRAJ_START)
        end_idx = logs.index(TRAJ_END)
        return logs[start_idx:end_idx].strip()
    except ValueError:
        return ""


def _extract_preds_json(logs: str, instance_id: str, model_name: str) -> dict:
    """Extract preds.json content from between PREDS_START/PREDS_END delimiters."""
    try:
        start_idx = logs.index(PREDS_START) + len(PREDS_START)
        end_idx = logs.index(PREDS_END)
        json_str = logs[start_idx:end_idx].strip()
        preds = json.loads(json_str)

        if instance_id in preds:
            pred = preds[instance_id]
            return {
                "instance_id": pred["instance_id"],
                "model_patch": pred.get("model_patch", ""),
                "model_name_or_path": pred.get("model_name_or_path", model_name),
            }
        if len(preds) == 1:
            pred = next(iter(preds.values()))
            return {
                "instance_id": pred.get("instance_id", instance_id),
                "model_patch": pred.get("model_patch", ""),
                "model_name_or_path": pred.get("model_name_or_path", model_name),
            }

        logger.warning(f"Instance {instance_id} not found in preds.json from logs")
    except (ValueError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse preds.json from logs for {instance_id}: {e}")

    return {"instance_id": instance_id, "model_patch": "", "model_name_or_path": model_name}


def _extract_raw_patch(logs: str) -> str:
    """Extract raw patch text from between PATCH_START/PATCH_END delimiters."""
    try:
        start_idx = logs.index(PATCH_START) + len(PATCH_START)
        end_idx = logs.index(PATCH_END)
        return logs[start_idx:end_idx].strip()
    except ValueError:
        logger.warning("Could not find patch delimiters in logs")
        return ""


# ── K8s Job lifecycle (used internally by the Ray actor) ─────────────

def _detect_namespace() -> str:
    ns_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    try:
        with open(ns_path) as f:
            return f.read().strip()
    except FileNotFoundError:
        return "default"


def _init_k8s():
    """Load K8s config and return API clients."""
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()
    return k8s_client.BatchV1Api(), k8s_client.CoreV1Api()


def _wait_for_job(
    batch_api: k8s_client.BatchV1Api, job_name: str, namespace: str,
) -> tuple[bool, bool]:
    """Wait for a K8s Job to complete. Returns (succeeded, timed_out)."""
    consecutive_errors = 0
    max_errors = 10

    while True:
        try:
            job = batch_api.read_namespaced_job(name=job_name, namespace=namespace)
            consecutive_errors = 0

            if job.status.succeeded and job.status.succeeded > 0:
                return True, False
            if job.status.failed and job.status.failed > 0:
                conditions = job.status.conditions or []
                timed_out = any(
                    c.type == "Failed" and c.reason == "DeadlineExceeded"
                    for c in conditions
                )
                return False, timed_out

        except k8s_client.ApiException as e:
            consecutive_errors += 1
            logger.warning(
                f"API error polling Job {job_name} "
                f"({consecutive_errors}/{max_errors}): {e.reason}"
            )
            if consecutive_errors >= max_errors:
                raise RuntimeError(
                    f"Too many consecutive API errors polling Job {job_name}"
                ) from e

        time.sleep(_POLL_INTERVAL)


def _get_pod_logs(
    core_api: k8s_client.CoreV1Api, job_name: str, namespace: str, retries: int = 5,
) -> str:
    """Get logs from the pod created by a Job."""
    for attempt in range(retries):
        pods = core_api.list_namespaced_pod(
            namespace=namespace, label_selector=f"job-name={job_name}",
        )
        if not pods.items:
            if attempt < retries - 1:
                time.sleep(3)
                continue
            return ""

        pod_name = pods.items[0].metadata.name
        try:
            logs = core_api.read_namespaced_pod_log(name=pod_name, namespace=namespace)
            if logs:
                return logs
            if attempt < retries - 1:
                time.sleep(3)
                continue
            return ""
        except k8s_client.ApiException as e:
            if attempt < retries - 1:
                time.sleep(3)
                continue
            logger.error(f"Failed to read logs for pod {pod_name}: {e}")
            return ""

    return ""


def _delete_job(
    batch_api: k8s_client.BatchV1Api, job_name: str, namespace: str,
) -> None:
    """Delete a Job and its pods."""
    try:
        batch_api.delete_namespaced_job(
            name=job_name, namespace=namespace, propagation_policy="Foreground",
        )
        logger.debug(f"Deleted Job {job_name}")
    except k8s_client.ApiException as e:
        logger.warning(f"Failed to delete Job {job_name}: {e}")


# ── Ray actor ────────────────────────────────────────────────────────

@ray.remote(num_cpus=1)
class AgentWorker:
    """Ray actor that generates patches using an agent via K8s Jobs.

    Each worker manages concurrent K8s Jobs using a ThreadPoolExecutor.
    Grading (parsing eval test output) happens inside the worker.
    """

    def __init__(
        self,
        agent_config_dict: dict,
        model_name: str,
        vllm_urls: list[str],
        model_api_key: str = "dummy",
        k8s_namespace: str | None = None,
        service_account: str = "swe-bench-eval",
        image_registry: str = "",
        swebench_namespace: str = "swebench",
        subset: str = "verified",
        split: str = "test",
        step_limit: int = 100,
        cost_limit: float = 3.0,
        max_concurrent_jobs: int = 4,
        job_timeout: int = 0,
        run_eval: bool = True,
        disable_thinking: bool = False,
    ):
        self.agent_config = AgentConfig(**agent_config_dict)
        # CLI --job-timeout overrides the agent config default (0 = use config)
        if job_timeout > 0:
            self.agent_config.job_timeout = job_timeout
        self.model_name = model_name
        if not vllm_urls:
            raise ValueError("vllm_urls must contain at least one endpoint")
        self._vllm_urls = vllm_urls
        self._call_count = 0
        self.model_api_key = model_api_key
        self.k8s_namespace = k8s_namespace or _detect_namespace()
        self.service_account = service_account
        self.image_registry = image_registry
        self.swebench_namespace = swebench_namespace
        self.subset = subset
        self.split = split
        self.step_limit = step_limit
        self.cost_limit = cost_limit
        self.max_concurrent_jobs = max_concurrent_jobs
        self.run_eval = run_eval
        self.disable_thinking = disable_thinking

        self.batch_api, self.core_api = _init_k8s()

    def _get_vllm_url(self) -> str:
        url = self._vllm_urls[self._call_count % len(self._vllm_urls)]
        self._call_count += 1
        return url

    def _generate_one(self, instance: dict, run_id: str) -> dict:
        """Run an agent on a single SWE-bench instance via a K8s Job.

        Returns a result dict with prediction, eval grading, logs, and trajectory.
        """
        instance_id = instance["instance_id"]
        job_name = None

        try:
            info = resolve_instance(
                instance, self.image_registry, self.swebench_namespace
            )
            logger.info(f"[{instance_id}] Using image: {info.image}")

            template_vars = {
                "instance_id": instance_id,
                "model_name": self.model_name,
                "model_base_url": self._get_vllm_url(),
                "model_api_key": self.model_api_key,
                "workdir": DOCKER_WORKDIR,
                "problem_statement_file": "/tmp/problem_statement.txt",
                "subset": self.subset,
                "split": self.split,
                "step_limit": str(self.step_limit),
                "cost_limit": str(self.cost_limit),
            }

            command = _build_job_command(
                agent_config=self.agent_config,
                instance=instance,
                template_vars=template_vars,
                eval_script=info.eval_script,
                run_eval=self.run_eval,
                disable_thinking=self.disable_thinking,
            )
            job = _build_job_manifest(
                instance_id=instance_id,
                run_id=run_id,
                image=info.image,
                command=command,
                namespace=self.k8s_namespace,
                agent_config=self.agent_config,
                service_account=self.service_account,
            )

            self.batch_api.create_namespaced_job(
                namespace=self.k8s_namespace, body=job
            )
            job_name = job.metadata.name
            logger.info(f"[{instance_id}] Created Job {job_name}")

            succeeded, timed_out = _wait_for_job(
                self.batch_api, job_name, self.k8s_namespace
            )
            if timed_out:
                logger.warning(f"[{instance_id}] Job timed out")

            logs = _get_pod_logs(self.core_api, job_name, self.k8s_namespace)

            # Extract prediction
            prediction = extract_prediction_from_logs(
                logs=logs,
                instance_id=instance_id,
                model_name=self.model_name,
                patch_extraction=self.agent_config.patch_extraction,
            )

            # Extract eval output and grade
            eval_output = extract_eval_output(logs) if self.run_eval else ""
            eval_report = None
            resolved = None

            if self.run_eval and eval_output and info.test_spec:
                try:
                    grade_result = grade_instance(
                        test_spec=info.test_spec,
                        prediction=prediction,
                        test_output=eval_output,
                    )
                    resolved = grade_result.resolved
                    eval_report = {
                        "resolved": grade_result.resolved,
                        "patch_exists": grade_result.patch_exists,
                        "patch_successfully_applied": grade_result.patch_successfully_applied,
                        "error": grade_result.error,
                    }
                    status = "RESOLVED" if grade_result.resolved else "NOT RESOLVED"
                    logger.info(f"[{instance_id}] Eval: {status}")
                except Exception as e:
                    logger.warning(f"[{instance_id}] Grading failed: {e}")

            # Extract trajectory
            traj_json = extract_trajectory(logs)

            return {
                "instance_id": instance_id,
                "model_patch": prediction.get("model_patch", ""),
                "model_name_or_path": prediction.get("model_name_or_path", self.model_name),
                "eval_report": eval_report,
                "resolved": resolved,
                "trajectory": traj_json,
                "full_logs": logs,
                "error": "timed_out" if timed_out else (
                    None if succeeded else "job_failed"
                ),
            }

        except Exception as e:
            logger.error(f"[{instance_id}] Error: {e}")
            return {
                "instance_id": instance_id,
                "model_patch": "",
                "model_name_or_path": self.model_name,
                "eval_report": None,
                "resolved": None,
                "trajectory": "",
                "full_logs": "",
                "error": str(e),
            }

        finally:
            if job_name:
                _delete_job(self.batch_api, job_name, self.k8s_namespace)

    def generate_patches(self, instances: list[dict], run_id: str) -> list[dict]:
        """Generate patches for a batch of instances with concurrent K8s Jobs.

        Args:
            instances: List of SWE-bench dataset instances.
            run_id: Unique run identifier.

        Returns:
            List of result dicts for all instances in this batch.
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.max_concurrent_jobs) as pool:
            future_to_id = {}
            for instance in instances:
                future = pool.submit(self._generate_one, instance, run_id)
                future_to_id[future] = instance["instance_id"]

            for future in as_completed(future_to_id):
                instance_id = future_to_id[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"[{instance_id}] Unexpected error: {e}")
                    result = {
                        "instance_id": instance_id,
                        "model_patch": "",
                        "model_name_or_path": self.model_name,
                        "eval_report": None,
                        "resolved": None,
                        "trajectory": "",
                        "full_logs": "",
                        "error": str(e),
                    }
                results.append(result)

        return results


# ── Debug helper ─────────────────────────────────────────────────────

def generate_job_yaml(
    instance: dict,
    run_id: str,
    agent_config: AgentConfig,
    model_name: str,
    model_base_url: str,
    model_api_key: str = "dummy",
    k8s_namespace: str = "default",
    service_account: str | None = "swe-bench-eval",
    image_registry: str = "",
    swebench_namespace: str = "swebench",
    subset: str = "verified",
    split: str = "test",
    step_limit: int = 100,
    cost_limit: float = 3.0,
    run_eval: bool = True,
) -> str:
    """Generate the K8s Job manifest as YAML without submitting it.

    Useful for debugging -- inspect the YAML before applying to the cluster.
    """
    instance_id = instance["instance_id"]
    info = resolve_instance(instance, image_registry, swebench_namespace)

    template_vars = {
        "instance_id": instance_id,
        "model_name": model_name,
        "model_base_url": model_base_url,
        "model_api_key": model_api_key,
        "workdir": DOCKER_WORKDIR,
        "problem_statement_file": "/tmp/problem_statement.txt",
        "subset": subset,
        "split": split,
        "step_limit": str(step_limit),
        "cost_limit": str(cost_limit),
    }

    command = _build_job_command(
        agent_config=agent_config,
        instance=instance,
        template_vars=template_vars,
        eval_script=info.eval_script,
        run_eval=run_eval,
    )

    job = _build_job_manifest(
        instance_id=instance_id,
        run_id=run_id,
        image=info.image,
        command=command,
        namespace=k8s_namespace,
        agent_config=agent_config,
        service_account=service_account,
    )

    from kubernetes.client import ApiClient
    job_dict = ApiClient().sanitize_for_serialization(job)

    import yaml
    return yaml.dump(job_dict, default_flow_style=False, sort_keys=False)
