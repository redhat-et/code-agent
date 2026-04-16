"""K8s Job management for evaluating individual SWE-bench instances.

Creates a K8s Job per instance using the pre-built swebench container image,
with the model patch and eval script embedded in the Job command. Test output
is collected from pod logs.

Adapted from https://github.com/MichaelClifford/swe-bench-on-kfp
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Optional

from kubernetes import client as k8s_client
from kubernetes import config as k8s_config

from .script_generator import job_eval_command

logger = logging.getLogger(__name__)

# How often to poll Job status (seconds)
_POLL_INTERVAL = 10


@dataclass
class JobResult:
    """Result of a K8s Job running a single SWE-bench instance."""

    instance_id: str
    succeeded: bool
    timed_out: bool
    test_output: str
    error: str | None = None


def _job_name(instance_id: str, run_id: str) -> str:
    """Generate a unique K8s-safe Job name from instance_id and run_id.

    K8s names must be lowercase, alphanumeric + hyphens, max 63 chars.
    Includes a short hash of the timestamp to avoid collisions with
    leftover Jobs from previous runs.
    """
    safe_id = instance_id.lower().replace("__", "-").replace("_", "-")
    uid = hashlib.md5(f"{instance_id}-{run_id}-{time.time()}".encode()).hexdigest()[:6]
    # Truncate safe_id first to preserve the uid suffix (uniqueness)
    max_base_len = 63 - len("swe--") - len(uid)
    safe_id = safe_id[:max_base_len].strip("-")
    return f"swe-{safe_id}-{uid}"


def _build_job_manifest(
    instance_id: str,
    run_id: str,
    image: str,
    command: list[str],
    namespace: str,
    timeout: int,
    service_account: Optional[str] = None,
) -> k8s_client.V1Job:
    """Build a K8s Job manifest for evaluating a single instance.

    Args:
        instance_id: SWE-bench instance ID.
        run_id: Unique run identifier.
        image: Pre-built swebench container image.
        command: Container command (from job_eval_command).
        namespace: K8s namespace to create the Job in.
        timeout: Job timeout in seconds (activeDeadlineSeconds).
        service_account: Optional K8s ServiceAccount name.

    Returns:
        V1Job manifest ready for creation.
    """
    job_name = _job_name(instance_id, run_id)

    container = k8s_client.V1Container(
        name="eval",
        image=image,
        image_pull_policy="IfNotPresent",
        command=command,
        working_dir="/testbed",
        resources=k8s_client.V1ResourceRequirements(
            requests={"cpu": "500m", "memory": "2Gi"},
            limits={"cpu": "2", "memory": "4Gi"},
        ),
    )

    pod_spec = k8s_client.V1PodSpec(
        containers=[container],
        restart_policy="Never",
        service_account_name=service_account,
    )

    template = k8s_client.V1PodTemplateSpec(
        metadata=k8s_client.V1ObjectMeta(
            labels={
                "app": "swe-bench-eval",
                "instance-id": instance_id[:63],
                "run-id": run_id[:63],
            },
        ),
        spec=pod_spec,
    )

    job_spec = k8s_client.V1JobSpec(
        template=template,
        backoff_limit=0,
        active_deadline_seconds=timeout,
        ttl_seconds_after_finished=600,
    )

    job = k8s_client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=k8s_client.V1ObjectMeta(
            name=job_name,
            namespace=namespace,
            labels={
                "app": "swe-bench-eval",
                "instance-id": instance_id[:63],
                "run-id": run_id[:63],
            },
        ),
        spec=job_spec,
    )

    return job


def _detect_namespace() -> str:
    """Detect the current K8s namespace when running in-cluster.

    Falls back to 'default' when running locally.
    """
    ns_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    try:
        with open(ns_path) as f:
            return f.read().strip()
    except FileNotFoundError:
        return "default"


class InstanceRunner:
    """Manages K8s Jobs for evaluating SWE-bench instances."""

    def __init__(
        self,
        k8s_namespace: Optional[str] = None,
        timeout: int = 1800,
        service_account: Optional[str] = None,
    ):
        """Initialize the runner.

        Args:
            k8s_namespace: K8s namespace to create Jobs in.
                           Auto-detected from in-cluster config if not set.
            timeout: Per-instance timeout in seconds (default 30 min).
            service_account: Optional K8s ServiceAccount for the Job pods.
        """
        self.k8s_namespace = k8s_namespace or _detect_namespace()
        self.timeout = timeout
        self.service_account = service_account

        # Load K8s config -- in-cluster when running in a pod,
        # or from kubeconfig when running locally
        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()

        self.batch_api = k8s_client.BatchV1Api()
        self.core_api = k8s_client.CoreV1Api()

    def create_job(
        self,
        instance_id: str,
        run_id: str,
        image: str,
        model_patch: str,
        eval_script: str,
    ) -> str:
        """Create a K8s Job for a single SWE-bench instance.

        Args:
            instance_id: SWE-bench instance ID.
            run_id: Unique run identifier.
            image: Pre-built swebench container image.
            model_patch: The model's prediction patch.
            eval_script: The eval script from TestSpec.eval_script.

        Returns:
            The Job name.
        """
        command = job_eval_command(model_patch, eval_script)

        job = _build_job_manifest(
            instance_id=instance_id,
            run_id=run_id,
            image=image,
            command=command,
            namespace=self.k8s_namespace,
            timeout=self.timeout,
            service_account=self.service_account,
        )

        self.batch_api.create_namespaced_job(
            namespace=self.k8s_namespace,
            body=job,
        )

        job_name = job.metadata.name
        logger.info(f"Created Job {job_name} for instance {instance_id}")
        return job_name

    def wait_for_job(self, job_name: str) -> tuple[bool, bool]:
        """Wait for a K8s Job to complete.

        Args:
            job_name: Name of the Job to wait for.

        Returns:
            Tuple of (succeeded: bool, timed_out: bool).
        """
        while True:
            job = self.batch_api.read_namespaced_job(
                name=job_name,
                namespace=self.k8s_namespace,
            )

            if job.status.succeeded and job.status.succeeded > 0:
                return True, False

            if job.status.failed and job.status.failed > 0:
                # Check if failure was due to timeout
                conditions = job.status.conditions or []
                timed_out = any(
                    c.type == "Failed" and c.reason == "DeadlineExceeded"
                    for c in conditions
                )
                return False, timed_out

            time.sleep(_POLL_INTERVAL)

    def get_pod_logs(self, job_name: str, retries: int = 5) -> str:
        """Get logs from the pod created by a Job.

        Retries a few times if the pod isn't found or logs aren't
        available yet -- there can be a brief delay between a Job
        completing and its pod logs being retrievable.

        Args:
            job_name: Name of the Job.
            retries: Number of retry attempts.

        Returns:
            Pod log output as a string.
        """
        for attempt in range(retries):
            pods = self.core_api.list_namespaced_pod(
                namespace=self.k8s_namespace,
                label_selector=f"job-name={job_name}",
            )

            if not pods.items:
                if attempt < retries - 1:
                    logger.debug(
                        f"No pods found for job {job_name}, "
                        f"retrying ({attempt + 1}/{retries})"
                    )
                    time.sleep(3)
                    continue
                return ""

            pod_name = pods.items[0].metadata.name

            try:
                logs = self.core_api.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=self.k8s_namespace,
                )
                if logs:
                    return logs
                # Empty logs -- pod may still be terminating, retry
                if attempt < retries - 1:
                    logger.debug(
                        f"Empty logs for pod {pod_name}, "
                        f"retrying ({attempt + 1}/{retries})"
                    )
                    time.sleep(3)
                    continue
                return ""
            except k8s_client.ApiException as e:
                if attempt < retries - 1:
                    logger.debug(
                        f"Failed to read logs for pod {pod_name}: {e}, "
                        f"retrying ({attempt + 1}/{retries})"
                    )
                    time.sleep(3)
                    continue
                logger.error(f"Failed to read logs for pod {pod_name}: {e}")
                return ""

        return ""

    def delete_job(self, job_name: str) -> None:
        """Delete a Job and its pods.

        Args:
            job_name: Name of the Job to delete.
        """
        try:
            self.batch_api.delete_namespaced_job(
                name=job_name,
                namespace=self.k8s_namespace,
                propagation_policy="Foreground",
            )
            logger.info(f"Deleted Job {job_name}")
        except k8s_client.ApiException as e:
            logger.warning(f"Failed to delete Job {job_name}: {e}")

    def run_instance(
        self,
        instance_id: str,
        run_id: str,
        image: str,
        model_patch: str,
        eval_script: str,
    ) -> JobResult:
        """Run a full evaluation for a single SWE-bench instance.

        Creates a K8s Job, waits for completion, collects pod logs,
        and cleans up the Job.

        Args:
            instance_id: SWE-bench instance ID.
            run_id: Unique run identifier.
            image: Pre-built swebench container image.
            model_patch: The model's prediction patch.
            eval_script: The eval script from TestSpec.eval_script.

        Returns:
            JobResult with test output and status.
        """
        job_name = None
        try:
            job_name = self.create_job(
                instance_id=instance_id,
                run_id=run_id,
                image=image,
                model_patch=model_patch,
                eval_script=eval_script,
            )

            succeeded, timed_out = self.wait_for_job(job_name)
            test_output = self.get_pod_logs(job_name)

            return JobResult(
                instance_id=instance_id,
                succeeded=succeeded,
                timed_out=timed_out,
                test_output=test_output,
            )

        except Exception as e:
            logger.error(f"Error running instance {instance_id}: {e}")
            return JobResult(
                instance_id=instance_id,
                succeeded=False,
                timed_out=False,
                test_output="",
                error=str(e),
            )

        finally:
            if job_name:
                self.delete_job(job_name)
