"""Phase 2 Ray worker: run SWE-bench test execution via K8s Jobs.

Each worker receives a sub-list of predictions and manages K8s Jobs
for test execution. Workers maintain a concurrency window of M
concurrent jobs to balance throughput with cluster capacity.

Workers are stateless with respect to results -- all results are
returned to the head node via ray.get(). The head handles persistence
and resumability to avoid stale/split state across worker pods.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import ray

from .grader import InstanceResult, grade_instance
from .instance_runner import InstanceRunner, JobResult
from swebench.harness.test_spec.test_spec import make_test_spec

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class TestWorker:
    """Runs SWE-bench test evaluation via K8s Jobs.

    Each worker manages a concurrency window of K8s Jobs,
    running M jobs in parallel within a single worker.

    Args:
        k8s_namespace: K8s namespace to create Jobs in.
        timeout: Per-instance timeout in seconds.
        service_account: K8s ServiceAccount for Job pods.
        max_concurrent_jobs: Max K8s Jobs to run in parallel per worker.
        swebench_namespace: DockerHub namespace for pre-built images.
        image_registry: If set, pull images from this registry instead of
            DockerHub.  The swebench namespace prefix (e.g. "swebench/") is
            replaced with "<registry>/<k8s_namespace>/".  For the OpenShift
            internal registry this looks like:
            image-registry.openshift-image-registry.svc:5000/code-agent
    """

    def __init__(
        self,
        k8s_namespace: str | None = None,
        timeout: int = 1800,
        service_account: str = "swe-bench-eval",
        max_concurrent_jobs: int = 4,
        swebench_namespace: str = "swebench",
        image_registry: str | None = None,
    ):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.swebench_namespace = swebench_namespace
        self.image_registry = image_registry

        self.runner = InstanceRunner(
            k8s_namespace=k8s_namespace,
            timeout=timeout,
            service_account=service_account,
        )

    def _evaluate_one(
        self,
        prediction: dict,
        instance_data: dict,
        run_id: str,
    ) -> dict:
        """Evaluate a single instance via a K8s Job.

        Args:
            prediction: Dict with instance_id, model_patch, model_name_or_path.
            instance_data: Full instance data from the dataset.
            run_id: Unique run identifier.

        Returns:
            Result dict with grading information.
        """
        instance_id = prediction["instance_id"]
        model_patch = prediction.get("model_patch", "")

        # Build TestSpec to get eval_script and image name
        test_spec = make_test_spec(
            instance_data,
            namespace=self.swebench_namespace,
        )

        image = test_spec.instance_image_key
        eval_script = test_spec.eval_script

        # Rewrite image ref to point at the internal registry when configured.
        # DockerHub image: swebench/sweb.eval.x86_64.django_1776_django-16938:latest
        # Internal image:  <registry>/sweb.eval.x86_64.django_1776_django-16938:latest
        if self.image_registry:
            # Strip the DockerHub namespace prefix (e.g. "swebench/")
            _, _, image_name = image.partition("/")
            image = f"{self.image_registry}/{image_name}"

        logger.info(f"Evaluating {instance_id} with image {image}")

        # Run the K8s Job
        job_result = self.runner.run_instance(
            instance_id=instance_id,
            run_id=run_id,
            image=image,
            model_patch=model_patch,
            eval_script=eval_script,
        )

        if job_result.error:
            logger.error(f"Job error for {instance_id}: {job_result.error}")
            return {
                "instance_id": instance_id,
                "resolved": False,
                "patch_exists": bool(model_patch),
                "patch_successfully_applied": False,
                "error": job_result.error,
                "tests_status": None,
            }

        if not job_result.test_output:
            logger.warning(f"No test output for {instance_id}")
            error = (
                "K8s Job timed out"
                if job_result.timed_out
                else "No test output captured from pod logs"
            )
            return {
                "instance_id": instance_id,
                "resolved": False,
                "patch_exists": bool(model_patch),
                "patch_successfully_applied": False,
                "error": error,
                "tests_status": None,
            }

        # Grade the result
        grade_result = grade_instance(
            test_spec=test_spec,
            prediction=prediction,
            test_output=job_result.test_output,
        )

        status = "RESOLVED" if grade_result.resolved else "NOT RESOLVED"
        logger.info(f"Instance {instance_id}: {status}")

        return {
            "instance_id": grade_result.instance_id,
            "resolved": grade_result.resolved,
            "patch_exists": grade_result.patch_exists,
            "patch_successfully_applied": grade_result.patch_successfully_applied,
            "error": grade_result.error,
            "tests_status": grade_result.tests_status,
        }

    def evaluate_batch(
        self,
        predictions: list[dict],
        instances_by_id: dict[str, dict],
        run_id: str,
    ) -> list[dict]:
        """Evaluate a batch of predictions with concurrent K8s Jobs.

        Uses a thread pool to manage multiple K8s Jobs in parallel.
        All results are returned to the caller (head node) -- no
        local persistence on the worker.

        Args:
            predictions: List of prediction dicts (instance_id, model_patch, ...).
            instances_by_id: Map of instance_id to full dataset instance.
            run_id: Unique run identifier.

        Returns:
            List of result dicts for all predictions in this batch.
        """
        results = []

        # Run instances with a concurrency window
        with ThreadPoolExecutor(max_workers=self.max_concurrent_jobs) as pool:
            future_to_pred = {}
            for pred in predictions:
                instance_id = pred["instance_id"]
                instance_data = instances_by_id.get(instance_id)

                if instance_data is None:
                    logger.error(f"Instance {instance_id} not found in dataset")
                    results.append({
                        "instance_id": instance_id,
                        "resolved": False,
                        "patch_exists": bool(pred.get("model_patch")),
                        "patch_successfully_applied": False,
                        "error": "Instance not found in dataset",
                        "tests_status": None,
                    })
                    continue

                future = pool.submit(
                    self._evaluate_one,
                    pred,
                    instance_data,
                    run_id,
                )
                future_to_pred[future] = pred

            for future in as_completed(future_to_pred):
                pred = future_to_pred[future]
                instance_id = pred["instance_id"]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Unexpected error for {instance_id}: {e}")
                    result = {
                        "instance_id": instance_id,
                        "resolved": False,
                        "patch_exists": bool(pred.get("model_patch")),
                        "patch_successfully_applied": False,
                        "error": str(e),
                        "tests_status": None,
                    }

                results.append(result)

        return results
