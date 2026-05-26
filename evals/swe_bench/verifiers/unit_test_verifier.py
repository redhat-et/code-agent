"""
SWE-bench unit test verifier (dynamic).

Runs the official SWE-bench unit test suite for a single instance via a K8s Job,
grades the result, and returns a VerifierResult.

This verifier requires:
  ctx.metadata["instance_data"]: full dataset instance dict (for make_test_spec)
  ctx.metadata["run_id"]:        unique run identifier (for K8s Job naming)
  ctx.metadata["runner"]:        InstanceRunner instance (injected by SWEBenchMultiTurnWorker)

The K8s Job applies the patch and runs the eval script inside the pre-built
SWE-bench container image. The result is scored as 1.0 (resolved) or 0.0 (not).
"""

from __future__ import annotations

import logging
from typing import ClassVar, Literal

from swebench.harness.test_spec.test_spec import make_test_spec

from evals.swe_bench.grader import grade_instance
from evals.swe_bench.instance_runner import InstanceRunner
from verifiers.base import BaseVerifier, PatchContext, VerifierResult, VerifierStatus

logger = logging.getLogger(__name__)


class SWEBenchUnitTestVerifier(BaseVerifier):
    """Dynamic verifier: runs the SWE-bench unit test suite via a K8s Job.

    Args:
        swebench_namespace: DockerHub namespace for pre-built images.
        image_registry: If set, pull from this internal registry instead of DockerHub.
        pass_threshold: Score threshold for pass/fail (default 1.0 = must fully resolve).
        timeout: Per-instance job timeout in seconds.
    """

    execution_mode: ClassVar[Literal["static", "dynamic"]] = "dynamic"

    def __init__(
        self,
        swebench_namespace: str = "swebench",
        image_registry: str | None = None,
        pass_threshold: float = 1.0,
        timeout: float = 1800.0,
    ):
        super().__init__(timeout=timeout, pass_threshold=pass_threshold)
        self.swebench_namespace = swebench_namespace
        self.image_registry = image_registry

    @property
    def name(self) -> str:
        return "swe_test"

    def format_feedback(self, result: VerifierResult) -> str:
        if result.status != VerifierStatus.OK:
            error = result.details.get("error", "unknown error")
            return f"[{self.name}] {result.status.value.upper()}: {error}"
        if result.passed:
            return f"[{self.name}] PASSED: issue resolved"
        lines = [f"[{self.name}] FAILED (score: {result.score:.2f})"]
        details = result.details
        if not details.get("patch_successfully_applied"):
            lines.append("  The patch could not be applied to the repository.")
            return "\n".join(lines)
        tests_status = details.get("tests_status") or {}
        failed_tests = [
            name for name, status in tests_status.items()
            if status in ("FAILED", "ERROR")
        ]
        if failed_tests:
            lines.append(f"  Failing tests ({len(failed_tests)}):")
            for t in failed_tests[:20]:
                lines.append(f"    - {t}")
            if len(failed_tests) > 20:
                lines.append(f"    ... and {len(failed_tests) - 20} more")
        stdout = result.stdout or ""
        if stdout and len(stdout) < 4000:
            lines.append("\n  Test output:\n  " + stdout.replace("\n", "\n  "))
        elif stdout:
            tail = stdout[-3000:]
            lines.append("\n  Test output (last 3000 chars):\n  " + tail.replace("\n", "\n  "))
        return "\n".join(lines)

    async def verify(self, ctx: PatchContext) -> VerifierResult:
        instance_data = ctx.metadata["instance_data"]
        run_id = ctx.metadata["run_id"]
        runner: InstanceRunner = ctx.metadata["runner"]

        instance_id = ctx.task_id
        model_patch = ctx.patch_diff

        # Build TestSpec to get eval_script and image
        test_spec = make_test_spec(instance_data, namespace=self.swebench_namespace)
        image = test_spec.instance_image_key
        eval_script = test_spec.eval_script

        # Rewrite image ref if using an internal registry
        if self.image_registry:
            _, _, image_name = image.partition("/")
            if image_name.endswith(":latest"):
                image_name = image_name.removesuffix(":latest") + ":v1"
            else:
                logger.warning(
                    f"Expected :latest tag for {instance_id}, got {image_name} -- using as-is"
                )
            image = f"{self.image_registry}/{image_name}"

        logger.info(f"[{instance_id}] Running K8s Job with image {image}")

        try:
            job_result = runner.run_instance(
                instance_id=instance_id,
                run_id=run_id,
                image=image,
                model_patch=model_patch,
                eval_script=eval_script,
            )
        except Exception as e:
            logger.exception("[%s] unit-test job execution failed", instance_id)
            return VerifierResult(
                name=self.name,
                status=VerifierStatus.ERROR,
                score=0.0,
                pass_threshold = self.pass_threshold,
                details = {"error": str(e)},
            )

        if job_result.error:
            return VerifierResult(
                name=self.name,
                status=VerifierStatus.ERROR,
                score=0.0,
                pass_threshold=self.pass_threshold,
                details={"error": job_result.error},
            )

        if not job_result.test_output:
            error = (
                "K8s Job timed out"
                if job_result.timed_out
                else "No test output captured from pod logs"
            )
            return VerifierResult(
                name=self.name,
                status=VerifierStatus.ERROR,
                score=0.0,
                pass_threshold=self.pass_threshold,
                details={"error": error},
            )

        prediction = {
            "instance_id": instance_id,
            "model_patch": model_patch,
            "model_name_or_path": ctx.metadata.get("model_name", "unknown"),
        }

        try:
            grade_result = grade_instance(
                test_spec=test_spec,
                prediction=prediction,
                test_output=job_result.test_output,
            )
        except Exception as e:
            logger.exception("[%s] grading failed", instance_id)
            return VerifierResult(
                name=self.name,
                status=VerifierStatus.ERROR,
                score=0.0,
                pass_threshold=self.pass_threshold,
                details={"error": str(e)},
                stdout=job_result.test_output,
            )

        score = 1.0 if grade_result.resolved else 0.0
        status = "RESOLVED" if grade_result.resolved else "NOT RESOLVED"
        logger.info(f"[{instance_id}] {status} (score={score:.1f})")

        return VerifierResult(
            name=self.name,
            status=VerifierStatus.OK,
            score=score,
            pass_threshold=self.pass_threshold,
            stdout=job_result.test_output,
            details={
                "resolved": grade_result.resolved,
                "patch_exists": grade_result.patch_exists,
                "patch_successfully_applied": grade_result.patch_successfully_applied,
                "tests_status": grade_result.tests_status,
                "error": grade_result.error,
            },
        )
