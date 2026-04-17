"""Grade SWE-bench evaluation results.

Wraps swebench's grading logic to parse test output captured from pod logs,
grade individual instances, and aggregate results across a full evaluation run.
All operations are in-memory -- no PVC or file I/O required (except a temp file
needed internally by swebench's get_eval_report).

Adapted from https://github.com/MichaelClifford/swe-bench-on-kfp
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import TestSpec


@dataclass
class InstanceResult:
    """Result of evaluating a single SWE-bench instance."""

    instance_id: str
    resolved: bool
    patch_exists: bool
    patch_successfully_applied: bool
    error: str | None = None
    tests_status: dict[str, Any] | None = None


@dataclass
class AggregateReport:
    """Aggregate report across all evaluated instances."""

    total_instances: int = 0
    resolved_instances: int = 0
    unresolved_instances: int = 0
    error_instances: int = 0
    empty_patch_instances: int = 0
    resolve_rate: float = 0.0
    resolved_ids: list[str] = field(default_factory=list)
    unresolved_ids: list[str] = field(default_factory=list)
    error_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_instances": self.total_instances,
            "resolved_instances": self.resolved_instances,
            "unresolved_instances": self.unresolved_instances,
            "error_instances": self.error_instances,
            "empty_patch_instances": self.empty_patch_instances,
            "resolve_rate": self.resolve_rate,
            "resolved_ids": self.resolved_ids,
            "unresolved_ids": self.unresolved_ids,
            "error_ids": self.error_ids,
        }


def grade_instance(
    test_spec: TestSpec,
    prediction: dict[str, str],
    test_output: str,
) -> InstanceResult:
    """Grade a single SWE-bench instance from its test output.

    Args:
        test_spec: The TestSpec for this instance.
        prediction: Dict with 'instance_id', 'model_name_or_path', and 'model_patch'.
        test_output: Raw test output text captured from pod logs.

    Returns:
        InstanceResult with grading details.
    """
    instance_id = prediction["instance_id"]
    temp_path = None

    try:
        # get_eval_report requires a file path, so write to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(test_output)
            temp_path = f.name

        report = get_eval_report(
            test_spec=test_spec,
            prediction=prediction,
            test_log_path=temp_path,
            include_tests_status=True,
        )

        instance_report = report[instance_id]

        return InstanceResult(
            instance_id=instance_id,
            resolved=instance_report["resolved"],
            patch_exists=instance_report["patch_exists"],
            patch_successfully_applied=instance_report["patch_successfully_applied"],
            tests_status=instance_report.get("tests_status"),
        )

    except Exception as e:
        return InstanceResult(
            instance_id=instance_id,
            resolved=False,
            patch_exists=bool(prediction.get("model_patch")),
            patch_successfully_applied=False,
            error=str(e),
        )
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def aggregate_reports(results: list[InstanceResult]) -> AggregateReport:
    """Aggregate individual instance results into a summary report.

    Args:
        results: List of InstanceResult from grading individual instances.

    Returns:
        AggregateReport with totals, resolve rate, and ID lists.
    """
    report = AggregateReport(total_instances=len(results))

    for result in results:
        if result.error is not None:
            report.error_instances += 1
            report.unresolved_instances += 1
            report.error_ids.append(result.instance_id)
            report.unresolved_ids.append(result.instance_id)
        elif not result.patch_exists:
            report.empty_patch_instances += 1
            report.unresolved_instances += 1
            report.unresolved_ids.append(result.instance_id)
        elif result.resolved:
            report.resolved_instances += 1
            report.resolved_ids.append(result.instance_id)
        else:
            report.unresolved_instances += 1
            report.unresolved_ids.append(result.instance_id)

    if report.total_instances > 0:
        report.resolve_rate = report.resolved_instances / report.total_instances

    return report
