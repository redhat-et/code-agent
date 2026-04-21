"""Common grading utilities for evaluation pipelines.

Provides a base AggregateReport class that can be inherited by
benchmark-specific aggregate reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaseAggregateReport:
    """Base class for aggregate evaluation reports.

    Provides common structure for counting passed/failed/error instances,
    computing pass rates, and tracking IDs.

    Subclasses should add benchmark-specific fields.
    """

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    empty: int = 0
    pass_rate: float = 0.0
    passed_ids: list[str] = field(default_factory=list)
    failed_ids: list[str] = field(default_factory=list)
    error_ids: list[str] = field(default_factory=list)

    def finalize(self) -> None:
        """Compute derived metrics. Call after populating counts."""
        if self.total > 0:
            self.pass_rate = self.passed / self.total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Subclasses should override and call super().to_dict() to
        include benchmark-specific fields.
        """
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "empty": self.empty,
            "pass_rate": self.pass_rate,
            "passed_ids": self.passed_ids,
            "failed_ids": self.failed_ids,
            "error_ids": self.error_ids,
        }
