"""
Base verifier interface.

All verifiers implement this interface. A verifier takes a patch and repo
context, runs a check, and returns a VerifierResult with a score and metadata.

VerifierStatus only signals execution errors (ERROR, TIMEOUT, SKIPPED).
Pass/fail semantics are derived from score >= pass_threshold, not from status.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Literal


class VerifierStatus(Enum):
    OK = "ok"           # ran successfully; consult score for pass/fail
    ERROR = "error"     # verifier itself crashed
    TIMEOUT = "timeout" # verifier timed out
    SKIPPED = "skipped" # not applicable (e.g., no relevant files changed)


@dataclass
class VerifierResult:
    """Result from a single verifier run."""
    name: str
    status: VerifierStatus
    score: float                          # normalized to [0.0, 1.0]
    pass_threshold: float = 1.0           # score must meet this to be considered passing
    wall_clock_seconds: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    # Raw output for debugging
    stdout: str = ""
    stderr: str = ""

    @property
    def passed(self) -> bool:
        """True only when the verifier ran successfully and score >= pass_threshold."""
        return self.status == VerifierStatus.OK and self.score >= self.pass_threshold

    def __repr__(self) -> str:
        return (
            f"VerifierResult(name={self.name!r}, status={self.status.value}, "
            f"score={self.score:.3f}, threshold={self.pass_threshold:.3f}, "
            f"passed={self.passed}, time={self.wall_clock_seconds:.1f}s)"
        )


@dataclass
class PatchContext:
    """Everything a verifier needs to check a patch."""
    patch_diff: str                        # the raw unified diff
    changed_files: list[str]               # list of files modified by the patch
    task_id: str                           # benchmark task identifier
    repo_path: Path | None = None          # path to the repo with patch applied (dynamic verifiers)
    test_cmd: str | None = None            # repo-specific test command
    ground_truth_patch: str | None = None  # for differential comparison
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVerifier(ABC):
    """Abstract base class for all verifiers.

    Subclasses must define:
      - execution_mode (ClassVar): "static" or "dynamic"
          static  — runs inline in the calling process; works from patch_diff alone
          dynamic — requires an isolated execution environment (e.g. K8s Job)
      - name (property): unique verifier identifier
      - verify (coroutine): core verification logic
    """

    execution_mode: ClassVar[Literal["static", "dynamic"]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only enforce on concrete (non-abstract) subclasses
        if not getattr(cls, "__abstractmethods__", None):
            mode = getattr(cls, "execution_mode", None)
            if mode not in {"static", "dynamic"}:
                raise TypeError(
                    f"{cls.__name__} must define an 'execution_mode' class attribute "
                    f"(Literal['static', 'dynamic'])"
                )

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        timeout: float = 60.0,
        pass_threshold: float = 1.0,
    ):
        self.config = config or {}
        self.timeout = timeout

        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError(
                f"pass_threshold must be in [0.0, 1.0], got {pass_threshold}"
            )
        self.pass_threshold = pass_threshold

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this verifier."""
        ...

    @abstractmethod
    async def verify(self, ctx: PatchContext) -> VerifierResult:
        """
        Run the verification check.

        Args:
            ctx: The patch context. Static verifiers use patch_diff/changed_files.
                 Dynamic verifiers may also use repo_path and ctx.metadata.

        Returns:
            VerifierResult with status OK and a score, or ERROR/TIMEOUT/SKIPPED.
        """
        ...

    def format_feedback(self, result: VerifierResult) -> str:
        """Convert a VerifierResult into a human-readable feedback string.

        Called by the multi-turn loop to build the feedback message shown to
        the model before its next attempt. Subclasses override to provide
        verifier-specific detail (error locations, failing tests, etc.).
        """
        if result.status != VerifierStatus.OK:
            return f"[{self.name}] {result.status.value.upper()}"
        status_label = "PASSED" if result.passed else "FAILED"
        return f"[{self.name}] {status_label} (score: {result.score:.2f})"

    async def safe_verify(self, ctx: PatchContext) -> VerifierResult:
        """
        Run verify() with timeout and error handling.
        Subclasses should not override this.
        """
        import asyncio

        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self.verify(ctx),
                timeout=self.timeout
            )
            result.pass_threshold = self.pass_threshold
            result.wall_clock_seconds = time.monotonic() - start
            return result
        except asyncio.TimeoutError:
            return VerifierResult(
                name=self.name,
                status=VerifierStatus.TIMEOUT,
                score=0.0,
                pass_threshold=self.pass_threshold,
                wall_clock_seconds=time.monotonic() - start,
                details={"timeout": self.timeout},
            )
        except Exception as e:
            return VerifierResult(
                name=self.name,
                status=VerifierStatus.ERROR,
                score=0.0,
                pass_threshold=self.pass_threshold,
                wall_clock_seconds=time.monotonic() - start,
                details={"error": str(e), "error_type": type(e).__name__},
            )
