"""
Score aggregation strategies for multi-verifier evaluation.

Each aggregator combines a list of (verifier_name, score) pairs into a single
aggregate score in [0.0, 1.0]. Verifiers that did not run successfully
(ERROR, TIMEOUT, SKIPPED) contribute a score of 0.0 unless excluded.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from verifiers.base import VerifierResult, VerifierStatus


class ScoreAggregator(ABC):
    """Abstract base for score aggregation strategies."""

    @abstractmethod
    def aggregate(self, results: list[VerifierResult]) -> float:
        """
        Combine verifier results into a single aggregate score.

        Args:
            results: List of VerifierResult from one evaluation pass.

        Returns:
            Aggregate score in [0.0, 1.0].
        """
        ...

    @staticmethod
    def _effective_score(result: VerifierResult) -> float:
        """Return the result's score, or 0.0 for non-OK statuses."""
        if result.status != VerifierStatus.OK:
            return 0.0
        return result.score


class MeanAggregator(ScoreAggregator):
    """Simple mean across all verifier scores (equal weights).

    Non-OK results contribute 0.0.
    Returns 1.0 if the result list is empty.
    """

    def aggregate(self, results: list[VerifierResult]) -> float:
        if not results:
            return 1.0
        return sum(self._effective_score(r) for r in results) / len(results)


class MinAggregator(ScoreAggregator):
    """Minimum score across all verifiers (any failure dominates).

    Useful when every check must pass independently.
    Non-OK results contribute 0.0.
    Returns 1.0 if the result list is empty.
    """

    def aggregate(self, results: list[VerifierResult]) -> float:
        if not results:
            return 1.0
        return min(self._effective_score(r) for r in results)


class WeightedSumAggregator(ScoreAggregator):
    """Weighted sum of verifier scores, normalized to [0.0, 1.0].

    Weights are keyed by verifier name. Any verifier not in the weights
    dict defaults to weight 1.0. Non-OK results contribute 0.0.

    Args:
        weights: Map of verifier name to non-negative weight.

    Example config::

        WeightedSumAggregator(weights={"ast_check": 0.2, "swe_test": 0.8})
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {}

    def aggregate(self, results: list[VerifierResult]) -> float:
        if not results:
            return 1.0

        total_weight = 0.0
        weighted_sum = 0.0

        for result in results:
            w = self.weights.get(result.name, 1.0)
            if w < 0:
                raise ValueError(
                    f"Weight for '{result.name}' must be non-negative, got {w}"
                )
            weighted_sum += self._effective_score(result) * w
            total_weight += w

        if total_weight == 0.0:
            return 1.0

        return weighted_sum / total_weight


def build_aggregator(name: str, config: dict[str, Any] | None = None) -> ScoreAggregator:
    """Construct a named aggregator from a config dict.

    Supported names: "mean", "min", "weighted_sum".
    For "weighted_sum": config may contain {"weights": {"verifier_name": float, ...}}
    """
    cfg = config or {}
    if name == "mean":
        return MeanAggregator()
    if name == "min":
        return MinAggregator()
    if name == "weighted_sum":
        return WeightedSumAggregator(weights=cfg.get("weights"))
    raise ValueError(f"Unknown aggregator '{name}'. Choose from: mean, min, weighted_sum")

