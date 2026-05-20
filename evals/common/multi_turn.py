"""
Multi-turn evaluation session.

Manages the iterative generate → evaluate → feedback loop for one instance.
All benchmark-specific logic (inference, evaluation dispatch, output extraction)
is injected via callables so this class remains benchmark-agnostic.

Loop flow per instance:
    1. Call generate_fn(messages) → raw model response
    2. Call extract_fn(response) → structured output string
    3. Call run_intermediate_fn(output) → list[VerifierResult]
    4. Compute aggregate score; check early-exit condition
    5. If not exiting: collect feedback from each verifier and append
       user message; go to 1
    After max_turns (or early exit):
    6. Call run_final_fn(last_output) → list[VerifierResult]
    7. Return MultiTurnResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from evals.common.score_aggregator import ScoreAggregator
from evals.common.verifier_set import VerifierSet
from verifiers.base import VerifierResult

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """Recorded state for a single generation turn."""
    turn: int                                  # 0-based turn index
    output: str                                # extracted output from model response
    verifier_results: list[VerifierResult]     # intermediate verifier results
    aggregate_score: float                     # aggregated intermediate score
    feedback: str                              # feedback message sent back to model
                                               # (empty string on last turn / early exit)


@dataclass
class MultiTurnResult:
    """Full result for one multi-turn evaluation instance."""
    final_output: str                          # output from the last generation turn
    final_verifier_results: list[VerifierResult]
    final_aggregate_score: float
    turns: list[TurnResult]                    # per-turn intermediate records
    stopped_early: bool                        # True if intermediate eval passed before max_turns

    @property
    def num_turns(self) -> int:
        return len(self.turns)


class MultiTurnSession:
    """Runs the multi-turn generate → evaluate → feedback loop for one instance.

    Args:
        generate_fn: Callable(messages: list[dict]) -> str.
            Calls the model with the current conversation and returns the raw response.
        extract_fn: Callable(response: str) -> str.
            Extracts the structured output (patch, solution, etc.) from the raw response.
        run_intermediate_fn: Callable(output: str) -> list[VerifierResult].
            Runs the intermediate verifier set. May be a no-op if the set is empty.
        run_final_fn: Callable(output: str) -> list[VerifierResult].
            Runs the final verifier set on the last output.
        aggregator: Combines verifier results into a scalar score.
        intermediate_verifier_set: Used for early-exit checking and feedback generation.
            Each verifier in the set is responsible for formatting its own feedback.
        max_turns: Hard cap on generation attempts.
    """

    def __init__(
        self,
        generate_fn: Callable[[list[dict]], str],
        extract_fn: Callable[[str], str],
        run_intermediate_fn: Callable[[str], list[VerifierResult]],
        run_final_fn: Callable[[str], list[VerifierResult]],
        aggregator: ScoreAggregator,
        intermediate_verifier_set: VerifierSet,
        max_turns: int = 1,
    ):
        self.generate_fn = generate_fn
        self.extract_fn = extract_fn
        self.run_intermediate_fn = run_intermediate_fn
        self.run_final_fn = run_final_fn
        self.aggregator = aggregator
        self.intermediate_verifier_set = intermediate_verifier_set
        self.max_turns = max_turns

    def _build_feedback(self, results: list[VerifierResult]) -> str:
        """Collect per-verifier feedback and wrap in a single message."""
        lines = ["Your solution was evaluated. Here is the feedback:\n"]
        for entry in self.intermediate_verifier_set.entries:
            result = next((r for r in results if r.name == entry.verifier.name), None)
            if result is not None:
                lines.append(entry.verifier.format_feedback(result))
        if all(r.passed for r in results):
            lines.append("\nAll checks passed.")
        else:
            lines.append("\nPlease revise your solution to address the issues above.")
        return "\n".join(lines)

    def run(self, initial_messages: list[dict]) -> MultiTurnResult:
        """Execute the full multi-turn loop for one instance.

        Args:
            initial_messages: Starting conversation (system + first user message).

        Returns:
            MultiTurnResult with the last output, final eval results, and turn history.
        """
        messages = list(initial_messages)
        turns: list[TurnResult] = []
        output = ""
        stopped_early = False

        for turn_idx in range(self.max_turns):
            logger.debug(f"Turn {turn_idx + 1}/{self.max_turns}")

            # Generate
            response = self.generate_fn(messages)
            output = self.extract_fn(response)

            # Append assistant turn to conversation
            messages.append({"role": "assistant", "content": response})

            # Intermediate evaluation (skipped when max_turns == 1)
            intermediate_results: list[VerifierResult] = []
            aggregate_score = 1.0
            feedback = ""

            if self.intermediate_verifier_set:
                intermediate_results = self.run_intermediate_fn(output)
                aggregate_score = self.aggregator.aggregate(intermediate_results)

                # Early exit: all intermediate verifiers passed
                if self.intermediate_verifier_set.all_passed(intermediate_results):
                    logger.debug(
                        f"Turn {turn_idx + 1}: all intermediate checks passed "
                        f"(score={aggregate_score:.3f}), stopping early"
                    )
                    turns.append(TurnResult(
                        turn=turn_idx,
                        output=output,
                        verifier_results=intermediate_results,
                        aggregate_score=aggregate_score,
                        feedback="",
                    ))
                    stopped_early = True
                    break

                # Not the last turn: generate feedback and continue
                if turn_idx < self.max_turns - 1:
                    feedback = self._build_feedback(intermediate_results)
                    messages.append({"role": "user", "content": feedback})

            turns.append(TurnResult(
                turn=turn_idx,
                output=output,
                verifier_results=intermediate_results,
                aggregate_score=aggregate_score,
                feedback=feedback,
            ))

        # Final evaluation on the last output
        logger.debug("Running final evaluation")
        final_results = self.run_final_fn(output)
        final_score = self.aggregator.aggregate(final_results)

        return MultiTurnResult(
            final_output=output,
            final_verifier_results=final_results,
            final_aggregate_score=final_score,
            turns=turns,
            stopped_early=stopped_early,
        )
