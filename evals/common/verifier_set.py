"""
VerifierSet: a configured collection of verifiers with scheduling metadata.

Each verifier in the set carries two flags:
  - run_intermediate: include this verifier after each generation turn
  - run_final:        include this verifier in the final evaluation pass

The set dispatches inline (static) vs. via an isolated environment (dynamic)
based on whether any member verifier has weight == "dynamic".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verifiers.base import BaseVerifier, VerifierResult


@dataclass
class VerifierEntry:
    """A verifier paired with its scheduling configuration."""
    verifier: BaseVerifier
    run_intermediate: bool = True
    run_final: bool = True


class VerifierSet:
    """An ordered collection of verifiers with intermediate/final scheduling.

    Args:
        entries: List of VerifierEntry objects.
    """

    def __init__(self, entries: list[VerifierEntry] | None = None):
        self.entries: list[VerifierEntry] = entries or []

    def add(
        self,
        verifier: BaseVerifier,
        run_intermediate: bool = True,
        run_final: bool = True,
    ) -> "VerifierSet":
        """Add a verifier with scheduling flags. Returns self for chaining."""
        self.entries.append(VerifierEntry(verifier, run_intermediate, run_final))
        return self

    def intermediate_subset(self) -> "VerifierSet":
        """Return a new VerifierSet containing only intermediate-scheduled verifiers."""
        return VerifierSet([e for e in self.entries if e.run_intermediate])

    def final_subset(self) -> "VerifierSet":
        """Return a new VerifierSet containing only final-scheduled verifiers."""
        return VerifierSet([e for e in self.entries if e.run_final])

    @property
    def verifiers(self) -> list[BaseVerifier]:
        return [e.verifier for e in self.entries]

    @property
    def is_dynamic(self) -> bool:
        """True if any verifier in this set requires isolated execution."""
        return any(e.verifier.execution_mode == "dynamic" for e in self.entries)

    def all_passed(self, results: list[VerifierResult]) -> bool:
        """True if every result in the list passes its threshold.

        Used for early-exit decisions: if all intermediate verifiers pass,
        there is no need to generate another turn.
        """
        return all(r.passed for r in results)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        names = [e.verifier.name for e in self.entries]
        return f"VerifierSet({names})"
