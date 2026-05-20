"""
AST validity verifier (static).

Checks that all changed Python files introduced by the patch parse successfully.
Works entirely from the patch diff — no repo checkout required.

For each changed .py file, new content is reconstructed by taking the context
lines (unchanged) and added lines (+) from the unified diff. This is a heuristic:
deletions create gaps, so the reconstructed content may differ from the true new
file. However, syntax errors introduced by the model (missing colons, unclosed
brackets, bad indentation in new code) will reliably surface.
"""

from __future__ import annotations

import ast
import re
from typing import Any, ClassVar, Literal

from .base import BaseVerifier, PatchContext, VerifierResult, VerifierStatus


def _extract_new_content(patch_diff: str, filepath: str) -> str | None:
    """Reconstruct the new content of a file from a unified diff.

    Collects context lines (prefix ' ') and added lines (prefix '+'),
    skipping hunk headers and deleted lines.

    Returns None if the file is not found in the diff (e.g. pure deletion).
    """
    lines = patch_diff.splitlines()
    in_file = False
    in_hunk = False
    new_lines: list[str] = []
    found = False

    for line in lines:
        # Detect file header (unified diff: +++ b/path or +++ path)
        if line.startswith("+++ "):
            candidate = line[4:]
            if candidate.startswith("b/"):
                candidate = candidate[2:]
            # Match by suffix to handle path prefixes
            in_file = candidate == filepath or candidate.endswith("/" + filepath)
            if in_file:
                found = True
                new_lines = []
            in_hunk = False
            continue

        if line.startswith("--- "):
            continue

        if in_file:
            if line.startswith("@@"):
                in_hunk = True
                continue
            if not in_hunk:
                continue
            if line.startswith("+") and not line.startswith("+++"):
                new_lines.append(line[1:])   # added line
            elif line.startswith(" "):
                new_lines.append(line[1:])   # context line
            elif line.startswith("-"):
                pass                          # deleted line — skip
            elif line.startswith("\\"):
                pass                          # "No newline at end of file"
            else:
                # Next file section begins
                in_file = False
                in_hunk = False

    if not found:
        return None
    return "\n".join(new_lines)


class ASTCheckVerifier(BaseVerifier):
    """Static verifier: checks Python syntax of all changed files in the patch."""

    execution_mode: ClassVar[Literal["static", "dynamic"]] = "static"

    @property
    def name(self) -> str:
        return "ast_check"

    def format_feedback(self, result: VerifierResult) -> str:
        if result.status != VerifierStatus.OK:
            return f"[{self.name}] {result.status.value.upper()}"
        if result.passed:
            return f"[{self.name}] PASSED: no syntax errors detected"
        errors = result.details.get("errors", [])
        lines = [f"[{self.name}] FAILED: syntax errors in the following files:"]
        for e in errors:
            loc = f"line {e['line']}" if e.get("line") else ""
            offset = f", offset {e['offset']}" if e.get("offset") else ""
            msg = e.get("message", "syntax error")
            lines.append(f"  - {e.get('file', '?')}: {loc}{offset}: {msg}")
        return "\n".join(lines)

    async def verify(self, ctx: PatchContext) -> VerifierResult:
        python_files = [f for f in ctx.changed_files if f.endswith(".py")]

        if not python_files:
            return VerifierResult(
                name=self.name,
                status=VerifierStatus.SKIPPED,
                score=1.0,
                pass_threshold=self.pass_threshold,
                details={"message": "No Python files changed"},
            )

        errors: list[dict[str, Any]] = []
        parsed = 0
        skipped = 0  # files not present in the diff (e.g. pure deletions)

        for filepath in python_files:
            content = _extract_new_content(ctx.patch_diff, filepath)

            if content is None:
                # File was deleted by the patch — nothing to parse
                skipped += 1
                continue

            if not content.strip():
                # Empty new content (file fully cleared) — counts as parseable
                parsed += 1
                continue

            try:
                ast.parse(content, filename=filepath)
                parsed += 1
            except SyntaxError as e:
                errors.append({
                    "file": filepath,
                    "line": e.lineno,
                    "offset": e.offset,
                    "message": e.msg,
                })

        checkable = parsed + len(errors)
        score = parsed / checkable if checkable > 0 else 1.0

        return VerifierResult(
            name=self.name,
            status=VerifierStatus.OK,
            score=score,
            pass_threshold=self.pass_threshold,
            details={
                "errors": errors,
                "files_total": len(python_files),
                "files_checkable": checkable,
                "files_parsed": parsed,
                "files_skipped": skipped,
                "files_errored": len(errors),
            },
        )
