"""
AST validity verifier (static).

Checks that all changed Python files introduced by the patch parse successfully.
Works entirely from the patch diff — no repo checkout required.

For each changed .py file, new content is reconstructed by taking the context
lines (unchanged) and added lines (+) from the unified diff. This is a heuristic:
deletions create gaps, so the reconstructed content may differ from the true new
file. However, syntax errors introduced by the model (missing colons, unclosed
brackets, bad indentation in new code) will reliably surface.

Parsing strategy (applied per file, in order):
  1. Try ast.parse() on the raw reconstructed fragment.
  2. If that fails with an indentation error on line 1 (fragment starts mid-block),
     apply textwrap.dedent() and retry.
  3. If that still fails with an indentation error on line 1 (mixed indentation
     levels after dedent), wrap in "async def _():\\n    if True:\\n" and retry.
  4. A failure at any stage with a non-indentation error, or a failure of the
     wrapped version, is reported as a real syntax error.

Line numbers in reported errors are adjusted to refer to the reconstructed
fragment, not to any wrapper lines added during parsing.

Error feedback includes a short code snippet showing the lines around the error,
with '+' prefixes on added lines and ' ' prefixes on context lines.
"""

from __future__ import annotations

import ast
import textwrap
from typing import Any, ClassVar, Literal

from .base import BaseVerifier, PatchContext, VerifierResult, VerifierStatus

# Number of lines prepended by the wrapper in step 3.
_WRAPPER = "async def _():\n    if True:\n"
_WRAPPER_LINES = _WRAPPER.count("\n")
_WRAPPER_INDENT = "        "  # 8 spaces (2 levels of 4)

# Lines of context to show before/after the error line in feedback.
_SNIPPET_BEFORE = 2
_SNIPPET_AFTER = 1


def _extract_new_content(patch_diff: str, filepath: str) -> list[tuple[str, str]] | None:
    """Reconstruct the new content of a file from a unified diff.

    Collects context lines (prefix ' ') and added lines (prefix '+'),
    skipping hunk headers and deleted lines.

    Returns a list of (prefix, text) tuples where prefix is '+' (added line)
    or ' ' (context line), or None if the file is not found in the diff
    (e.g. pure deletion).
    """
    lines = patch_diff.splitlines()
    in_file = False
    in_hunk = False
    line_records: list[tuple[str, str]] = []
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
                line_records = []
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
                line_records.append(("+", line[1:]))  # added line
            elif line.startswith(" "):
                line_records.append((" ", line[1:]))  # context line
            elif line.startswith("-"):
                pass                                   # deleted line — skip
            elif line.startswith("\\"):
                pass                                   # "No newline at end of file"
            else:
                # Next file section begins
                in_file = False
                in_hunk = False

    if not found:
        return None
    return line_records


def _records_to_content(line_records: list[tuple[str, str]]) -> str:
    """Join line records into a plain content string for ast.parse()."""
    return "\n".join(text for _, text in line_records)


def _is_indentation_error_on_line_1(exc: SyntaxError) -> bool:
    """True if the exception is an indentation problem on the very first line."""
    if exc.lineno != 1:
        return False
    if isinstance(exc, IndentationError):
        return True
    msg = exc.msg.lower()
    return "indent" in msg


def _try_parse(content: str, filename: str) -> SyntaxError | None:
    """Attempt ast.parse(); return the SyntaxError or None on success."""
    try:
        ast.parse(content, filename=filename)
        return None
    except SyntaxError as e:
        return e


def _build_snippet(line_records: list[tuple[str, str]], error_lineno: int | None) -> str | None:
    """Build a short code snippet around the error line for display in feedback.

    Line numbers are 1-based and refer to the reconstructed fragment.
    Returns None if error_lineno is None or out of range.
    """
    if error_lineno is None or not (1 <= error_lineno <= len(line_records)):
        return None

    start = max(0, error_lineno - 1 - _SNIPPET_BEFORE)
    end = min(len(line_records), error_lineno + _SNIPPET_AFTER)

    snippet_lines = []
    for i in range(start, end):
        lineno = i + 1
        prefix, text = line_records[i]
        marker = "-->" if lineno == error_lineno else "   "
        snippet_lines.append(f"    {marker} {lineno:4d} {prefix} {text}")

    return "\n".join(snippet_lines)


def _parse_fragment(
    line_records: list[tuple[str, str]],
    filepath: str,
) -> dict[str, Any] | None:
    """Parse a reconstructed diff fragment using the multi-stage strategy.

    Returns an error dict {file, line, offset, message, snippet} if a real
    syntax error is found, or None if the fragment parses successfully at
    some stage.
    """
    content = _records_to_content(line_records)

    def _make_error(lineno: int | None, offset: int | None, msg: str) -> dict[str, Any]:
        return {
            "file": filepath,
            "line": lineno,
            "offset": offset,
            "message": msg,
            "snippet": _build_snippet(line_records, lineno),
        }

    # Stage 1: raw content
    err = _try_parse(content, filepath)
    if err is None:
        return None
    if not _is_indentation_error_on_line_1(err):
        return _make_error(err.lineno, err.offset, err.msg)

    # Stage 2: dedented content
    dedented = textwrap.dedent(content)
    err = _try_parse(dedented, filepath)
    if err is None:
        return None
    if not _is_indentation_error_on_line_1(err):
        return _make_error(err.lineno, err.offset, err.msg)

    # Stage 3: wrap in "async def _():\n    if True:\n"
    wrapped = _WRAPPER + textwrap.indent(dedented, _WRAPPER_INDENT)
    err = _try_parse(wrapped, filepath)
    if err is None:
        return None
    # Adjust line number back to the fragment (subtract wrapper lines)
    lineno = err.lineno - _WRAPPER_LINES if err.lineno is not None else None
    return _make_error(lineno, err.offset, err.msg)


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
            if e.get("snippet"):
                lines.append(e["snippet"])
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
            line_records = _extract_new_content(ctx.patch_diff, filepath)

            if line_records is None:
                # File was deleted by the patch — nothing to parse
                skipped += 1
                continue

            content = _records_to_content(line_records)
            if not content.strip():
                # Empty new content (file fully cleared) — counts as parseable
                parsed += 1
                continue

            error = _parse_fragment(line_records, filepath)
            if error is None:
                parsed += 1
            else:
                errors.append(error)

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
