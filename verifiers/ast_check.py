"""
AST validity verifier (static).

Checks that all changed Python files introduced by the patch parse successfully.

Two parsing strategies, tried in order:

  1. **Full-file parsing** (preferred): when SWE-bench metadata is available
     (repo name and base commit), performs a shallow sparse clone of the
     repository, checks out the base commit, applies the patch, and runs
     ast.parse() on each complete modified Python file.  This eliminates
     the false positives inherent in fragment-based parsing.

  2. **Hunk-based parsing** (fallback): when repo metadata is unavailable or
     the clone fails, falls back to parsing each unified-diff hunk in
     isolation.  This works entirely from the patch diff string with no I/O.

Hunk-based parsing strategy (applied per hunk, in order):
  1. Try ast.parse() on the raw reconstructed fragment.
  2. If that fails with an indentation error on line 1 (fragment starts mid-block),
     apply textwrap.dedent() and retry.
  3. If that still fails with an indentation error on line 1 (mixed indentation
     levels after dedent), wrap in "async def _():\\n    if True:\\n" and retry.
  4. A failure at any stage with a non-indentation error, or a failure of the
     wrapped version, is reported as a real syntax error.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
import shutil
import tempfile
import textwrap
from typing import Any, ClassVar, Literal

from .base import BaseVerifier, PatchContext, VerifierResult, VerifierStatus

logger = logging.getLogger(__name__)

# Number of lines prepended by the wrapper in step 3.
_WRAPPER = "async def _():\n    if True:\n"
_WRAPPER_LINES = _WRAPPER.count("\n")
_WRAPPER_INDENT = "        "  # 8 spaces (2 levels of 4)

# Lines of context to show before/after the error line in feedback.
_SNIPPET_BEFORE = 2
_SNIPPET_AFTER = 1

_HUNK_HEADER_RE = re.compile(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


# ── Full-file parsing (preferred) ──────────────────────────────────────


async def _run_git(*args: str, cwd: str | None = None) -> tuple[int, str, str]:
    """Run a git command asynchronously. Returns (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode(), stderr.decode()


async def _clone_and_parse(
    repo: str,
    base_commit: str,
    patch_diff: str,
    python_files: list[str],
    verifier_name: str,
    pass_threshold: float,
) -> VerifierResult | None:
    """Clone the repo, apply the patch, and parse full Python files.

    Returns a VerifierResult on success, or None if the git operations fail
    (signaling the caller to fall back to hunk-based parsing).
    """
    tmpdir = tempfile.mkdtemp(prefix="ast_check_")
    repo_dir = os.path.join(tmpdir, "repo")

    try:
        url = f"https://github.com/{repo}.git"

        rc, _, stderr = await _run_git(
            "clone", "--depth", "1", "--filter=blob:none",
            "--sparse", url, repo_dir,
        )
        if rc != 0:
            logger.warning("ast_check: git clone failed: %s", stderr.strip())
            return None

        rc, _, stderr = await _run_git(
            "sparse-checkout", "set", "--skip-checks", *python_files, cwd=repo_dir,
        )
        if rc != 0:
            logger.warning("ast_check: sparse-checkout failed: %s", stderr.strip())
            return None

        rc, _, stderr = await _run_git(
            "fetch", "--depth", "1", "origin", base_commit, cwd=repo_dir,
        )
        if rc != 0:
            logger.warning("ast_check: git fetch failed: %s", stderr.strip())
            return None

        rc, _, stderr = await _run_git(
            "checkout", base_commit, cwd=repo_dir,
        )
        if rc != 0:
            logger.warning("ast_check: git checkout failed: %s", stderr.strip())
            return None

        patch_file = os.path.join(tmpdir, "patch.diff")
        with open(patch_file, "w") as f:
            f.write(patch_diff)

        rc, _, stderr = await _run_git(
            "apply", patch_file, cwd=repo_dir,
        )
        if rc != 0:
            logger.warning("ast_check: git apply failed: %s", stderr.strip())
            return None

        errors: list[dict[str, Any]] = []
        parsed = 0
        errored = 0
        skipped = 0

        for filepath in python_files:
            full_path = os.path.join(repo_dir, filepath)
            if not os.path.isfile(full_path):
                skipped += 1
                continue

            try:
                with open(full_path) as f:
                    content = f.read()
            except OSError:
                skipped += 1
                continue

            try:
                ast.parse(content, filename=filepath)
                parsed += 1
            except SyntaxError as e:
                errored += 1
                snippet = _build_full_file_snippet(content, e.lineno)
                errors.append({
                    "file": filepath,
                    "line": e.lineno,
                    "offset": e.offset,
                    "message": e.msg,
                    "snippet": snippet,
                })

        checkable = parsed + errored
        score = parsed / checkable if checkable > 0 else 1.0

        return VerifierResult(
            name=verifier_name,
            status=VerifierStatus.OK,
            score=score,
            pass_threshold=pass_threshold,
            details={
                "errors": errors,
                "method": "full_file",
                "files_total": len(python_files),
                "files_checkable": checkable,
                "files_parsed": parsed,
                "files_skipped": skipped,
                "files_errored": errored,
            },
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _build_full_file_snippet(content: str, error_lineno: int | None) -> str | None:
    """Build a code snippet around the error line from full file content."""
    if error_lineno is None:
        return None

    lines = content.splitlines()
    if not (1 <= error_lineno <= len(lines)):
        return None

    start = max(0, error_lineno - 1 - _SNIPPET_BEFORE)
    end = min(len(lines), error_lineno + _SNIPPET_AFTER)

    snippet_lines = []
    for i in range(start, end):
        lineno = i + 1
        marker = "-->" if lineno == error_lineno else "   "
        snippet_lines.append(f"    {marker} {lineno:4d}   {lines[i]}")

    return "\n".join(snippet_lines)


# ── Hunk-based parsing (fallback) ──────────────────────────────────────


def _parse_hunk_new_start(header: str) -> int:
    """Extract the new-file start line from a unified diff hunk header."""
    m = _HUNK_HEADER_RE.search(header)
    return int(m.group(1)) if m else 1


def _extract_new_content(
    patch_diff: str, filepath: str,
) -> list[tuple[int, list[tuple[str, str]]]] | None:
    """Extract per-hunk new content from a unified diff.

    Each hunk is returned as (new_start, records) where new_start is the
    1-based starting line in the new file (from the @@ header) and records
    is a list of (prefix, text) tuples ('+' for added, ' ' for context).

    Returns None if the file is not found in the diff (e.g. pure deletion).
    """
    lines = patch_diff.splitlines()
    in_file = False
    in_hunk = False
    found = False
    hunks: list[tuple[int, list[tuple[str, str]]]] = []
    current_records: list[tuple[str, str]] = []
    current_start = 1

    for line in lines:
        if line.startswith("+++ "):
            if in_file and current_records:
                hunks.append((current_start, current_records))
            candidate = line[4:]
            if candidate.startswith("b/"):
                candidate = candidate[2:]
            in_file = candidate == filepath or candidate.endswith("/" + filepath)
            if in_file:
                found = True
                hunks = []
                current_records = []
            in_hunk = False
            continue

        if line.startswith("--- "):
            continue

        if in_file:
            if line.startswith("@@"):
                if current_records:
                    hunks.append((current_start, current_records))
                    current_records = []
                current_start = _parse_hunk_new_start(line)
                in_hunk = True
                continue
            if not in_hunk:
                continue
            if line.startswith("+") and not line.startswith("+++"):
                current_records.append(("+", line[1:]))
            elif line.startswith(" "):
                current_records.append((" ", line[1:]))
            elif line.startswith("-"):
                pass
            elif line.startswith("\\"):
                pass
            else:
                in_file = False
                in_hunk = False

    if in_file and current_records:
        hunks.append((current_start, current_records))

    if not found:
        return None
    return hunks


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


def _build_snippet(
    line_records: list[tuple[str, str]],
    error_lineno: int | None,
    line_offset: int = 0,
) -> str | None:
    """Build a short code snippet around the error line for display in feedback.

    error_lineno is 1-based within line_records.  line_offset is added to
    displayed line numbers so they reflect new-file positions.
    Returns None if error_lineno is None or out of range.
    """
    if error_lineno is None or not (1 <= error_lineno <= len(line_records)):
        return None

    start = max(0, error_lineno - 1 - _SNIPPET_BEFORE)
    end = min(len(line_records), error_lineno + _SNIPPET_AFTER)

    snippet_lines = []
    for i in range(start, end):
        lineno = i + 1
        display_lineno = lineno + line_offset
        prefix, text = line_records[i]
        marker = "-->" if lineno == error_lineno else "   "
        snippet_lines.append(f"    {marker} {display_lineno:4d} {prefix} {text}")

    return "\n".join(snippet_lines)


def _parse_fragment(
    line_records: list[tuple[str, str]],
    filepath: str,
    line_offset: int = 0,
) -> dict[str, Any] | None:
    """Parse a reconstructed diff fragment using the multi-stage strategy.

    line_offset is added to reported line numbers so they refer to new-file
    positions rather than hunk-local positions.

    Returns an error dict {file, line, offset, message, snippet} if a real
    syntax error is found, or None if the fragment parses successfully at
    some stage.
    """
    content = _records_to_content(line_records)

    def _make_error(lineno: int | None, offset: int | None, msg: str) -> dict[str, Any]:
        display_lineno = lineno + line_offset if lineno is not None else None
        return {
            "file": filepath,
            "line": display_lineno,
            "offset": offset,
            "message": msg,
            "snippet": _build_snippet(line_records, lineno, line_offset),
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


# ── Verifier class ─────────────────────────────────────────────────────


class ASTCheckVerifier(BaseVerifier):
    """Static verifier: checks Python syntax of all changed files in the patch.

    Uses full-file parsing when SWE-bench metadata (repo, base_commit) is
    available; falls back to hunk-based parsing otherwise.
    """

    execution_mode: ClassVar[Literal["static", "dynamic"]] = "static"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("timeout", 120.0)
        super().__init__(**kwargs)

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

        instance_data = ctx.metadata.get("instance_data", {})
        repo = instance_data.get("repo")
        base_commit = instance_data.get("base_commit")

        if repo and base_commit:
            result = await _clone_and_parse(
                repo=repo,
                base_commit=base_commit,
                patch_diff=ctx.patch_diff,
                python_files=python_files,
                verifier_name=self.name,
                pass_threshold=self.pass_threshold,
            )
            if result is not None:
                return result
            logger.info("ast_check: full-file parsing failed, falling back to hunk parsing")

        return self._verify_from_hunks(ctx, python_files)

    def _verify_from_hunks(
        self, ctx: PatchContext, python_files: list[str],
    ) -> VerifierResult:
        """Fallback: parse each diff hunk in isolation."""
        errors: list[dict[str, Any]] = []
        parsed = 0
        errored = 0
        skipped = 0

        for filepath in python_files:
            hunks = _extract_new_content(ctx.patch_diff, filepath)

            if hunks is None:
                skipped += 1
                continue

            file_has_error = False
            for new_start, hunk_records in hunks:
                content = _records_to_content(hunk_records)
                if not content.strip():
                    continue

                error = _parse_fragment(
                    hunk_records, filepath, line_offset=new_start - 1,
                )
                if error is not None:
                    errors.append(error)
                    file_has_error = True

            if file_has_error:
                errored += 1
            else:
                parsed += 1

        checkable = parsed + errored
        score = parsed / checkable if checkable > 0 else 1.0

        return VerifierResult(
            name=self.name,
            status=VerifierStatus.OK,
            score=score,
            pass_threshold=self.pass_threshold,
            details={
                "errors": errors,
                "method": "hunk",
                "files_total": len(python_files),
                "files_checkable": checkable,
                "files_parsed": parsed,
                "files_skipped": skipped,
                "files_errored": errored,
            },
        )
