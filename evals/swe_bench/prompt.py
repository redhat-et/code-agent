"""SWE-bench prompt construction.

Wraps swebench's own prompt pipeline to build prompts identical to the
ones used in the SWE-bench paper. This ensures our evaluation results
are directly comparable to published benchmarks.

The standard flow is:
  1. create_prompt_dataset() -- clones repos, reads source files, builds
     prompts using swebench's style-3 format. Run once as a prep step.
  2. Workers receive pre-built prompts and just call vLLM.

For diff extraction from model responses, we use swebench's extract_diff
which handles <patch>, <diff>, and code fence formats.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from swebench.inference.make_datasets.utils import extract_diff

logger = logging.getLogger(__name__)


def create_prompt_dataset(
    instances: list[dict],
    output_path: str | Path,
    prompt_style: str = "style-3",
    file_source: str = "oracle",
    max_context_len: int | None = None,
    tokenizer_name: str | None = None,
) -> Path:
    """Build a prompted dataset using swebench's pipeline.

    Clones repos, reads source files, and constructs prompts identical
    to the SWE-bench paper. Results are written to a JSONL file where
    each line has an instance_id and text_inputs field.

    This is a heavyweight operation (clones repos from GitHub) and
    should be run once as a prep step, not per-worker.

    Args:
        instances: SWE-bench dataset instances (list of dicts).
        output_path: Path to write the prompted dataset JSONL.
        prompt_style: Prompt style key (default "style-3", the standard).
        file_source: How to select source files for the prompt:
            "oracle" -- files changed in the gold patch (standard).
            "bm25" -- BM25-retrieved files.
            "none" -- no source files (issue-only prompt).
        max_context_len: Optional max token length for the prompt.
        tokenizer_name: Tokenizer name for token counting (required
            if max_context_len is set).

    Returns:
        Path to the output JSONL file.
    """
    from swebench.inference.make_datasets.create_instance import add_text_inputs

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # add_text_inputs expects a dict of {instance_id: instance}
    instances_dict = {inst["instance_id"]: dict(inst) for inst in instances}

    logger.info(
        f"Building prompted dataset: {len(instances_dict)} instances, "
        f"style={prompt_style}, file_source={file_source}"
    )

    add_text_inputs(
        instances_dict,
        retrieval_file=None,
        k=None,
        prompt_style=prompt_style,
        file_source=file_source,
        max_context_len=max_context_len,
        tokenizer_name=tokenizer_name,
        verbose=True,
        progress_file=str(output_path),
    )

    logger.info(f"Wrote prompted dataset to {output_path}")
    return output_path


def load_prompt_dataset(path: str | Path) -> dict[str, str]:
    """Load a prompted dataset into a map of instance_id -> prompt text.

    Args:
        path: Path to the JSONL file produced by create_prompt_dataset.

    Returns:
        Dict mapping instance_id to the prompt text (text_inputs field).
    """
    prompts = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            instance_id = entry.get("instance_id")
            text_inputs = entry.get("text_inputs", "")
            if instance_id and text_inputs:
                prompts[instance_id] = text_inputs
    return prompts


def extract_diff_from_response(response: str) -> str:
    """Extract a unified diff from a model response.

    Delegates to swebench's extract_diff which handles:
      - <patch>...</patch> XML tags
      - <diff>...</diff> XML tags
      - ```diff ... ``` markdown code fences
      - Raw diff content

    Args:
        response: Raw text from the LLM.

    Returns:
        The extracted diff text, or the full response if no diff
        markers are found.
    """
    extracted = extract_diff(response)
    if extracted:
        return extracted
    return response.strip()
