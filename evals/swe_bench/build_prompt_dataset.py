"""Build the SWE-bench prompted dataset.

One-time prep step that clones repos, reads source files, and constructs
prompts using swebench's own pipeline. The output is a JSONL file where
each line has instance_id and text_inputs fields. This file is uploaded
to S3/MinIO and reused by all future Phase 1 inference runs.

This script does NOT require Ray, vLLM, or any cluster infrastructure.
It can be run locally or as a standalone K8s Job.

Usage:
    # Run locally
    python -m evals.swe_bench.build_prompt_dataset \
        --dataset SWE-bench/SWE-bench_Lite \
        --output /tmp/prompted_dataset.jsonl \
        --s3-output s3://swe-bench/prompts/style-3-oracle.jsonl

    # Or just build locally without S3
    python -m evals.swe_bench.build_prompt_dataset \
        --dataset SWE-bench/SWE-bench_Lite \
        --output /tmp/prompted_dataset.jsonl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from swebench.harness.utils import load_swebench_dataset

from evals.common.prompt_builder import build_prompted_dataset_main
from evals.swe_bench.prompt import create_prompt_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_dataset(name: str, split: str) -> list[dict]:
    """Load SWE-bench dataset."""
    return load_swebench_dataset(name, split=split)


def build_prompts(instances: list[dict], output_path: Path, **kwargs) -> Path:
    """Build SWE-bench prompts."""
    return create_prompt_dataset(
        instances=instances,
        output_path=output_path,
        prompt_style=kwargs.get("prompt_style", "style-3"),
        file_source=kwargs.get("file_source", "oracle"),
    )


def add_swe_bench_args(parser: argparse.ArgumentParser) -> None:
    """Add SWE-bench-specific arguments."""
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="style-3",
        help="SWE-bench prompt style (default: style-3)",
    )
    parser.add_argument(
        "--file-source",
        type=str,
        default="oracle",
        help="Source file selection: oracle, bm25, none (default: oracle)",
    )


def main():
    build_prompted_dataset_main(
        dataset_loader=load_dataset,
        prompt_builder=build_prompts,
        default_dataset="SWE-bench/SWE-bench_Lite",
        description="Build SWE-bench prompted dataset (one-time prep step)",
        extra_args_fn=add_swe_bench_args,
    )


if __name__ == "__main__":
    main()
