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

from swebench.harness.utils import load_swebench_dataset

from evals.swe_bench.prompt import create_prompt_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build SWE-bench prompted dataset (one-time prep step)"
    )
    parser.add_argument("--dataset", type=str, default="SWE-bench/SWE-bench_Lite",
                        help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split")
    parser.add_argument("--output", type=str, required=True,
                        help="Local path to write the prompted dataset JSONL")
    parser.add_argument("--prompt-style", type=str, default="style-3",
                        help="SWE-bench prompt style (default: style-3)")
    parser.add_argument("--file-source", type=str, default="oracle",
                        help="Source file selection: oracle, bm25, none "
                             "(default: oracle)")
    parser.add_argument("--instance-limit", type=int, default=0,
                        help="Max instances to process (0 = no limit)")
    parser.add_argument("--s3-output", type=str, default=None,
                        help="S3 URI to upload the prompted dataset "
                             "(e.g. s3://swe-bench/prompts/style-3-oracle.jsonl)")
    args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset} split={args.split}")
    dataset = load_swebench_dataset(args.dataset, split=args.split)
    logger.info(f"Loaded {len(dataset)} instances")

    if args.instance_limit > 0:
        dataset = dataset[:args.instance_limit]
        logger.info(f"Limited to {len(dataset)} instances")

    # Build prompted dataset
    output_path = create_prompt_dataset(
        instances=dataset,
        output_path=args.output,
        prompt_style=args.prompt_style,
        file_source=args.file_source,
    )

    logger.info(f"Prompted dataset written to {output_path}")

    # Upload to S3 if requested
    if args.s3_output:
        from evals.swe_bench.s3_storage import upload_file
        upload_file(output_path, args.s3_output)
        logger.info(f"Uploaded to {args.s3_output}")


if __name__ == "__main__":
    main()
