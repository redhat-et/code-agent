"""Common scaffolding for building prompted datasets.

Provides reusable main() function that handles:
- Dataset loading
- Instance limiting
- Prompt building (benchmark-specific)
- S3 upload

Each benchmark provides callbacks for dataset-specific logic.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable

from evals.common.s3_storage import upload_file

logger = logging.getLogger(__name__)


def build_prompted_dataset_main(
    dataset_loader: Callable[[str, str], list[dict]],
    prompt_builder: Callable[[list[dict], Path, dict[str, Any]], Path],
    default_dataset: str,
    description: str,
    extra_args_fn: Callable[[argparse.ArgumentParser], None] | None = None,
) -> None:
    """Generic main() for building prompted datasets.

    Args:
        dataset_loader: Function(dataset_name, split) -> list[dict]
            Loads the dataset and returns instances as list of dicts.
        prompt_builder: Function(instances, output_path, kwargs) -> Path
            Builds prompts and writes to output_path. Returns path.
            kwargs contains all extra args added by extra_args_fn.
        default_dataset: Default dataset name for --dataset arg.
        description: Description for argparse.
        extra_args_fn: Optional function to add benchmark-specific args.
            Called with parser as argument.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        help="Dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Local path to write the prompted dataset JSONL",
    )
    parser.add_argument(
        "--instance-limit",
        type=int,
        default=0,
        help="Max instances to process (0 = no limit)",
    )
    parser.add_argument(
        "--s3-output",
        type=str,
        default=None,
        help="S3 URI to upload the prompted dataset",
    )

    # Add benchmark-specific args
    if extra_args_fn:
        extra_args_fn(parser)

    args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset} split={args.split}")
    instances = dataset_loader(args.dataset, args.split)
    logger.info(f"Loaded {len(instances)} instances")

    # Apply instance limit
    if args.instance_limit > 0:
        instances = instances[:args.instance_limit]
        logger.info(f"Limited to {len(instances)} instances")

    # Build prompted dataset (pass all args except the standard ones)
    standard_args = {"dataset", "split", "output", "instance_limit", "s3_output"}
    extra_kwargs = {
        k: v for k, v in vars(args).items() if k not in standard_args
    }

    output_path = prompt_builder(
        instances=instances,
        output_path=Path(args.output),
        **extra_kwargs,
    )

    logger.info(f"Prompted dataset written to {output_path}")

    # Upload to S3 if requested
    if args.s3_output:
        upload_file(output_path, args.s3_output)
        logger.info(f"Uploaded to {args.s3_output}")
