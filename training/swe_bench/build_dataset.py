"""Build a training dataset for SWE-bench GRPO from HuggingFace.

Converts SWE-bench Verified instances into the JSONL format that
OpenRLHF expects::

    {"input": "<problem_statement>", "label": "<instance_metadata_json>"}

``input``  becomes ``states["observation"]`` in AgentInstanceBase.reset().
``label``  is forwarded through all step() calls and used by reset()
           to create the correct K8s Pod.

Usage:
    python build_dataset.py --output swe_bench_train.jsonl
    python build_dataset.py --output swe_bench_test.jsonl --instance-limit 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"

LABEL_FIELDS = [
    "instance_id",
    "repo",
    "base_commit",
    "version",
    "problem_statement",
]


def build(output: Path, instance_limit: int = 0) -> None:
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    if instance_limit > 0:
        ds = ds.select(range(min(instance_limit, len(ds))))

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for row in ds:
            label = {k: row[k] for k in LABEL_FIELDS if k in row}
            entry = {
                "input": row["problem_statement"],
                "label": json.dumps(label),
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(ds)} instances to {output}")


def main():
    parser = argparse.ArgumentParser(description="Build SWE-bench training dataset")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output JSONL path"
    )
    parser.add_argument(
        "--instance-limit",
        type=int,
        default=0,
        help="Limit number of instances (0 = all)",
    )
    args = parser.parse_args()
    build(args.output, args.instance_limit)


if __name__ == "__main__":
    main()
