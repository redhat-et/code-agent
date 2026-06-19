"""Build a training dataset for SWE-bench GRPO from HuggingFace.

Converts SWE-bench Verified instances into the JSONL format that
OpenRLHF expects::

    {"input": "<chat_template_applied_prompt>", "label": "<instance_metadata_json>"}

The chat template is applied here with ``enable_thinking=False`` so that
OpenRLHF does not need to apply it again (do NOT pass
``--data.apply_chat_template`` to the training script).

Usage:
    python build_dataset.py --output swe_bench_train.jsonl
    python build_dataset.py --output swe_bench_test.jsonl --instance-limit 16
    python build_dataset.py --output swe_bench_train.jsonl --model Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"


def build(output: Path, instance_limit: int = 0, model_name: str = DEFAULT_MODEL, enable_thinking: bool = False) -> None:
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    if instance_limit > 0:
        ds = ds.select(range(min(instance_limit, len(ds))))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for row in ds:
            # Format as a chat message for the model
            messages = [{"role": "user", "content": row["problem_statement"]}]

            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )

            # Include the full instance dict — make_test_spec() needs
            # test_patch, FAIL_TO_PASS, PASS_TO_PASS, etc.
            label = dict(row)
            entry = {
                "input": formatted,
                "label": json.dumps(label),
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(ds)} instances to {output} (model: {model_name}, thinking: {'enabled' if enable_thinking else 'disabled'})")


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
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model for chat template (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable model thinking/reasoning mode (default: disabled)",
    )
    args = parser.parse_args()
    build(args.output, args.instance_limit, args.model, args.enable_thinking)


if __name__ == "__main__":
    main()
