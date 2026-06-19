"""Convert a fine-tuned Qwen3.5 checkpoint for vLLM serving.

vLLM loads Qwen3.5 via Qwen3_5ForConditionalGeneration, which expects
weight keys prefixed with ``language_model.``.  DeepSpeed/OpenRLHF save
text-only weights without that prefix (``model.layers.*``).

This script renames the keys so vLLM can load the checkpoint, and
patches config.json to declare the correct architecture.

Usage:
    python convert_checkpoint_for_vllm.py \\
        --input  /tmp/checkpoints/step-100 \\
        --output /tmp/checkpoints/step-100-vllm
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def convert(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy non-weight files (tokenizer, config, etc.)
    for f in input_dir.iterdir():
        if f.suffix in (".json", ".txt", ".model", ".tiktoken") or f.name == "tokenizer.model":
            shutil.copy2(f, output_dir / f.name)

    # Rename weight keys
    weight_files = sorted(input_dir.glob("*.bin")) + sorted(input_dir.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {input_dir}")

    is_safetensors = weight_files[0].suffix == ".safetensors"

    for wf in weight_files:
        if is_safetensors:
            from safetensors.torch import load_file, save_file
            state_dict = load_file(wf)
            renamed = {}
            for k, v in state_dict.items():
                new_key = k if k.startswith("language_model.") else f"language_model.{k}"
                renamed[new_key] = v
            save_file(renamed, output_dir / wf.name)
        else:
            state_dict = torch.load(wf, map_location="cpu", weights_only=True)
            renamed = {}
            for k, v in state_dict.items():
                new_key = k if k.startswith("language_model.") else f"language_model.{k}"
                renamed[new_key] = v
            torch.save(renamed, output_dir / wf.name)

        print(f"Converted {wf.name}: {len(state_dict)} keys")

    # Patch config.json architecture
    config_path = output_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        archs = config.get("architectures", [])
        patched = [
            a.replace("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration")
            for a in archs
        ]
        if patched != archs:
            config["architectures"] = patched
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            print(f"Patched architectures: {archs} -> {patched}")

    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert checkpoint for vLLM Qwen3.5 serving")
    parser.add_argument("--input", type=Path, required=True, help="Input checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for vLLM-compatible checkpoint")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
