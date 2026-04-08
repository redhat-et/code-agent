"""
GSM8K reward function for OpenRLHF GRPO training.

Extracts the final numerical answer from model responses and compares
against the ground truth label. GSM8K answers are always integers
following "#### " in the label.

This function is passed to OpenRLHF via --remote_rm_url.
"""

import re
import torch


def _extract_gsm8k_answer(text: str) -> str | None:
    """
    Extract the final numerical answer from a model response.

    Tries multiple patterns in order of specificity:
    1. \\boxed{...}  -- standard math reasoning format
    2. #### <number> -- GSM8K native format
    3. Last standalone number in the response
    """
    # Pattern 1: \boxed{...}
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    # Pattern 2: #### <number>
    hash_match = re.findall(r"####\s*(-?[\d,]+\.?\d*)", text)
    if hash_match:
        return hash_match[-1].replace(",", "").strip()

    # Pattern 3: last number in text (fallback)
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None


def _normalize_number(s: str) -> float | None:
    """Normalize a number string to a float for comparison."""
    if s is None:
        return None
    try:
        return float(s.replace(",", "").strip())
    except ValueError:
        return None


def reward_func(queries, prompts, labels, **kwargs):
    """
    Compute rewards for GSM8K math problems.

    Args:
        queries: List[str] - Full text (prompt + response)
        prompts: List[str] - Original prompts only
        labels:  List[str] - Ground truth answers (e.g. "42" or "#### 42")

    Returns:
        dict with rewards, scores, and extra_logs
    """
    rewards = []

    for query, prompt, label in zip(queries, prompts, labels):
        # Isolate the response (strip the prompt prefix)
        if isinstance(prompt, str) and query.startswith(prompt):
            response = query[len(prompt):]
        else:
            response = query

        # Extract predicted answer from model response
        pred = _extract_gsm8k_answer(response)
        pred_num = _normalize_number(pred)

        # Extract ground truth answer from label
        # GSM8K labels may contain "#### <number>" or just the number
        gold = _extract_gsm8k_answer(label) if label else None
        gold_num = _normalize_number(gold) if gold else _normalize_number(label)

        # Binary reward: 1.0 if correct, 0.0 otherwise
        if pred_num is not None and gold_num is not None:
            correct = abs(pred_num - gold_num) < 1e-6
        else:
            correct = False

        rewards.append(1.0 if correct else 0.0)

    rewards_tensor = torch.tensor(rewards, dtype=torch.float)

    return {
        "rewards": rewards_tensor,
        "scores": rewards_tensor,
        "extra_logs": {
            "gsm8k_accuracy": rewards_tensor.mean().item(),
        },
    }
