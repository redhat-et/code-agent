"""Phase 1 Ray worker: generate patches via vLLM.

Each worker receives a sub-list of SWE-bench instances with pre-built
prompts (from swebench's prompt pipeline), calls the vLLM endpoint
to generate patches, and returns structured results.

Workers do NOT build prompts -- they receive them pre-built. Prompt
construction (which requires git cloning) is done separately via
build_prompt_dataset.py / job-build-prompts.yaml.
"""

from __future__ import annotations

import logging

import ray

from evals.common.inference_worker import InferenceWorker
from .prompt import extract_diff_from_response

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class PatchWorker(InferenceWorker):
    """Generates patches for SWE-bench instances via vLLM.

    Inherits from InferenceWorker and customizes for SWE-bench:
    - Handles SWE-bench's prompt format (first line = system message)
    - Extracts patches using swebench's extract_diff
    - Returns predictions with SWE-bench schema (model_patch)

    Args:
        vllm_urls: List of vLLM OpenAI-compatible base URLs.
            Requests are round-robined across them.
        model_name: Model name as registered in vLLM.
        max_tokens: Maximum tokens for patch generation.
        temperature: Sampling temperature.
    """

    def _generate(self, prompt: str) -> str:
        """Override to handle SWE-bench's prompt format.

        SWE-bench convention: first line is the system message,
        rest is the user message.

        Args:
            prompt: Pre-built prompt text from swebench's pipeline.

        Returns:
            Raw response from the model.
        """
        client = self._get_client()

        # Split prompt into system and user messages
        lines = prompt.split("\n", 1)
        if len(lines) == 2:
            system_msg = lines[0]
            user_msg = lines[1]
        else:
            system_msg = None
            user_msg = prompt

        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": user_msg})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return response.choices[0].message.content or ""

    def generate_patches(
        self,
        instances: list[dict],
        prompts: dict[str, str],
    ) -> list[dict]:
        """Generate patches for a batch of SWE-bench instances.

        Uses parent's generate_batch with SWE-bench-specific configuration.

        Args:
            instances: List of SWE-bench dataset instances.
            prompts: Map of instance_id -> pre-built prompt text.

        Returns:
            List of dicts with keys: instance_id, model_patch,
            full_output, model_name_or_path, error.
        """
        results = self.generate_batch(
            instances=instances,
            prompts=prompts,
            extract_fn=extract_diff_from_response,
            instance_id_key="instance_id",
        )

        # Rename 'prediction' to 'model_patch' for SWE-bench schema
        for result in results:
            result["model_patch"] = result.pop("prediction")

        return results
