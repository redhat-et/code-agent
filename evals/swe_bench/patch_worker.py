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

from .prompt import extract_diff_from_response

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class PatchWorker:
    """Generates patches for SWE-bench instances via vLLM.

    Each worker processes its assigned instances sequentially.
    Multiple workers run in parallel across the Ray cluster,
    each pointing at the same vLLM endpoint (or different
    endpoints when scaling).

    Args:
        vllm_urls: List of vLLM OpenAI-compatible base URLs.
            Requests are round-robined across them.
        model_name: Model name as registered in vLLM.
        max_tokens: Maximum tokens for patch generation.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        vllm_urls: list[str],
        model_name: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        import openai

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.clients = [
            openai.OpenAI(base_url=url, api_key="not-needed")
            for url in vllm_urls
        ]
        self._call_count = 0

    def _get_client(self):
        """Round-robin across vLLM clients."""
        client = self.clients[self._call_count % len(self.clients)]
        self._call_count += 1
        return client

    def _generate_patch(self, instance_id: str, prompt: str) -> tuple[str, str]:
        """Call vLLM to generate a patch for one instance.

        Args:
            instance_id: SWE-bench instance ID (for logging).
            prompt: Pre-built prompt text from swebench's pipeline.

        Returns:
            Tuple of (extracted_patch, raw_response).
        """
        client = self._get_client()

        # SWE-bench convention: first line is the system message,
        # rest is the user message.
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

        raw = response.choices[0].message.content or ""
        patch = extract_diff_from_response(raw)
        return patch, raw

    def generate_patches(
        self,
        instances: list[dict],
        prompts: dict[str, str],
    ) -> list[dict]:
        """Generate patches for a batch of instances.

        Args:
            instances: List of SWE-bench dataset instances.
            prompts: Map of instance_id -> pre-built prompt text.

        Returns:
            List of dicts with keys: instance_id, model_patch,
            full_output, model_name_or_path, error.
        """
        results = []

        for instance in instances:
            instance_id = instance["instance_id"]
            prompt = prompts.get(instance_id)

            if not prompt:
                logger.error(f"No prompt found for {instance_id}")
                results.append({
                    "instance_id": instance_id,
                    "model_patch": "",
                    "full_output": "",
                    "model_name_or_path": self.model_name,
                    "error": "No prompt found for instance",
                })
                continue

            logger.info(f"Generating patch for {instance_id}")

            try:
                patch, raw = self._generate_patch(instance_id, prompt)

                results.append({
                    "instance_id": instance_id,
                    "model_patch": patch,
                    "full_output": raw,
                    "model_name_or_path": self.model_name,
                    "error": None,
                })

            except Exception as e:
                logger.error(f"Error generating patch for {instance_id}: {e}")
                results.append({
                    "instance_id": instance_id,
                    "model_patch": "",
                    "full_output": "",
                    "model_name_or_path": self.model_name,
                    "error": str(e),
                })

        return results
