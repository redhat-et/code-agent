"""Generic Ray worker for LLM inference via vLLM.

Provides a reusable base class for generating predictions from LLM prompts
across different benchmarks. Subclasses customize prompt formatting and
response extraction.
"""

from __future__ import annotations

import logging
from typing import Callable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class InferenceWorker:
    """Generates predictions via vLLM for arbitrary evaluation tasks.

    Each worker processes its assigned instances sequentially.
    Multiple workers run in parallel across the Ray cluster,
    each pointing at the same vLLM endpoint (or different
    endpoints when scaling).

    Args:
        vllm_urls: List of vLLM OpenAI-compatible base URLs.
            Requests are round-robined across them.
        model_name: Model name as registered in vLLM.
        max_tokens: Maximum tokens for generation.
        temperature: Sampling temperature.
        system_message: Optional system message to prepend to prompts.
    """

    def __init__(
        self,
        vllm_urls: list[str],
        model_name: str,
        api_key: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        system_message: str | None = None,
        timeout: float = 600.0,
    ):
        import openai

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_message = system_message
        self.timeout = timeout

        if not vllm_urls:
            raise ValueError("vllm_urls must contain at least one endpoint")

        self.clients = [
            openai.OpenAI(base_url=url, api_key=api_key, timeout=timeout)
            for url in vllm_urls
        ]
        self._call_count = 0

    def _get_client(self):
        """Round-robin across vLLM clients."""
        client = self.clients[self._call_count % len(self.clients)]
        self._call_count += 1
        return client

    def _generate(self, prompt: str) -> str:
        """Call vLLM to generate a response for one prompt.

        Args:
            prompt: The prompt text.

        Returns:
            Raw response from the model.
        """
        client = self._get_client()

        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return response.choices[0].message.content or ""

    def generate_batch(
        self,
        instances: list[dict],
        prompts: dict[str, str] | None = None,
        extract_fn: Callable[[str], str] | None = None,
        instance_id_key: str = "instance_id",
        prompt_key: str = "prompt"
    ) -> list[dict]:
        """Generate predictions for a batch of instances.

        Args:
            instances: List of dataset instances.
            prompts: Map of instance_id -> prompt text.
            extract_fn: Optional function to extract the prediction from raw response.
                       If None, uses the raw response as-is.
            instance_id_key: Key to use for instance ID in the instance dict.

        Returns:
            List of dicts with keys: instance_id, prediction, full_output,
            model_name_or_path, error.
        """
        results = []

        for instance in instances:
            instance_id = instance[instance_id_key]

            if prompts is None:
                prompt = instance.get(prompt_key)
            else:
                prompt = prompts.get(instance_id)

            if prompt is None:
                logger.error(f"No prompt found for {instance_id}")
                results.append({
                    "instance_id": instance_id,
                    "prediction": "",
                    "full_output": "",
                    "model_name_or_path": self.model_name,
                    "error": "No prompt found for instance",
                })
                continue

            try:
                logger.info(f"Generating prediction for {instance_id}")
                raw_response = self._generate(prompt)
                logger.debug(f"{raw_response}")

                if extract_fn:
                    prediction = extract_fn(raw_response)
                else:
                    prediction = raw_response

                results.append({
                    "instance_id": instance_id,
                    "prediction": prediction,
                    "full_output": raw_response,
                    "model_name_or_path": self.model_name,
                    "error": None,
                })

            except Exception as e:
                logger.error(f"Error generating prediction for {instance_id}: {e}")
                results.append({
                    "instance_id": instance_id,
                    "prediction": "",
                    "full_output": "",
                    "model_name_or_path": self.model_name,
                    "error": str(e),
                })

        return results
