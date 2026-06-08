"""Generic multi-turn worker base class.

Provides the benchmark-agnostic pieces of a Ray-based multi-turn evaluation worker:
- vLLM client management with round-robin across endpoints
- Naive (inline vLLM) generation
- Static and dynamic verifier execution (inline async vs thread pool)
- Batch evaluation entry point

Subclasses provide the benchmark-specific parts:
- _generate_turn(messages, instance, run_id) -> str
- _evaluate_instance(instance, prompts, run_id) -> dict
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from evals.common.score_aggregator import MeanAggregator, ScoreAggregator
from evals.common.verifier_set import VerifierSet
from verifiers.base import VerifierResult, VerifierStatus

logger = logging.getLogger(__name__)


class MultiTurnWorkerBase:
    """Base class for multi-turn evaluation Ray workers.

    Args:
        vllm_urls: vLLM OpenAI-compatible base URLs (round-robined).
        model_name: Model name as registered in vLLM.
        max_tokens: Max tokens per vLLM call.
        temperature: Sampling temperature.
        verifier_set: Full verifier set; intermediate/final subsets are derived from flags.
        aggregator: Aggregates per-verifier scores into a scalar.
        max_turns: Hard cap on generation attempts per instance.
        max_concurrent_jobs: Max concurrent threads for dynamic verifiers.
    """

    def __init__(
        self,
        vllm_urls: list[str],
        model_name: str,
        max_tokens: int = 16000,
        temperature: float = 0.15,
        verifier_set: VerifierSet | None = None,
        aggregator: ScoreAggregator | None = None,
        max_turns: int = 1,
        max_concurrent_jobs: int = 4,
        disable_thinking: bool = False,
    ):
        import openai

        if not vllm_urls:
            raise ValueError("vllm_urls must contain at least one endpoint")

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_turns = max_turns
        self.max_concurrent_jobs = max_concurrent_jobs
        self.disable_thinking = disable_thinking

        self.vllm_urls = vllm_urls
        self.clients = [
            openai.OpenAI(base_url=url, api_key="not-needed", timeout=600.0)
            for url in vllm_urls
        ]
        self._call_count = 0

        self.verifier_set = verifier_set or VerifierSet()
        self.intermediate_set = self.verifier_set.intermediate_subset()
        self.final_set = self.verifier_set.final_subset()
        self.aggregator = aggregator or MeanAggregator()

    # ── vLLM (naive generation) ─────────────────────────────────────────────

    def _next_index(self) -> int:
        idx = self._call_count % len(self.clients)
        self._call_count += 1
        return idx

    def _get_client(self):
        return self.clients[self._next_index()]

    def _get_vllm_url(self) -> str:
        return self.vllm_urls[self._next_index()]

    def _generate_naive(self, messages: list[dict]) -> str:
        """Generate a response via inline vLLM inference."""
        client = self._get_client()
        kwargs: dict = dict(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if self.disable_thinking:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
            }
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    # ── Verifier execution ──────────────────────────────────────────────────

    def _run_verifier_set_inline(self, vset: VerifierSet, ctx) -> list[VerifierResult]:
        """Run all verifiers in vset concurrently (async, in-process)."""
        async def _run_all():
            return await asyncio.gather(*(entry.verifier.safe_verify(ctx) for entry in vset.entries))
        return asyncio.run(_run_all())

    def _run_mixed_verifier_set(self, vset: VerifierSet, ctx) -> list[VerifierResult]:
        """Run static verifiers inline and dynamic verifiers in a thread pool."""
        static_entries = [e for e in vset.entries if e.verifier.execution_mode == "static"]
        dynamic_entries = [e for e in vset.entries if e.verifier.execution_mode == "dynamic"]

        results: dict[str, VerifierResult] = {}

        if static_entries:
            for r in self._run_verifier_set_inline(VerifierSet(static_entries), ctx):
                results[r.name] = r

        if dynamic_entries:
            with ThreadPoolExecutor(
                max_workers=min(self.max_concurrent_jobs, len(dynamic_entries))
            ) as pool:
                futures = {
                    pool.submit(asyncio.run, entry.verifier.safe_verify(ctx)): entry
                    for entry in dynamic_entries
                }
                for future in as_completed(futures):
                    entry = futures[future]
                    try:
                        r = future.result()
                    except Exception as e:
                        r = VerifierResult(
                            name=entry.verifier.name,
                            status=VerifierStatus.ERROR,
                            score=0.0,
                            pass_threshold=entry.verifier.pass_threshold,
                            details={"error": str(e)},
                        )
                    results[r.name] = r

        return [results[e.verifier.name] for e in vset.entries if e.verifier.name in results]

    def _run_verifier_set(self, vset: VerifierSet, ctx) -> list[VerifierResult]:
        """Run a verifier set against a pre-built context."""
        if not vset:
            return []
        if vset.is_dynamic:
            return self._run_mixed_verifier_set(vset, ctx)
        return self._run_verifier_set_inline(vset, ctx)

    # ── Batch entry point ───────────────────────────────────────────────────

    def evaluate_batch(
        self,
        instances: list[dict],
        prompts: dict[str, str],
        run_id: str,
    ) -> list[dict]:
        """Evaluate a batch of instances sequentially."""
        results: list[dict] = []
        for inst in instances:
            try:
                results.append(self._evaluate_instance(inst, prompts, run_id))
            except Exception as e:
                instance_id = inst.get("instance_id")
                logger.exception("Failed evaluating instance %s", instance_id)
                results.append({
                        "instance_id": instance_id,
                        "error": str(e),
                        "resolved": False,
                })
        return results
