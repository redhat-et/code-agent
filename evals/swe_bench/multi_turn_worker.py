"""
SWE-bench multi-turn Ray worker.

Subclasses MultiTurnWorkerBase with SWE-bench-specific generation and evaluation:
  - naive  — inline vLLM inference per turn
  - agent  — K8s agent Job per turn (run_eval=False; grading via SWEBenchUnitTestVerifier)
             Prior patches and feedback are injected into the problem statement
             for subsequent agent turns.

The base class (MultiTurnWorkerBase) handles vLLM client management, verifier
execution, and the batch entry point. This class wires them to SWE-bench.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import ray

from evals.common.multi_turn import MultiTurnResult, MultiTurnSession
from evals.common.multi_turn_worker import MultiTurnWorkerBase
from evals.common.score_aggregator import ScoreAggregator
from evals.common.verifier_set import VerifierSet
from evals.swe_bench.instance_runner import InstanceRunner
from evals.swe_bench.prompt import extract_diff_from_response
from verifiers.base import PatchContext, VerifierResult

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class SWEBenchMultiTurnWorker(MultiTurnWorkerBase):
    """SWE-bench multi-turn evaluation worker.

    Extends MultiTurnWorkerBase with:
    - SWE-bench verifier context construction (_make_context)
    - naive strategy: inline vLLM + patch extraction from diff
    - agent strategy: K8s Job per turn with prior context injection
    - SWE-bench result format (model_patch, eval_report, resolved)

    Agent-only args:
        agent_config_dict: Serialised AgentConfig dict.
        model_api_key: API key forwarded to the agent.
        subset: SWE-bench subset name forwarded to the agent.
        split: Dataset split forwarded to the agent.
        step_limit: Max agent steps per turn.
        cost_limit: Max cost per turn in dollars.
    """

    def __init__(
        self,
        vllm_urls: list[str],
        model_name: str,
        strategy: Literal["naive", "agent"] = "naive",
        max_tokens: int = 16000,
        temperature: float = 0.15,
        verifier_set: VerifierSet | None = None,
        aggregator: ScoreAggregator | None = None,
        max_turns: int = 1,
        k8s_namespace: str | None = None,
        timeout: int = 1800,
        service_account: str = "swe-bench-eval",
        max_concurrent_jobs: int = 4,
        swebench_namespace: str = "swebench",
        image_registry: str | None = None,
        # agent strategy only
        agent_config_dict: dict | None = None,
        model_api_key: str = "dummy",
        subset: str = "verified",
        split: str = "test",
        step_limit: int = 100,
        cost_limit: float = 3.0,
    ):
        from evals.swe_bench.agent_config import AgentConfig
        from evals.swe_bench.agent_worker import _detect_namespace, _init_k8s

        super().__init__(
            vllm_urls=vllm_urls,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            verifier_set=verifier_set,
            aggregator=aggregator,
            max_turns=max_turns,
            max_concurrent_jobs=max_concurrent_jobs,
        )

        self.strategy = strategy
        self.swebench_namespace = swebench_namespace
        self.image_registry = image_registry

        self.k8s_namespace = k8s_namespace or _detect_namespace()
        self.service_account = service_account
        self.batch_api, self.core_api = _init_k8s()

        self.runner = InstanceRunner(
            k8s_namespace=self.k8s_namespace,
            timeout=timeout,
            service_account=service_account,
        )

        self.agent_config = AgentConfig(**agent_config_dict) if agent_config_dict else None
        self.model_api_key = model_api_key
        self.subset = subset
        self.split = split
        self.step_limit = step_limit
        self.cost_limit = cost_limit

    # ── SWE-bench verifier context ──────────────────────────────────────────

    def _make_context(self, patch: str, instance_data: dict, run_id: str) -> PatchContext:
        return PatchContext(
            patch_diff=patch,
            changed_files=_extract_changed_files(patch),
            task_id=instance_data["instance_id"],
            metadata={
                "instance_data": instance_data,
                "run_id": run_id,
                "runner": self.runner,
                "model_name": self.model_name,
                "image_registry": self.image_registry,
                "swebench_namespace": self.swebench_namespace,
            },
        )

    # ── Generation dispatch ─────────────────────────────────────────────────

    def _generate_turn(
        self,
        messages: list[dict],
        instance: dict,
        run_id: str,
    ) -> str:
        if self.strategy == "naive":
            return self._generate_naive(messages)
        return self._generate_agent(messages, instance, run_id)

    def _generate_agent(
        self,
        messages: list[dict],
        instance: dict,
        run_id: str,
    ) -> str:
        """Generate a patch via a K8s agent Job (eval disabled; grading done separately).

        For turns after the first, prior patches and feedback are injected into
        the problem statement so the agent has full context of previous attempts.
        """
        from swebench.harness.constants import DOCKER_WORKDIR

        from evals.swe_bench.agent_worker import (
            _build_job_command,
            _build_job_manifest,
            _delete_job,
            _get_pod_logs,
            _wait_for_job,
            extract_prediction_from_logs,
            resolve_instance,
        )

        assert self.agent_config is not None, "agent_config_dict required for agent strategy"

        instance_id = instance["instance_id"]
        n_initial = 2 if messages and messages[0]["role"] == "system" else 1
        turn_idx = (len(messages) - n_initial) // 2

        modified_instance = dict(instance)
        if turn_idx > 0 and not self.agent_config.needs_swebench_dataset:
            prior_context = self._extract_prior_context(messages)
            if prior_context:
                modified_instance["problem_statement"] = (
                    instance.get("problem_statement", "")
                    + "\n\n---\nPrior attempts and feedback:\n\n"
                    + prior_context
                )

        job_name = None
        try:
            info = resolve_instance(
                modified_instance,
                self.image_registry or "",
                self.swebench_namespace,
            )

            template_vars = {
                "instance_id": instance_id,
                "model_name": self.model_name,
                "model_base_url": self._get_vllm_url(),
                "model_api_key": self.model_api_key,
                "workdir": DOCKER_WORKDIR,
                "problem_statement_file": "/tmp/problem_statement.txt",
                "subset": self.subset,
                "split": self.split,
                "step_limit": str(self.step_limit),
                "cost_limit": str(self.cost_limit),
            }

            command = _build_job_command(
                agent_config=self.agent_config,
                instance=modified_instance,
                template_vars=template_vars,
                eval_script="",
                run_eval=False,
            )
            job = _build_job_manifest(
                instance_id=instance_id,
                run_id=f"{run_id}-t{turn_idx}",
                image=info.image,
                command=command,
                namespace=self.k8s_namespace,
                agent_config=self.agent_config,
                service_account=self.service_account,
            )

            self.batch_api.create_namespaced_job(namespace=self.k8s_namespace, body=job)
            job_name = job.metadata.name
            logger.info(f"[{instance_id}] Turn {turn_idx}: created agent Job {job_name}")

            _, timed_out = _wait_for_job(self.batch_api, job_name, self.k8s_namespace)
            if timed_out:
                logger.warning(f"[{instance_id}] Turn {turn_idx}: agent Job timed out")

            logs = _get_pod_logs(self.core_api, job_name, self.k8s_namespace)
            prediction = extract_prediction_from_logs(
                logs=logs,
                instance_id=instance_id,
                model_name=self.model_name,
                patch_extraction=self.agent_config.patch_extraction,
            )
            return prediction.get("model_patch", "")

        except Exception as e:
            logger.exception(f"[{instance_id}] Turn {turn_idx}: agent Job failed")
            raise RuntimeError(f"Agent turn {turn_idx} failed for {instance_id}") from e
        finally:
            if job_name:
                _delete_job(self.batch_api, job_name, self.k8s_namespace)

    def _extract_prior_context(self, messages: list[dict]) -> str:
        """Format prior assistant+user message pairs as plain text for context injection."""
        start = 2 if messages and messages[0]["role"] == "system" else 1
        parts = []
        attempt = 1
        i = start
        while i < len(messages):
            if messages[i]["role"] == "assistant":
                parts.append(f"### Attempt {attempt}\n{messages[i]['content']}")
                i += 1
                if i < len(messages) and messages[i]["role"] == "user":
                    parts.append(f"### Feedback on Attempt {attempt}\n{messages[i]['content']}")
                    i += 1
                attempt += 1
            else:
                i += 1
        return "\n\n".join(parts)

    # ── Per-instance loop ───────────────────────────────────────────────────

    def _evaluate_instance(
        self,
        instance: dict,
        prompts: dict[str, str],
        run_id: str,
    ) -> dict[str, Any]:
        instance_id = instance["instance_id"]

        if self.strategy == "naive":
            prompt = prompts.get(instance_id, "")
            if not prompt:
                logger.error(f"No prompt found for {instance_id}")
                return _error_result(instance_id, self.model_name, "No prompt found for instance")
            lines = prompt.split("\n", 1)
            if len(lines) == 2:
                initial_messages = [
                    {"role": "system", "content": lines[0]},
                    {"role": "user", "content": lines[1]},
                ]
            else:
                initial_messages = [{"role": "user", "content": prompt}]
            extract_fn = extract_diff_from_response
        else:
            initial_messages = [
                {"role": "user", "content": instance.get("problem_statement", "")},
            ]
            extract_fn = lambda x: x  # agent jobs return the patch directly

        logger.info(
            f"[{instance_id}] Starting multi-turn evaluation "
            f"(strategy={self.strategy}, max_turns={self.max_turns})"
        )

        session = MultiTurnSession(
            generate_fn=lambda messages: self._generate_turn(messages, instance, run_id),
            extract_fn=extract_fn,
            run_intermediate_fn=lambda patch: self._run_verifier_set(
                self.intermediate_set, self._make_context(patch, instance, run_id)
            ),
            run_final_fn=lambda patch: self._run_verifier_set(
                self.final_set, self._make_context(patch, instance, run_id)
            ),
            aggregator=self.aggregator,
            intermediate_verifier_set=self.intermediate_set,
            max_turns=self.max_turns,
        )

        try:
            result: MultiTurnResult = session.run(initial_messages)
        except Exception as e:
            logger.error(f"[{instance_id}] Multi-turn session failed: {e}")
            return _error_result(instance_id, self.model_name, str(e))

        logger.info(
            f"[{instance_id}] Completed: {result.num_turns} turn(s), "
            f"early_exit={result.stopped_early}, "
            f"final_score={result.final_aggregate_score:.3f}"
        )

        swe_test_result = next(
            (r for r in result.final_verifier_results if r.name == "swe_test"), None
        )
        eval_report = swe_test_result.details if swe_test_result else {}

        return {
            "instance_id": instance_id,
            "model_patch": result.final_output,
            "model_name_or_path": self.model_name,
            "error": None,
            "eval_report": eval_report,
            "resolved": eval_report.get("resolved", False),
            "multi_turn": {
                "num_turns": result.num_turns,
                "stopped_early": result.stopped_early,
                "final_aggregate_score": result.final_aggregate_score,
                "turns": [
                    {
                        "turn": t.turn,
                        "output": t.output,
                        "feedback": t.feedback,
                        "aggregate_score": t.aggregate_score,
                        "verifier_results": [
                            {
                                "name": r.name,
                                "status": r.status.value,
                                "score": r.score,
                                "passed": r.passed,
                            }
                            for r in t.verifier_results
                        ],
                    }
                    for t in result.turns
                ],
            },
        }


# ── Utilities ────────────────────────────────────────────────────────────────

def _extract_changed_files(patch_diff: str) -> list[str]:
    files = []
    for line in patch_diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:]
            if path.startswith("b/"):
                path = path[2:]
            if path != "/dev/null":
                files.append(path)
    return files


def _error_result(instance_id: str, model_name: str, error: str) -> dict:
    return {
        "instance_id": instance_id,
        "model_patch": "",
        "model_name_or_path": model_name,
        "error": error,
        "eval_report": {},
        "resolved": None,
        "multi_turn": {"num_turns": 0, "stopped_early": False,
                       "final_aggregate_score": 0.0, "turns": []},
    }
