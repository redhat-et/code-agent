"""SWE-bench evaluation: patch generation and test execution.

Supports two strategies, both of which optionally support multi-turn iteration:
  - naive:  vLLM inference from pre-built prompts, followed by K8s-based test
            execution. With --max-turns > 1, the model receives intermediate
            feedback (from configurable verifiers) and can revise its patch.
  - agent:  Agentic loop using mini-swe-agent (or any agent via YAML config)
            running inside K8s Jobs with SWE-bench container images.
            With --max-turns > 1, the agent runs repeatedly, receiving feedback
            from intermediate verifiers between attempts.

Usage (naive, single-shot):
    python -m evals.swe_bench.run_patch_generation \
        --strategy naive \
        --vllm-url http://vllm-server:8000/v1 \
        --model-name Qwen/Qwen3-1.7B \
        --prompts s3://swe-bench/prompts/style-3-oracle.jsonl \
        --output-dir /tmp/swe-bench-results/

Usage (naive, multi-turn):
    python -m evals.swe_bench.run_patch_generation \
        --strategy naive \
        --vllm-url http://vllm-server:8000/v1 \
        --model-name Qwen/Qwen3-1.7B \
        --prompts s3://swe-bench/prompts/style-3-oracle.jsonl \
        --max-turns 3 \
        --intermediate-verifiers ast_check \
        --aggregator mean \
        --output-dir /tmp/swe-bench-results/

Usage (agent, single-shot):
    python -m evals.swe_bench.run_patch_generation \
        --strategy agent \
        --vllm-url http://vllm-server:8000/v1 \
        --model-name Qwen/Qwen3-1.7B \
        --agent-config evals/swe_bench/agents/mini_swe_agent.yaml \
        --output-dir /tmp/swe-bench-results/

Usage (agent, multi-turn):
    python -m evals.swe_bench.run_patch_generation \
        --strategy agent \
        --vllm-url http://vllm-server:8000/v1 \
        --model-name Qwen/Qwen3-1.7B \
        --agent-config evals/swe_bench/agents/mini_swe_agent.yaml \
        --max-turns 3 \
        --intermediate-verifiers ast_check \
        --output-dir /tmp/swe-bench-results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import ray
from swebench.harness.utils import load_swebench_dataset

from evals.common.s3_storage import download_file, upload_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Subset name mapping (for agent strategy) ────────────────────────

_SUBSET_REVERSE_MAP = {
    "princeton-nlp/SWE-Bench": "full",
    "princeton-nlp/SWE-bench": "full",
    "princeton-nlp/SWE-Bench_Verified": "verified",
    "princeton-nlp/SWE-bench_Verified": "verified",
    "princeton-nlp/SWE-Bench_Lite": "lite",
    "princeton-nlp/SWE-bench_Lite": "lite",
    "SWE-bench/SWE-bench_Lite": "lite",
    "SWE-bench/SWE-bench_Verified": "verified",
    "SWE-bench/SWE-Bench_Verified": "verified",
    "princeton-nlp/SWE-Bench_Multimodal": "multimodal",
    "swe-bench/SWE-Bench_Multilingual": "multilingual",
    "SWE-bench/SWE-smith": "smith",
    "nebius/SWE-rebench": "rebench",
}


def _resolve_subset_name(dataset: str) -> str:
    """Map a HuggingFace dataset name to mini-extra swebench --subset value."""
    return _SUBSET_REVERSE_MAP.get(dataset, dataset)


# ── Prediction I/O ──────────────────────────────────────────────────

def load_existing_predictions(output_dir: Path) -> set[str]:
    """Find instance IDs that already have predictions on disk."""
    completed = set()
    if not output_dir.exists():
        return completed
    for instance_dir in output_dir.iterdir():
        if instance_dir.is_dir():
            pred_file = instance_dir / "prediction.json"
            if pred_file.exists():
                try:
                    data = json.loads(pred_file.read_text())
                    completed.add(data["instance_id"])
                except (json.JSONDecodeError, KeyError, OSError):
                    pass
    return completed


def save_prediction(output_dir: Path, result: dict) -> None:
    """Save per-instance results to disk.

    Each instance gets a directory with:
      - prediction.json             (the prediction)
      - report.json                 (eval grading result)
      - pod_logs.txt                (full pod stdout, agent strategy only)
      - <instance_id>.traj.json     (agent trajectory, if available)
    """
    instance_id = result["instance_id"]
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)

    # prediction.json
    prediction = {
        "instance_id": result["instance_id"],
        "model_patch": result.get("model_patch", ""),
        "model_name_or_path": result.get("model_name_or_path", "unknown"),
    }
    (instance_dir / "prediction.json").write_text(
        json.dumps(prediction, indent=2)
    )

    # report.json
    report_data = result.get("eval_report") or {"error": result.get("error")}
    (instance_dir / "report.json").write_text(
        json.dumps(report_data, indent=2)
    )

    # pod_logs.txt (agent strategy only)
    if result.get("full_logs"):
        (instance_dir / "pod_logs.txt").write_text(result["full_logs"])

    # <instance_id>.traj.json (agent strategy only)
    if result.get("trajectory"):
        (instance_dir / f"{instance_id}.traj.json").write_text(
            result["trajectory"]
        )


def _resolve_prompts(source: str, output_dir: Path) -> Path:
    """Resolve prompts source to a local file path."""
    if source.startswith("s3://"):
        local_path = output_dir / "prompted_dataset.jsonl"
        download_file(source, local_path)
        return local_path
    return Path(source)


def _write_merged_predictions(output_dir: Path, dataset: list) -> Path:
    """Read all per-instance prediction files and merge into predictions.jsonl."""
    predictions_path = output_dir / "predictions.jsonl"
    count = 0

    with open(predictions_path, "w") as f:
        for instance in dataset:
            instance_id = instance["instance_id"]
            pred_file = output_dir / instance_id / "prediction.json"
            if pred_file.exists():
                try:
                    pred = json.loads(pred_file.read_text())
                    f.write(json.dumps({
                        "instance_id": pred["instance_id"],
                        "model_patch": pred.get("model_patch", ""),
                        "model_name_or_path": pred.get("model_name_or_path", "unknown"),
                    }) + "\n")
                    count += 1
                except (json.JSONDecodeError, KeyError, OSError) as e:
                    logger.warning(f"Failed to read prediction for {instance_id}: {e}")

    logger.info(f"Wrote {count} predictions to {predictions_path}")
    return predictions_path


def _upload_to_s3(local_path: Path, s3_uri: str | None) -> None:
    """Upload predictions file to S3 if an S3 URI was provided."""
    if not s3_uri:
        logger.info("No --s3-output specified, skipping S3 upload")
        return
    upload_file(local_path, s3_uri)


# ── Verifier set builder (shared across strategies) ─────────────────

def _build_verifier_set(args):
    """Build the VerifierSet and aggregator from CLI args."""
    from evals.common.score_aggregator import build_aggregator
    from evals.common.verifier_set import VerifierSet
    from evals.swe_bench.verifiers.unit_test_verifier import SWEBenchUnitTestVerifier
    from verifiers.ast_check import ASTCheckVerifier

    vset = VerifierSet()
    intermediate_names = set(args.intermediate_verifiers or [])

    if "ast_check" in intermediate_names:
        vset.add(ASTCheckVerifier(), run_intermediate=True, run_final=False)

    # SWE-bench test verifier always runs as final; optionally also intermediate
    run_swe_intermediate = "swe_test" in intermediate_names
    vset.add(
        SWEBenchUnitTestVerifier(
            swebench_namespace=args.swebench_namespace,
            image_registry=args.image_registry or None,
            timeout=float(args.job_timeout or 1800),
        ),
        run_intermediate=run_swe_intermediate,
        run_final=True,
    )

    aggregator = build_aggregator(args.aggregator)
    return vset, aggregator


# ── Strategy: naive (vLLM inference, single or multi-turn) ──────────

def _run_naive(args, pending: list, output_dir: Path) -> list[dict]:
    """Run naive vLLM inference via SWEBenchMultiTurnWorker Ray actors.

    With max_turns=1 this is equivalent to single-shot generation followed by
    K8s test execution. With max_turns>1 the model receives intermediate
    verifier feedback and can revise its patch.
    """
    from evals.swe_bench.multi_turn_worker import SWEBenchMultiTurnWorker
    from evals.swe_bench.prompt import load_prompt_dataset

    prompts_path = _resolve_prompts(args.prompts, output_dir)
    prompts = load_prompt_dataset(prompts_path)
    logger.info(f"Loaded {len(prompts)} prompts")

    missing = [inst["instance_id"] for inst in pending
               if inst["instance_id"] not in prompts]
    if missing:
        logger.warning(
            f"{len(missing)} instances have no prompt. "
            f"First missing: {missing[:5]}"
        )

    vset, aggregator = _build_verifier_set(args)
    num_workers = min(args.num_workers, len(pending))
    workers = [
        SWEBenchMultiTurnWorker.remote(
            vllm_urls=args.vllm_url,
            model_name=args.model_name,
            strategy="naive",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            verifier_set=vset,
            aggregator=aggregator,
            max_turns=args.max_turns,
            k8s_namespace=args.k8s_namespace,
            timeout=int(args.job_timeout or 1800),
            service_account=args.service_account,
            max_concurrent_jobs=args.max_concurrent_jobs,
            swebench_namespace=args.swebench_namespace,
            image_registry=args.image_registry or None,
        )
        for _ in range(num_workers)
    ]

    batches = [[] for _ in range(num_workers)]
    for i, instance in enumerate(pending):
        batches[i % num_workers].append(dict(instance))

    prompts_ref = ray.put(prompts)
    logger.info(
        f"Distributing {len(pending)} instances across {num_workers} workers "
        f"(max_turns={args.max_turns})"
    )
    futures = [
        worker.evaluate_batch.remote(batch, prompts_ref, args.run_id)
        for worker, batch in zip(workers, batches)
        if batch
    ]

    return _collect_results(futures)


# ── Strategy: agent (agentic loop via K8s Jobs, single or multi-turn)

def _run_agent(args, pending: list, output_dir: Path) -> list[dict]:
    """Run agent-based patch generation via AgentWorker or SWEBenchMultiTurnWorker.

    With max_turns=1, uses AgentWorker which is self-contained (generation +
    evaluation happen inside the K8s Job). With max_turns>1, uses
    SWEBenchMultiTurnWorker with strategy="agent" so intermediate verifier feedback
    can be injected between attempts.
    """
    from evals.swe_bench.agent_config import load_agent_config

    agent_config = load_agent_config(args.agent_config)
    logger.info(f"Agent: {agent_config.name}")

    subset = _resolve_subset_name(args.dataset)
    num_workers = min(args.num_workers, len(pending))

    batches = [[] for _ in range(num_workers)]
    for i, instance in enumerate(pending):
        batches[i % num_workers].append(dict(instance))

    if args.max_turns > 1:
        from evals.swe_bench.multi_turn_worker import SWEBenchMultiTurnWorker

        vset, aggregator = _build_verifier_set(args)

        workers = [
            SWEBenchMultiTurnWorker.remote(
                vllm_urls=args.vllm_url,
                model_name=args.model_name,
                strategy="agent",
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                verifier_set=vset,
                aggregator=aggregator,
                max_turns=args.max_turns,
                k8s_namespace=args.k8s_namespace,
                timeout=int(args.job_timeout or 1800),
                service_account=args.service_account,
                max_concurrent_jobs=args.max_concurrent_jobs,
                swebench_namespace=args.swebench_namespace,
                image_registry=args.image_registry or None,
                agent_config_dict=asdict(agent_config),
                model_api_key=args.model_api_key,
                subset=subset,
                split=args.split,
                step_limit=args.step_limit,
                cost_limit=args.cost_limit,
            )
            for _ in range(num_workers)
        ]

        prompts_ref = ray.put({})  # agent strategy does not use pre-built prompts
        logger.info(
            f"Distributing {len(pending)} instances across {num_workers} workers "
            f"(agent, max_turns={args.max_turns}, "
            f"max {args.max_concurrent_jobs} concurrent Jobs per worker)"
        )
        futures = [
            worker.evaluate_batch.remote(batch, prompts_ref, args.run_id)
            for worker, batch in zip(workers, batches)
            if batch
        ]
    else:
        from evals.swe_bench.agent_worker import AgentWorker

        workers = [
            AgentWorker.remote(
                agent_config_dict=asdict(agent_config),
                model_name=args.model_name,
                model_base_url=args.vllm_url[0],
                model_api_key=args.model_api_key,
                k8s_namespace=args.k8s_namespace,
                service_account=args.service_account,
                image_registry=args.image_registry,
                swebench_namespace=args.swebench_namespace,
                subset=subset,
                split=args.split,
                step_limit=args.step_limit,
                cost_limit=args.cost_limit,
                max_concurrent_jobs=args.max_concurrent_jobs,
                job_timeout=args.job_timeout,
                run_eval=args.run_eval,
            )
            for _ in range(num_workers)
        ]

        logger.info(
            f"Distributing {len(pending)} instances across {num_workers} workers "
            f"(agent, max {args.max_concurrent_jobs} concurrent Jobs per worker)"
        )
        futures = [
            worker.generate_patches.remote(batch, args.run_id)
            for worker, batch in zip(workers, batches)
            if batch
        ]

    return _collect_results(futures)


def _collect_results(futures: list) -> list[dict]:
    """Collect results from Ray futures one at a time."""
    all_results = []
    pending_futures = list(futures)
    while pending_futures:
        ready, pending_futures = ray.wait(pending_futures, num_returns=1)
        try:
            batch_results = ray.get(ready[0])
        except Exception as e:
            logger.error(f"Worker batch failed: {e}")
            continue
        all_results.extend(batch_results)
    return all_results


# ── Dry run ─────────────────────────────────────────────────────────

def _dry_run(args, dataset: list) -> None:
    """Print the K8s Job YAML for the first instance and exit."""
    from evals.swe_bench.agent_config import load_agent_config
    from evals.swe_bench.agent_worker import generate_job_yaml

    agent_config = load_agent_config(args.agent_config)
    if args.job_timeout > 0:
        agent_config.job_timeout = args.job_timeout
    instance = dict(dataset[0])
    subset = _resolve_subset_name(args.dataset)

    yaml_str = generate_job_yaml(
        instance=instance,
        run_id=args.run_id,
        agent_config=agent_config,
        model_name=args.model_name,
        model_base_url=args.vllm_url[0],
        model_api_key=args.model_api_key,
        k8s_namespace=args.k8s_namespace or "default",
        service_account=args.service_account,
        image_registry=args.image_registry,
        swebench_namespace=args.swebench_namespace,
        subset=subset,
        split=args.split,
        step_limit=args.step_limit,
        cost_limit=args.cost_limit,
        run_eval=args.run_eval,
    )
    print(yaml_str)


# ── CLI ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SWE-bench patches (naive or agent strategy, single or multi-turn)"
    )

    # Common args
    common = parser.add_argument_group("common")
    common.add_argument("--strategy", type=str, default="naive",
                        choices=["naive", "agent"],
                        help="Patch generation strategy")
    common.add_argument("--vllm-url", type=str, nargs="+", required=True,
                        help="vLLM OpenAI-compatible base URL(s)")
    common.add_argument("--model-name", type=str, required=True,
                        help="Model name as registered in vLLM")
    common.add_argument("--dataset", type=str, default="SWE-bench/SWE-bench_Verified",
                        help="HuggingFace dataset name")
    common.add_argument("--split", type=str, default="test",
                        help="Dataset split")
    common.add_argument("--output-dir", type=str, required=True,
                        help="Local directory for results")
    common.add_argument("--s3-output", type=str, default=None,
                        help="S3 URI to upload predictions.jsonl")
    common.add_argument("--num-workers", type=int, default=2,
                        help="Number of Ray workers")
    common.add_argument("--instance-limit", type=int, default=0,
                        help="Max instances to evaluate (0 = no limit)")
    common.add_argument("--run-id", type=str, default="eval-run",
                        help="Unique run identifier")
    common.add_argument("--max-turns", type=int, default=1,
                        help="Max generation attempts per instance "
                             "(1 = single-shot, >1 = multi-turn with feedback)")
    common.add_argument("--intermediate-verifiers", type=str, nargs="*", default=[],
                        choices=["ast_check"],
                        help="Verifiers to run after each intermediate turn. "
                             "Options: ast_check")
    common.add_argument("--aggregator", type=str, default="mean",
                        choices=["mean", "min", "weighted_sum"],
                        help="Score aggregation strategy for multi-turn")

    # Naive strategy args
    naive = parser.add_argument_group("naive strategy")
    naive.add_argument("--prompts", type=str, default=None,
                       help="S3 URI or local path to prompted dataset JSONL "
                            "(required for naive strategy)")
    naive.add_argument("--max-tokens", type=int, default=16000,
                       help="Max tokens for patch generation")
    naive.add_argument("--temperature", type=float, default=0.15,
                       help="Sampling temperature")

    # Agent strategy args
    agent = parser.add_argument_group("agent strategy")
    agent.add_argument("--agent-config", type=str,
                       default="evals/swe_bench/agents/mini_swe_agent.yaml",
                       help="Path to agent config YAML")
    agent.add_argument("--model-api-key", type=str, default="dummy",
                       help="API key for the model endpoint")
    agent.add_argument("--step-limit", type=int, default=100,
                       help="Max agent steps per instance")
    agent.add_argument("--cost-limit", type=float, default=3.0,
                       help="Max cost in dollars per instance")
    agent.add_argument("--k8s-namespace", type=str, default=None,
                       help="K8s namespace (auto-detected if not set)")
    agent.add_argument("--service-account", type=str, default="swe-bench-eval",
                       help="K8s ServiceAccount for agent Jobs")
    agent.add_argument("--image-registry", type=str, default="",
                       help="Internal registry instead of DockerHub")
    agent.add_argument("--swebench-namespace", type=str, default="swebench",
                       help="DockerHub namespace for SWE-bench images")
    agent.add_argument("--max-concurrent-jobs", type=int, default=4,
                       help="Max concurrent K8s Jobs per worker")
    agent.add_argument("--job-timeout", type=int, default=0,
                       help="K8s Job timeout in seconds (0 = use agent config default)")
    agent.add_argument("--run-eval", action="store_true", default=True,
                       help="Run in-container evaluation after agent (default: True, "
                            "only applies when max_turns=1)")
    agent.add_argument("--skip-eval", action="store_true",
                       help="Skip in-container evaluation (only applies when max_turns=1)")

    # Debug
    agent.add_argument("--dry-run", action="store_true",
                       help="Print K8s Job YAML for first instance and exit")

    return parser.parse_args()


def main():
    args = _parse_args()
    start_time = time.time()

    if args.skip_eval:
        args.run_eval = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset} split={args.split}")
    dataset = load_swebench_dataset(args.dataset, split=args.split)
    logger.info(f"Loaded {len(dataset)} instances")

    # Apply instance limit
    if args.instance_limit > 0:
        dataset = dataset[:args.instance_limit]
        logger.info(f"Limited to {len(dataset)} instances")

    # Dry run (agent strategy only)
    if args.strategy == "agent" and args.dry_run:
        if not dataset:
            logger.error("No instances for dry run")
            return
        _dry_run(args, dataset)
        return

    # Filter out already-completed instances (resumability)
    completed = load_existing_predictions(output_dir)
    if completed:
        logger.info(f"Skipping {len(completed)} already-completed instances")
    pending = [inst for inst in dataset if inst["instance_id"] not in completed]
    logger.info(f"Generating patches for {len(pending)} instances")

    if not pending:
        logger.info("All instances already completed")
        predictions_path = _write_merged_predictions(output_dir, dataset)
        _upload_to_s3(predictions_path, args.s3_output)
        return

    # Validate strategy-specific requirements
    if args.strategy == "naive" and not args.prompts:
        logger.error("--prompts is required for naive strategy")
        return

    # Run
    ray.init()

    if args.strategy == "naive":
        all_results = _run_naive(args, pending, output_dir)
    else:
        all_results = _run_agent(args, pending, output_dir)

    # Save per-instance results
    for result in all_results:
        save_prediction(output_dir, result)

    # Merge predictions
    predictions_path = _write_merged_predictions(output_dir, dataset)

    # Upload to S3
    _upload_to_s3(predictions_path, args.s3_output)

    # Write aggregate results.json when evaluation was performed
    uses_multi_turn = args.max_turns > 1
    agent_with_eval = args.strategy == "agent" and args.max_turns == 1 and args.run_eval
    if uses_multi_turn or agent_with_eval:
        _write_aggregate_results(output_dir, all_results)

    # Summary
    elapsed = time.time() - start_time
    total = len(dataset)
    errors = sum(1 for r in all_results if r.get("error"))
    patches_generated = sum(
        1 for inst in dataset
        if (output_dir / inst["instance_id"] / "prediction.json").exists()
    )

    turn_label = f", max_turns={args.max_turns}" if args.max_turns > 1 else ""
    logger.info("=" * 60)
    logger.info(f"Evaluation complete ({args.strategy}{turn_label})")
    logger.info("=" * 60)
    logger.info(f"  Patches generated:  {patches_generated}/{total}")
    logger.info(f"  Errors:             {errors}")

    if uses_multi_turn or agent_with_eval:
        resolved = sum(1 for r in all_results if r.get("resolved") is True)
        evaluated = sum(1 for r in all_results if r.get("resolved") is not None)
        rate = resolved / evaluated if evaluated > 0 else 0
        logger.info(f"  Evaluated:          {evaluated}")
        logger.info(f"  Resolved:           {resolved}")
        logger.info(f"  Resolve rate:       {rate:.1%}")

    if uses_multi_turn:
        turn_counts = [
            r.get("multi_turn", {}).get("num_turns", 1)
            for r in all_results if not r.get("error")
        ]
        if turn_counts:
            avg_turns = sum(turn_counts) / len(turn_counts)
            early_exits = sum(
                1 for r in all_results
                if r.get("multi_turn", {}).get("stopped_early", False)
            )
            logger.info(f"  Avg turns:          {avg_turns:.2f}")
            logger.info(f"  Early exits:        {early_exits}/{len(turn_counts)}")

    logger.info(f"  Total time:         {elapsed / 60:.1f} minutes")
    logger.info("=" * 60)

    # MLflow tracking (optional)
    if os.environ.get("MLFLOW_TRACKING_URI"):
        _log_to_mlflow(args, total, patches_generated, errors, all_results,
                       predictions_path, output_dir)

    ray.shutdown()


def _write_aggregate_results(output_dir: Path, all_results: list[dict]) -> None:
    """Write aggregate results.json from eval results."""
    evaluated = [r for r in all_results if r.get("resolved") is not None]
    resolved = [r for r in evaluated if r.get("resolved") is True]
    errors = [r for r in all_results if r.get("error")]

    report = {
        "total_instances": len(all_results),
        "evaluated": len(evaluated),
        "resolved_instances": len(resolved),
        "unresolved_instances": len(evaluated) - len(resolved),
        "error_instances": len(errors),
        "resolve_rate": len(resolved) / len(evaluated) if evaluated else 0,
        "resolved_ids": [r["instance_id"] for r in resolved],
        "error_ids": [r["instance_id"] for r in errors],
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Aggregate results written to {results_path}")


def _log_to_mlflow(args, total, patches_generated, errors, all_results,
                   predictions_path, output_dir: Path):
    """Log results to MLflow (optional, activated by MLFLOW_TRACKING_URI).

    Logs:
      - Parameters: strategy, model, dataset, agent config, etc.
      - Metrics: total instances, patches generated, resolve rate, etc.
      - Artifacts: predictions.jsonl, results.json, and per-instance
        directories (prediction.json, report.json, pod_logs.txt, traj.json)
    """
    try:
        import mlflow

        s3_endpoint = os.environ.get("S3_ENDPOINT_URL") or os.environ.get("MINIO_ENDPOINT_URL")
        if s3_endpoint and not os.environ.get("MLFLOW_S3_ENDPOINT_URL"):
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint

        mlflow.set_experiment("swe-bench-eval")
        with mlflow.start_run(run_name=f"{args.run_id}-{args.strategy}",
                              tags={"run_id": args.run_id,
                                    "strategy": args.strategy}):
            params = {
                "strategy": args.strategy,
                "model_name": args.model_name,
                "dataset": args.dataset,
                "split": args.split,
                "run_id": args.run_id,
                "num_workers": args.num_workers,
                "instance_limit": args.instance_limit,
                "max_turns": args.max_turns,
            }
            if args.strategy == "naive":
                params.update({
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                })
            else:
                params.update({
                    "agent_config": args.agent_config,
                    "step_limit": args.step_limit,
                    "cost_limit": args.cost_limit,
                })
            if args.max_turns > 1:
                params.update({
                    "intermediate_verifiers": ",".join(args.intermediate_verifiers or []),
                    "aggregator": args.aggregator,
                })

            mlflow.log_params(params)

            uses_multi_turn = args.max_turns > 1
            agent_with_eval = args.strategy == "agent" and args.max_turns == 1 and args.run_eval
            metrics = {
                "total_instances": total,
                "patches_generated": patches_generated,
                "generation_errors": errors,
            }
            if uses_multi_turn or agent_with_eval:
                resolved = sum(1 for r in all_results if r.get("resolved") is True)
                evaluated = sum(1 for r in all_results if r.get("resolved") is not None)
                metrics["evaluated"] = evaluated
                metrics["resolved"] = resolved
                metrics["resolve_rate"] = resolved / evaluated if evaluated else 0

            mlflow.log_metrics(metrics)

            # Log top-level artifacts
            mlflow.log_artifact(str(predictions_path))
            results_json = output_dir / "results.json"
            if results_json.exists():
                mlflow.log_artifact(str(results_json))

            # Log per-instance directories as artifacts
            for instance_dir in sorted(output_dir.iterdir()):
                if instance_dir.is_dir() and (instance_dir / "prediction.json").exists():
                    mlflow.log_artifacts(str(instance_dir), artifact_path=instance_dir.name)

            logger.info("Results logged to MLflow")
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
