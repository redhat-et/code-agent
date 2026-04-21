"""Phase 1: Generate patches for SWE-bench instances via vLLM.

Loads pre-built prompts (from S3 or local file, created by
build_prompt_dataset.py), distributes inference across Ray workers,
and saves predictions.jsonl to S3/MinIO for Phase 2.

Results are optionally logged to MLflow when MLFLOW_TRACKING_URI is set.

Prerequisites:
    Run build_prompt_dataset.py first to create the prompted dataset.

Usage:
    python run_patch_generation.py \
        --vllm-url http://vllm-server:8000/v1 \
        --model-name Qwen/Qwen3-1.7B \
        --prompts s3://swe-bench/prompts/style-3-oracle.jsonl \
        --output-dir /tmp/swe-bench-results/ \
        --s3-output s3://swe-bench/runs/run-001/predictions.jsonl

    # With MLflow tracking:
    MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
    python run_patch_generation.py ... --run-id my-eval-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import ray
from swebench.harness.utils import load_swebench_dataset

from evals.common.s3_storage import download_file, upload_file
from evals.swe_bench.patch_worker import PatchWorker
from evals.swe_bench.prompt import load_prompt_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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


def save_prediction(output_dir: Path, prediction: dict) -> None:
    """Save a single prediction to disk for resumability."""
    instance_id = prediction["instance_id"]
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    (instance_dir / "prediction.json").write_text(json.dumps(prediction, indent=2))


def _resolve_prompts(source: str, output_dir: Path) -> Path:
    """Resolve prompts source to a local file path.

    If source is an S3 URI, download to output_dir.
    Otherwise treat as a local path.
    """
    if source.startswith("s3://"):
        local_path = output_dir / "prompted_dataset.jsonl"
        download_file(source, local_path)
        return local_path
    return Path(source)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Generate SWE-bench patches")
    parser.add_argument("--vllm-url", type=str, nargs="+", required=True,
                        help="vLLM OpenAI-compatible base URL(s)")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name as registered in vLLM")
    parser.add_argument("--dataset", type=str, default="SWE-bench/SWE-bench_Lite",
                        help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split")
    parser.add_argument("--prompts", type=str, required=True,
                        help="S3 URI or local path to the prompted dataset JSONL "
                             "(from build_prompt_dataset.py)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Local directory for intermediate files")
    parser.add_argument("--s3-output", type=str, default=None,
                        help="S3 URI to upload predictions.jsonl "
                             "(e.g. s3://swe-bench/runs/run-001/predictions.jsonl)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of Ray workers")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens for patch generation")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--instance-limit", type=int, default=0,
                        help="Max instances to evaluate (0 = no limit)")
    parser.add_argument("--run-id", type=str, default="eval-run",
                        help="Unique run identifier (shared with Phase 2 for MLflow)")
    args = parser.parse_args()

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

    # Load pre-built prompts (from S3 or local file)
    prompts_path = _resolve_prompts(args.prompts, output_dir)
    prompts = load_prompt_dataset(prompts_path)
    logger.info(f"Loaded {len(prompts)} prompts")

    # Check that we have prompts for all pending instances
    missing = [inst["instance_id"] for inst in pending
               if inst["instance_id"] not in prompts]
    if missing:
        logger.warning(
            f"{len(missing)} instances have no prompt in the dataset. "
            f"Run build_prompt_dataset.py to generate them. "
            f"First missing: {missing[:5]}"
        )

    # Distribute inference across Ray workers
    ray.init()

    num_workers = min(args.num_workers, len(pending))
    workers = [
        PatchWorker.remote(
            vllm_urls=args.vllm_url,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        for _ in range(num_workers)
    ]

    # Split instances across workers
    batches = [[] for _ in range(num_workers)]
    for i, instance in enumerate(pending):
        batches[i % num_workers].append(dict(instance))

    # Prompts dict shared across workers via Ray object store
    prompts_ref = ray.put(prompts)
    logger.info(f"Distributing {len(pending)} instances across {num_workers} workers")
    futures = [
        worker.generate_patches.remote(batch, prompts_ref)
        for worker, batch in zip(workers, batches)
        if batch
    ]

    # Collect results -- process futures one at a time so a single
    # worker failure doesn't discard results from other workers
    all_results = []
    pending_futures = list(futures)
    while pending_futures:
        ready, pending_futures = ray.wait(pending_futures, num_returns=1)
        try:
            batch_results = ray.get(ready[0])
        except Exception as e:
            logger.error(f"Patch worker batch failed: {e}")
            continue

        for result in batch_results:
            save_prediction(output_dir, result)
            all_results.append(result)

    # Merge all predictions (completed + new) into predictions.jsonl
    predictions_path = _write_merged_predictions(output_dir, dataset)

    # Upload to S3/MinIO
    _upload_to_s3(predictions_path, args.s3_output)

    # Summary
    total = len(dataset)
    errors = sum(1 for r in all_results if r.get("error"))
    # Count all patches present on disk (new + resumed) for accurate metrics.
    patches_generated = sum(
        1 for inst in dataset
        if (output_dir / inst["instance_id"] / "prediction.json").exists()
    )
    logger.info(
        f"Phase 1 complete: {patches_generated} patches generated, "
        f"{errors} errors, {total} total instances in dataset"
    )

    # ── MLflow tracking (optional) ──────────────────────────────
    # Logs params, metrics, and the predictions artifact to MLflow.
    # Activated when MLFLOW_TRACKING_URI is set in the environment.
    # The run is tagged with the run_id so Phase 2 can resume it.
    if os.environ.get("MLFLOW_TRACKING_URI"):
        try:
            import mlflow

            # Ensure MLflow's artifact client talks to the in-cluster MinIO,
            # not AWS. The S3_ENDPOINT_URL env var is set on the Ray head
            # from the minio-credentials secret.
            s3_endpoint = os.environ.get("S3_ENDPOINT_URL") or os.environ.get("MINIO_ENDPOINT_URL")
            if s3_endpoint and not os.environ.get("MLFLOW_S3_ENDPOINT_URL"):
                os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint

            mlflow.set_experiment("swe-bench-eval")
            with mlflow.start_run(run_name=f"{args.run_id}-phase1",
                                  tags={"run_id": args.run_id, "phase": "1"}):
                mlflow.log_params({
                    "model_name": args.model_name,
                    "dataset": args.dataset,
                    "split": args.split,
                    "run_id": args.run_id,
                    "num_workers": args.num_workers,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "instance_limit": args.instance_limit,
                })
                mlflow.log_metrics({
                    "total_instances": total,
                    "patches_generated": patches_generated,
                    "generation_errors": errors,
                })
                mlflow.log_artifact(str(predictions_path))
                logger.info("Phase 1 results logged to MLflow")
        except Exception as e:
            logger.warning(f"MLflow logging failed (non-fatal): {e}")

    ray.shutdown()


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
                        "model_patch": pred["model_patch"],
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


if __name__ == "__main__":
    main()
