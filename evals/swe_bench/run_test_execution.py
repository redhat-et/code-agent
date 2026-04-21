"""Phase 2: Run SWE-bench test execution via K8s Jobs.

Reads predictions from Phase 1 (from S3/MinIO or a local path),
distributes test execution across Ray workers, each managing
concurrent K8s Jobs using pre-built SWE-bench container images.
Results are graded, aggregated, and uploaded to S3/MinIO.

Results are optionally logged to MLflow when MLFLOW_TRACKING_URI is set.

Usage:
    python run_test_execution.py \
        --predictions s3://swe-bench/runs/run-001/predictions.jsonl \
        --output-dir /tmp/swe-bench-results/ \
        --s3-output s3://swe-bench/runs/run-001/results.json
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
from evals.swe_bench.grader import InstanceResult, aggregate_reports
from evals.swe_bench.test_worker import TestWorker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_predictions(predictions_path: Path) -> list[dict]:
    """Load predictions from a JSONL file."""
    predictions = []
    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


def _build_gold_predictions(dataset: list, output_dir: Path, limit: int = 0) -> Path:
    """Build predictions from gold patches in the dataset.

    Uses the dataset's own 'patch' field as the model prediction.
    This is the standard way to verify the eval harness is working.

    Args:
        dataset: SWE-bench dataset instances.
        output_dir: Directory to write the gold predictions file.
        limit: Max instances (0 = no limit).

    Returns:
        Path to the generated predictions.jsonl.
    """
    instances = dataset[:limit] if limit > 0 else dataset
    predictions_path = output_dir / "gold_predictions.jsonl"

    with open(predictions_path, "w") as f:
        for inst in instances:
            f.write(json.dumps({
                "instance_id": inst["instance_id"],
                "model_patch": inst["patch"],
                "model_name_or_path": "gold",
            }) + "\n")

    logger.info(f"Built {len(instances)} gold predictions from dataset")
    return predictions_path


def _resolve_predictions(source: str, output_dir: Path) -> Path:
    """Resolve predictions source to a local file path.

    Handles:
      - "gold": special value, resolved later after dataset is loaded
      - "s3://...": download from S3/MinIO
      - local path: used as-is

    Returns Path for s3/local, or None for gold (handled separately).
    """
    if source == "gold":
        # Sentinel -- caller handles this after loading the dataset
        return None
    if source.startswith("s3://"):
        local_path = output_dir / "predictions.jsonl"
        download_file(source, local_path)
        return local_path
    return Path(source)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Run SWE-bench test execution")
    parser.add_argument("--predictions", type=str, required=True,
                        help="'gold' to use dataset ground-truth patches, "
                             "S3 URI (s3://...), or local path to predictions.jsonl")
    parser.add_argument("--dataset", type=str, default="SWE-bench/SWE-bench_Lite",
                        help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write results")
    parser.add_argument("--run-id", type=str, default="eval-run",
                        help="Unique run identifier")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of Ray workers")
    parser.add_argument("--max-concurrent-jobs", type=int, default=4,
                        help="Max concurrent K8s Jobs per worker")
    parser.add_argument("--k8s-namespace", type=str, default=None,
                        help="K8s namespace for eval Jobs (auto-detected if not set)")
    parser.add_argument("--service-account", type=str, default="swe-bench-eval",
                        help="K8s ServiceAccount for eval Job pods")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-instance timeout in seconds")
    parser.add_argument("--swebench-namespace", type=str, default="swebench",
                        help="DockerHub namespace for pre-built images")
    parser.add_argument("--image-registry", type=str, default=None,
                        help="Pull SWE-bench images from this registry instead of "
                             "DockerHub (e.g. image-registry.openshift-image-registry"
                             ".svc:5000/code-agent)")
    parser.add_argument("--instance-limit", type=int, default=0,
                        help="Max instances to evaluate (0 = no limit)")
    parser.add_argument("--s3-output", type=str, default=None,
                        help="S3 URI to upload results.json "
                             "(e.g. s3://swe-bench/runs/run-001/results.json)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset for instance metadata (needed by make_test_spec and gold mode)
    logger.info(f"Loading dataset: {args.dataset} split={args.split}")
    dataset = load_swebench_dataset(args.dataset, split=args.split)
    instances_by_id = {inst["instance_id"]: dict(inst) for inst in dataset}
    logger.info(f"Loaded {len(dataset)} dataset instances")

    # Load predictions (gold, S3, or local)
    is_gold = args.predictions == "gold"
    if is_gold:
        logger.info("Using GOLD patches from dataset (harness verification mode)")
        predictions_path = _build_gold_predictions(
            dataset, output_dir, limit=args.instance_limit,
        )
    else:
        predictions_path = _resolve_predictions(args.predictions, output_dir)

    logger.info(f"Loading predictions from {predictions_path}")
    predictions = load_predictions(predictions_path)
    logger.info(f"Loaded {len(predictions)} predictions")

    # Apply instance limit (for non-gold mode; gold already applies it)
    if not is_gold and args.instance_limit > 0:
        predictions = predictions[:args.instance_limit]
        logger.info(f"Limited to {len(predictions)} instances")

    if not predictions:
        logger.info("No predictions to evaluate")
        return

    # Initialize Ray
    ray.init()

    # Check for already-completed results from a previous run (resumability).
    # This runs on the head node where results are persisted.
    completed_results = {}
    for pred in predictions:
        iid = pred["instance_id"]
        report_path = output_dir / iid / "report.json"
        if report_path.exists():
            try:
                completed_results[iid] = json.loads(report_path.read_text())
                logger.info(f"Skipping {iid} (already completed)")
            except (json.JSONDecodeError, OSError):
                pass

    pending = [p for p in predictions if p["instance_id"] not in completed_results]
    logger.info(
        f"{len(completed_results)} already completed, "
        f"{len(pending)} pending"
    )

    # Create workers and distribute pending predictions
    all_results = list(completed_results.values())

    if pending:
        num_workers = min(args.num_workers, len(pending))
        workers = [
            TestWorker.remote(
                k8s_namespace=args.k8s_namespace,
                timeout=args.timeout,
                service_account=args.service_account,
                max_concurrent_jobs=args.max_concurrent_jobs,
                swebench_namespace=args.swebench_namespace,
                image_registry=args.image_registry,
            )
            for _ in range(num_workers)
        ]

        # Split pending predictions across workers
        batches = [[] for _ in range(num_workers)]
        for i, pred in enumerate(pending):
            batches[i % num_workers].append(pred)

        # Submit work -- instances dict is shared via Ray object store
        # (serialized once, not per-worker)
        instances_ref = ray.put(instances_by_id)
        logger.info(
            f"Distributing {len(pending)} instances across {num_workers} workers "
            f"({args.max_concurrent_jobs} concurrent jobs per worker)"
        )
        futures = [
            worker.evaluate_batch.remote(batch, instances_ref, args.run_id)
            for worker, batch in zip(workers, batches)
            if batch
        ]

        # Collect results -- process futures one at a time so a single
        # worker failure doesn't discard results from other workers
        pending_futures = list(futures)
        while pending_futures:
            ready, pending_futures = ray.wait(pending_futures, num_returns=1)
            try:
                batch_results = ray.get(ready[0])
            except Exception as e:
                logger.error(f"Test worker batch failed: {e}")
                continue

            for result in batch_results:
                iid = result["instance_id"]
                instance_dir = output_dir / iid
                instance_dir.mkdir(parents=True, exist_ok=True)
                (instance_dir / "report.json").write_text(
                    json.dumps(result, indent=2)
                )
                all_results.append(result)

    # Aggregate
    instance_results = [
        InstanceResult(
            instance_id=r["instance_id"],
            resolved=r.get("resolved", False),
            patch_exists=r.get("patch_exists", False),
            patch_successfully_applied=r.get("patch_successfully_applied", False),
            error=r.get("error"),
            tests_status=r.get("tests_status"),
        )
        for r in all_results
    ]
    report = aggregate_reports(instance_results)

    # Write final report
    full_report = {
        "summary": report.to_dict(),
        "instance_results": all_results,
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(full_report, indent=2))
    logger.info(f"Wrote full report to {results_path}")

    # Upload to S3/MinIO
    if args.s3_output:
        upload_file(results_path, args.s3_output)

    # Summary
    logger.info(
        f"Phase 2 complete: "
        f"{report.resolved_instances}/{report.total_instances} resolved "
        f"({report.resolve_rate:.1%}), "
        f"{report.error_instances} errors, "
        f"{report.empty_patch_instances} empty patches"
    )

    if report.resolved_ids:
        logger.info(f"Resolved: {report.resolved_ids}")

    # ── MLflow tracking (optional) ──────────────────────────────
    # Logs params, metrics, and the results artifact to MLflow.
    # Activated when MLFLOW_TRACKING_URI is set in the environment.
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
            with mlflow.start_run(run_name=f"{args.run_id}-phase2",
                                  tags={"run_id": args.run_id, "phase": "2"}):
                mlflow.log_params({
                    "dataset": args.dataset,
                    "split": args.split,
                    "run_id": args.run_id,
                    "predictions_source": args.predictions,
                    "num_workers": args.num_workers,
                    "max_concurrent_jobs": args.max_concurrent_jobs,
                    "timeout": args.timeout,
                    "instance_limit": args.instance_limit,
                })
                mlflow.log_metrics({
                    "total_instances": report.total_instances,
                    "resolved_instances": report.resolved_instances,
                    "unresolved_instances": report.unresolved_instances,
                    "error_instances": report.error_instances,
                    "empty_patch_instances": report.empty_patch_instances,
                    "resolve_rate": report.resolve_rate,
                })
                mlflow.log_artifact(str(results_path))
                logger.info("Phase 2 results logged to MLflow")
        except Exception as e:
            logger.warning(f"MLflow logging failed (non-fatal): {e}")

    ray.shutdown()


if __name__ == "__main__":
    main()
