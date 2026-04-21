#!/bin/bash
#
# Phase 2: Run SWE-bench test execution via K8s Jobs.
#
# Reads predictions from S3/MinIO (output of Phase 1), distributes test
# execution across Ray workers. Each worker creates K8s Jobs using
# pre-built SWE-bench container images. Results are graded, aggregated,
# and uploaded to S3/MinIO.
#
# Prerequisites:
#   - RayCluster deployed: oc apply -f evals/swe_bench/deploy/raycluster-test-exec.yaml
#   - RBAC applied: oc apply -f evals/swe_bench/deploy/rbac.yaml
#   - predictions.jsonl from Phase 1 (in S3)
#   - MinIO credentials secret (same as Phase 1)
#   - Port-forward active: oc port-forward svc/swe-bench-test-exec-head-svc 8265:8265
#   - (optional) Images mirrored to internal registry:
#       oc apply -f evals/swe_bench/deploy/job-mirror-images.yaml
#
# Usage:
#   bash run_swe_bench_phase2.sh
#
# Quick test with 16 instances:
#   INSTANCE_LIMIT=16 bash run_swe_bench_phase2.sh
#
# Verify harness with gold patches (skips Phase 1 entirely):
#   PREDICTIONS=gold INSTANCE_LIMIT=16 RUN_ID=gold-test bash run_swe_bench_phase2.sh
#
# Use internal registry (after mirroring images):
#   IMAGE_REGISTRY=image-registry.openshift-image-registry.svc:5000/code-agent \
#     bash run_swe_bench_phase2.sh

set -euo pipefail

# ── Configurable ────────────────────────────────────────────────
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
DATASET="${DATASET:-SWE-bench/SWE-bench_Verified}"
RUN_ID="${RUN_ID:-eval-run}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/swe-bench-results/${RUN_ID}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-4}"
K8S_NAMESPACE="${K8S_NAMESPACE:-}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-swe-bench-eval}"
TIMEOUT="${TIMEOUT:-1800}"
INSTANCE_LIMIT="${INSTANCE_LIMIT:-0}"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-}"
S3_BUCKET="${S3_BUCKET:-swe-bench}"
PREDICTIONS="${PREDICTIONS:-s3://${S3_BUCKET}/runs/${RUN_ID}/predictions.jsonl}"
S3_OUTPUT="${S3_OUTPUT:-s3://${S3_BUCKET}/runs/${RUN_ID}/results.json}"
# MLflow tracking (optional). Set to the in-cluster MLflow service URL
# to enable experiment tracking. Unset to disable.
# e.g. MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-}"

if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

# Build command args
CMD_ARGS=(
    python3 -m evals.swe_bench.run_test_execution
    --predictions "${PREDICTIONS}"
    --dataset "${DATASET}"
    --output-dir "${OUTPUT_DIR}"
    --run-id "${RUN_ID}"
    --num-workers "${NUM_WORKERS}"
    --max-concurrent-jobs "${MAX_CONCURRENT_JOBS}"
    --service-account "${SERVICE_ACCOUNT}"
    --timeout "${TIMEOUT}"
    --instance-limit "${INSTANCE_LIMIT}"
    --s3-output "${S3_OUTPUT}"
)

# Only pass --k8s-namespace if explicitly set (otherwise auto-detected in-cluster)
if [[ -n "${K8S_NAMESPACE}" ]]; then
    CMD_ARGS+=(--k8s-namespace "${K8S_NAMESPACE}")
fi

# Use internal registry instead of DockerHub when set
if [[ -n "${IMAGE_REGISTRY}" ]]; then
    CMD_ARGS+=(--image-registry "${IMAGE_REGISTRY}")
fi

# Pass MLflow tracking URI via Ray runtime env so workers pick it up
ENV_ARGS=()
if [[ -n "${MLFLOW_TRACKING_URI}" ]]; then
    ENV_ARGS+=(--runtime-env-json "{\"env_vars\": {\"MLFLOW_TRACKING_URI\": \"${MLFLOW_TRACKING_URI}\"}}")
fi

ray job submit \
    --address="${RAY_ADDRESS}" \
    "${ENV_ARGS[@]}" \
    -- "${CMD_ARGS[@]}"
