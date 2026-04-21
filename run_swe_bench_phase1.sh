#!/bin/bash
#
# Phase 1: Generate patches for SWE-bench instances via vLLM.
#
# Loads pre-built prompts from S3 (created by run_build_prompts.sh),
# distributes inference across Ray workers, and uploads predictions
# to S3/MinIO for Phase 2.
#
# Prerequisites:
#   - Prompted dataset built: oc apply -f evals/swe_bench/deploy/job-build-prompts.yaml
#   - RayCluster deployed: oc apply -f evals/swe_bench/deploy/raycluster-patch-gen.yaml
#   - vLLM server deployed: oc apply -f inference/deploy/vllm-server-deployment.yaml
#   - MinIO credentials secret configured
#   - Port-forward active: oc port-forward svc/swe-bench-patch-gen-head-svc 8265:8265
#
# Usage:
#   bash run_swe_bench_phase1.sh
#
# Quick test with 16 instances:
#   INSTANCE_LIMIT=16 bash run_swe_bench_phase1.sh

set -euo pipefail

# ── Configurable ────────────────────────────────────────────────
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
VLLM_URL="${VLLM_URL:-http://vllm-server:8000/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
DATASET="${DATASET:-SWE-bench/SWE-bench_Verified}"
NUM_WORKERS="${NUM_WORKERS:-2}"
INSTANCE_LIMIT="${INSTANCE_LIMIT:-0}"
RUN_ID="${RUN_ID:-eval-run}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/swe-bench-results/${RUN_ID}}"
S3_BUCKET="${S3_BUCKET:-swe-bench}"
S3_OUTPUT="${S3_OUTPUT:-s3://${S3_BUCKET}/runs/${RUN_ID}/predictions.jsonl}"
PROMPTS="${PROMPTS:-s3://${S3_BUCKET}/verified/prompts/style-3-oracle.jsonl}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
TEMPERATURE="${TEMPERATURE:-0.15}"
# MLflow tracking (optional). Set to the in-cluster MLflow service URL
# to enable experiment tracking. Unset to disable.
# e.g. MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-}"

if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

# Build command args
CMD_ARGS=(
    python3 -m evals.swe_bench.run_patch_generation
    --vllm-url "${VLLM_URL}"
    --model-name "${MODEL_NAME}"
    --dataset "${DATASET}"
    --prompts "${PROMPTS}"
    --num-workers "${NUM_WORKERS}"
    --output-dir "${OUTPUT_DIR}"
    --instance-limit "${INSTANCE_LIMIT}"
    --s3-output "${S3_OUTPUT}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --run-id "${RUN_ID}"
)

# Pass MLflow tracking URI via Ray runtime env so workers pick it up
ENV_ARGS=()
if [[ -n "${MLFLOW_TRACKING_URI}" ]]; then
    ENV_ARGS+=(--runtime-env-json "{\"env_vars\": {\"MLFLOW_TRACKING_URI\": \"${MLFLOW_TRACKING_URI}\"}}")
fi

ray job submit \
    --address="${RAY_ADDRESS}" \
    "${ENV_ARGS[@]}" \
    -- "${CMD_ARGS[@]}"
