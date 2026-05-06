#!/bin/bash
#
# Phase 1: Generate patches for SWE-bench instances.
#
# Supports two strategies:
#   - naive: Single-shot vLLM inference from pre-built prompts.
#   - agent: Agentic loop using mini-swe-agent (or any agent via YAML config)
#            running inside K8s Jobs with SWE-bench container images.
#
# Prerequisites:
#   - RayCluster deployed
#   - vLLM server deployed and accessible from the cluster
#   - Port-forward active: oc port-forward svc/<ray-head-svc> 8265:8265
#   - For naive: prompted dataset built
#   - For agent: SWE-bench images available (DockerHub or internal registry)
#
# Usage:
#   # Naive strategy (default)
#   bash run_swe_bench_phase1.sh
#
#   # Agent strategy
#   STRATEGY=agent bash run_swe_bench_phase1.sh
#
#   # Quick test with 2 instances
#   STRATEGY=agent INSTANCE_LIMIT=2 bash run_swe_bench_phase1.sh

set -euo pipefail

# ── Common config ───────────────────────────────────────────────
STRATEGY="${STRATEGY:-naive}"
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
VLLM_URL="${VLLM_URL:-http://vllm-server:8000/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
DATASET="${DATASET:-SWE-bench/SWE-bench_Verified}"
SPLIT="${SPLIT:-test}"
NUM_WORKERS="${NUM_WORKERS:-2}"
INSTANCE_LIMIT="${INSTANCE_LIMIT:-0}"
RUN_ID="${RUN_ID:-eval-run}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/swe-bench-results/${RUN_ID}}"
S3_BUCKET="${S3_BUCKET:-swe-bench}"
S3_OUTPUT="${S3_OUTPUT:-s3://${S3_BUCKET}/runs/${RUN_ID}/predictions.jsonl}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-}"

# ── Naive strategy config ──────────────────────────────────────
PROMPTS="${PROMPTS:-s3://${S3_BUCKET}/verified/prompts/style-3-oracle.jsonl}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
TEMPERATURE="${TEMPERATURE:-0.15}"

# ── Agent strategy config ──────────────────────────────────────
AGENT_CONFIG="${AGENT_CONFIG:-evals/swe_bench/agents/mini_swe_agent.yaml}"
MODEL_API_KEY="${MODEL_API_KEY:-dummy}"
STEP_LIMIT="${STEP_LIMIT:-150}"
COST_LIMIT="${COST_LIMIT:-3.0}"
K8S_NAMESPACE="${K8S_NAMESPACE:-}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-swe-bench-eval}"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-}"
SWEBENCH_NAMESPACE="${SWEBENCH_NAMESPACE:-swebench}"
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-4}"
JOB_TIMEOUT="${JOB_TIMEOUT:-600}"
RUN_EVAL="${RUN_EVAL:-1}"

if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

# ── Build command ───────────────────────────────────────────────
CMD_ARGS=(
    python3 -m evals.swe_bench.run_patch_generation
    --strategy "${STRATEGY}"
    --vllm-url "${VLLM_URL}"
    --model-name "${MODEL_NAME}"
    --dataset "${DATASET}"
    --split "${SPLIT}"
    --num-workers "${NUM_WORKERS}"
    --output-dir "${OUTPUT_DIR}"
    --instance-limit "${INSTANCE_LIMIT}"
    --s3-output "${S3_OUTPUT}"
    --run-id "${RUN_ID}"
)

if [[ "${STRATEGY}" == "naive" ]]; then
    CMD_ARGS+=(
        --prompts "${PROMPTS}"
        --max-tokens "${MAX_TOKENS}"
        --temperature "${TEMPERATURE}"
    )
elif [[ "${STRATEGY}" == "agent" ]]; then
    CMD_ARGS+=(
        --agent-config "${AGENT_CONFIG}"
        --model-api-key "${MODEL_API_KEY}"
        --step-limit "${STEP_LIMIT}"
        --cost-limit "${COST_LIMIT}"
        --service-account "${SERVICE_ACCOUNT}"
        --swebench-namespace "${SWEBENCH_NAMESPACE}"
        --max-concurrent-jobs "${MAX_CONCURRENT_JOBS}"
        --job-timeout "${JOB_TIMEOUT}"
    )
    if [[ -n "${K8S_NAMESPACE}" ]]; then
        CMD_ARGS+=(--k8s-namespace "${K8S_NAMESPACE}")
    fi
    if [[ -n "${IMAGE_REGISTRY}" ]]; then
        CMD_ARGS+=(--image-registry "${IMAGE_REGISTRY}")
    fi
    if [[ "${RUN_EVAL}" == "0" ]]; then
        CMD_ARGS+=(--skip-eval)
    fi
fi

# Pass MLflow tracking URI via Ray runtime env so workers pick it up
ENV_ARGS=()
if [[ -n "${MLFLOW_TRACKING_URI}" ]]; then
    ENV_ARGS+=(--runtime-env-json "{\"env_vars\": {\"MLFLOW_TRACKING_URI\": \"${MLFLOW_TRACKING_URI}\"}}")
fi

echo "Strategy: ${STRATEGY}"
echo "Running: ray job submit -- ${CMD_ARGS[*]}"

ray job submit \
    --address="${RAY_ADDRESS}" \
    ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} \
    -- "${CMD_ARGS[@]}"
