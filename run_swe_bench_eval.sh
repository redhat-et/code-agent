#!/bin/bash
#
# SWE-bench evaluation: patch generation and test execution.
#
# Supports two strategies, both of which support single-shot and multi-turn:
#   - naive:  vLLM inference from pre-built prompts, followed by K8s test
#             execution. Set MAX_TURNS > 1 for multi-turn with feedback.
#   - agent:  Agentic loop (mini-swe-agent or custom) inside K8s Jobs.
#             Set MAX_TURNS > 1 for multi-turn with feedback between runs.
#
# All strategies produce results.json with resolve rate and per-instance details.
#
# Prerequisites:
#   - RayCluster deployed (see evals/swe_bench/deploy/)
#   - RBAC applied: oc apply -f evals/swe_bench/deploy/rbac.yaml
#   - vLLM server deployed and accessible from the cluster
#   - Port-forward active: oc port-forward svc/<ray-head-svc> 8265:8265
#   - For naive strategy: prompted dataset built (see job-build-prompts.yaml)
#   - SWE-bench images available (DockerHub or internal registry)
#
# Usage:
#   # Naive strategy, single-shot (default)
#   bash run_swe_bench_eval.sh
#
#   # Naive strategy, multi-turn (3 attempts, AST check between turns)
#   MAX_TURNS=3 INTERMEDIATE_VERIFIERS="ast_check" bash run_swe_bench_eval.sh
#
#   # Agent strategy, single-shot
#   STRATEGY=agent bash run_swe_bench_eval.sh
#
#   # Agent strategy, multi-turn
#   STRATEGY=agent MAX_TURNS=3 INTERMEDIATE_VERIFIERS="ast_check" \
#     bash run_swe_bench_eval.sh
#
#   # Quick test with 2 instances
#   INSTANCE_LIMIT=2 bash run_swe_bench_eval.sh

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
S3_OUTPUT="${S3_OUTPUT:-s3://${S3_BUCKET}/runs/${RUN_ID}/results.json}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-}"

# ── Multi-turn config (applies to both strategies) ─────────────
MAX_TURNS="${MAX_TURNS:-1}"
INTERMEDIATE_VERIFIERS="${INTERMEDIATE_VERIFIERS:-}"  # space-separated, e.g. "ast_check"
AGGREGATOR="${AGGREGATOR:-mean}"

# ── Naive strategy config ──────────────────────────────────────
PROMPTS="${PROMPTS:-s3://${S3_BUCKET}/verified/prompts/style-3-oracle.jsonl}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
TEMPERATURE="${TEMPERATURE:-0.15}"

# ── Agent strategy config ──────────────────────────────────────
AGENT_CONFIG="${AGENT_CONFIG:-evals/swe_bench/agents/mini_swe_agent.yaml}"
MODEL_API_KEY="${MODEL_API_KEY:-dummy}"
STEP_LIMIT="${STEP_LIMIT:-150}"
COST_LIMIT="${COST_LIMIT:-3.0}"
RUN_EVAL="${RUN_EVAL:-1}"

# ── Shared K8s / registry config ───────────────────────────────
K8S_NAMESPACE="${K8S_NAMESPACE:-}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-swe-bench-eval}"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-}"
SWEBENCH_NAMESPACE="${SWEBENCH_NAMESPACE:-swebench}"
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-4}"
JOB_TIMEOUT="${JOB_TIMEOUT:-1800}"

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
    --max-turns "${MAX_TURNS}"
    --aggregator "${AGGREGATOR}"
)

# Append each intermediate verifier as a separate arg
if [[ -n "${INTERMEDIATE_VERIFIERS}" ]]; then
    for v in ${INTERMEDIATE_VERIFIERS}; do
        CMD_ARGS+=(--intermediate-verifiers "${v}")
    done
fi

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
    if [[ "${RUN_EVAL}" == "0" ]]; then
        CMD_ARGS+=(--skip-eval)
    fi
fi

# K8s and registry args (needed for agent and multi-turn naive)
if [[ "${STRATEGY}" == "agent" ]] || [[ "${MAX_TURNS}" -gt 1 ]]; then
    if [[ -n "${K8S_NAMESPACE}" ]]; then
        CMD_ARGS+=(--k8s-namespace "${K8S_NAMESPACE}")
    fi
    if [[ -n "${IMAGE_REGISTRY}" ]]; then
        CMD_ARGS+=(--image-registry "${IMAGE_REGISTRY}")
    fi
    if [[ "${STRATEGY}" == "naive" ]]; then
        # naive multi-turn also needs K8s args for SWEBenchUnitTestVerifier
        CMD_ARGS+=(
            --service-account "${SERVICE_ACCOUNT}"
            --swebench-namespace "${SWEBENCH_NAMESPACE}"
            --max-concurrent-jobs "${MAX_CONCURRENT_JOBS}"
            --job-timeout "${JOB_TIMEOUT}"
        )
    fi
fi

# Pass MLflow tracking URI via Ray runtime env so workers pick it up
ENV_ARGS=()
if [[ -n "${MLFLOW_TRACKING_URI}" ]]; then
    ENV_ARGS+=(--runtime-env-json "{\"env_vars\": {\"MLFLOW_TRACKING_URI\": \"${MLFLOW_TRACKING_URI}\"}}")
fi

echo "Strategy: ${STRATEGY}, max_turns: ${MAX_TURNS}"
echo "Running: ray job submit -- ${CMD_ARGS[*]}"

ray job submit \
    --address="${RAY_ADDRESS}" \
    ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} \
    -- "${CMD_ARGS[@]}"
