#!/bin/bash
#
# Evaluate a training checkpoint on SWE-bench.
#
# Takes a checkpoint path (local or S3), deploys a vLLM server for it,
# waits for it to be ready, runs the SWE-bench eval pipeline, and
# collects results.
#
# The checkpoint can be a HuggingFace model ID (for baseline eval) or
# a local/S3 path to a training checkpoint directory.
#
# Prerequisites:
#   - Eval RayCluster deployed: oc apply -f evals/swe_bench/deploy/raycluster-patch-gen.yaml
#   - RBAC applied: oc apply -f evals/swe_bench/deploy/rbac.yaml
#   - Port-forward active for Ray: oc port-forward svc/swe-bench-patch-gen-head-svc 8265:8265
#   - Prompted dataset built (for naive strategy)
#
# Usage:
#   # Evaluate a training checkpoint
#   bash eval_checkpoint.sh /tmp/checkpoints/qwen3.5-9b-grpo-swe/step-100
#
#   # Evaluate baseline (pre-training) model
#   bash eval_checkpoint.sh Qwen/Qwen3.5-9B
#
#   # With custom settings
#   INSTANCE_LIMIT=16 STRATEGY=agent \
#     bash eval_checkpoint.sh /tmp/checkpoints/step-200
#
# Environment variables (all optional, sensible defaults):
#   STRATEGY          naive or agent (default: naive)
#   INSTANCE_LIMIT    limit instances for quick checks (default: 0 = all)
#   VLLM_TP           vLLM tensor parallel size (default: 1)
#   VLLM_GPU_MEM      vLLM GPU memory utilization (default: 0.90)
#   VLLM_DEPLOYMENT   K8s deployment name (default: vllm-ckpt-eval)
#   VLLM_PORT         local port for port-forward (default: 8800)
#   CLEANUP           delete vLLM deployment after eval (default: 1)

set -euo pipefail

# ── Arguments ──────────────────────────────────────────────────
CHECKPOINT="${1:?Usage: bash eval_checkpoint.sh <checkpoint_path_or_model_id>}"

# ── Configuration ──────────────────────────────────────────────
STRATEGY="${STRATEGY:-naive}"
INSTANCE_LIMIT="${INSTANCE_LIMIT:-0}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.90}"
VLLM_DEPLOYMENT="${VLLM_DEPLOYMENT:-vllm-ckpt-eval}"
VLLM_PORT="${VLLM_PORT:-8800}"
CLEANUP="${CLEANUP:-1}"

PF_PID=""
cleanup() {
    [ -n "${PF_PID}" ] && kill "${PF_PID}" 2>/dev/null || true
    if [[ "${CLEANUP}" == "1" ]]; then
        kubectl delete deployment/${VLLM_DEPLOYMENT} svc/${VLLM_DEPLOYMENT} --ignore-not-found 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Derive a run ID from the checkpoint path
CKPT_BASENAME=$(basename "${CHECKPOINT}")
RUN_ID="${RUN_ID:-eval-${CKPT_BASENAME}-$(date +%s)}"

echo "=== Checkpoint Evaluation ==="
echo "  Checkpoint:  ${CHECKPOINT}"
echo "  Strategy:    ${STRATEGY}"
echo "  Run ID:      ${RUN_ID}"
echo ""

# ── Step 1: Deploy vLLM for the checkpoint ─────────────────────

echo ">>> Deploying vLLM server for checkpoint..."

kubectl apply -f - <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${VLLM_DEPLOYMENT}
  labels:
    app: ${VLLM_DEPLOYMENT}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${VLLM_DEPLOYMENT}
  template:
    metadata:
      labels:
        app: ${VLLM_DEPLOYMENT}
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          env:
            - name: HF_HOME
              value: "/cache/huggingface"
            - name: XDG_CACHE_HOME
              value: "/cache"
            - name: HOME
              value: "/cache"
            - name: USER
              value: "vllm"
          command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
          args:
            - "--model=${CHECKPOINT}"
            - "--tensor-parallel-size=${VLLM_TP}"
            - "--host=0.0.0.0"
            - "--port=8000"
            - "--gpu-memory-utilization=${VLLM_GPU_MEM}"
            - "--max-model-len=16384"
            - "--enable-auto-tool-choice"
            - "--tool-call-parser=qwen3_coder"
            - "--reasoning-parser=qwen3"
          ports:
            - name: http
              containerPort: 8000
          readinessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 60
            periodSeconds: 10
          resources:
            requests:
              cpu: "4"
              memory: "32Gi"
              nvidia.com/gpu: "${VLLM_TP}"
            limits:
              cpu: "4"
              memory: "64Gi"
              nvidia.com/gpu: "${VLLM_TP}"
          volumeMounts:
            - name: cache
              mountPath: /cache
            - name: shm
              mountPath: /dev/shm
      volumes:
        - name: cache
          persistentVolumeClaim:
            claimName: vllm-cache
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: "8Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: ${VLLM_DEPLOYMENT}
spec:
  selector:
    app: ${VLLM_DEPLOYMENT}
  ports:
    - port: 8000
      targetPort: 8000
YAML

# ── Step 2: Wait for vLLM to be ready ─────────────────────────

echo ">>> Waiting for vLLM to be ready..."
kubectl rollout status deployment/${VLLM_DEPLOYMENT} --timeout=600s

# Port-forward in background
kubectl port-forward svc/${VLLM_DEPLOYMENT} ${VLLM_PORT}:8000 &
PF_PID=$!
sleep 3

# Verify the model is serving
for i in $(seq 1 30); do
    if curl -s http://localhost:${VLLM_PORT}/v1/models | grep -q "id"; then
        echo ">>> vLLM is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: vLLM not ready after 5 minutes"
        exit 1
    fi
    sleep 10
done

# ── Step 3: Run evaluation ─────────────────────────────────────

echo ">>> Running SWE-bench evaluation..."
VLLM_URL="http://localhost:${VLLM_PORT}/v1" \
MODEL_NAME="${CHECKPOINT}" \
RUN_ID="${RUN_ID}" \
STRATEGY="${STRATEGY}" \
INSTANCE_LIMIT="${INSTANCE_LIMIT}" \
    bash run_swe_bench_eval.sh

EVAL_EXIT=$?

# ── Step 4: Report ────────────────────────────────────────────
# (Cleanup runs automatically via the EXIT trap.)

if [[ $EVAL_EXIT -eq 0 ]]; then
    echo ""
    echo "=== Evaluation complete ==="
    echo "  Run ID: ${RUN_ID}"
    echo "  Results: /tmp/swe-bench-results/${RUN_ID}/results.json"
else
    echo "ERROR: Evaluation failed with exit code ${EVAL_EXIT}"
    exit $EVAL_EXIT
fi
