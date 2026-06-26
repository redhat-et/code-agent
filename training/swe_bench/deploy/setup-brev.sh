#!/bin/bash
#
# End-to-end setup for SWE-bench RL training on a Brev GPU instance.
#
# Performs ALL steps needed to go from a fresh Brev instance to a running
# Ray cluster ready for training.  Designed to be idempotent — safe to
# re-run after partial failures.
#
# Prerequisites:
#   - Brev instance with microk8s, docker, helm, kubectl
#   - Storage relocated:  sudo bash training/swe_bench/deploy/setup-brev-storage.sh
#   - This repo cloned at ~/code-agent
#
# Usage:
#   cd ~/code-agent
#   bash training/swe_bench/deploy/setup-brev.sh              # full setup
#   SKIP_IMAGE_BUILD=1 bash training/swe_bench/deploy/setup-brev.sh  # skip Docker build
#   TRAINING_GPUS=8 TRAINING_MEMORY=180Gi bash training/swe_bench/deploy/setup-brev.sh
#
# After setup, start training:
#   kubectl port-forward svc/swe-rl-cluster-head-svc 8265:8265 &
#   RAY_IMAGE=localhost:5000/ray-swe-rl:latest \
#     TRAINING_GPUS=<N> TRAINING_MEMORY=<M> \
#     bash train_swe_bench_grpo.sh

set -euo pipefail
cd "$(dirname "$0")/../../.."
REPO_ROOT="$(pwd)"

# ═══════════════════════════════════════════════════════════════
# Configuration — override via environment variables
# ═══════════════════════════════════════════════════════════════

TRAINING_GPUS="${TRAINING_GPUS:-1}"
TRAINING_GPUS_PER_NODE="${TRAINING_GPUS_PER_NODE:-${TRAINING_GPUS}}"
TRAINING_MEMORY="${TRAINING_MEMORY:-64Gi}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
INSTANCE_LIMIT="${INSTANCE_LIMIT:-0}"
SKIP_IMAGE_BUILD="${SKIP_IMAGE_BUILD:-}"

REGISTRY="localhost:5000"
RAY_IMAGE="${REGISTRY}/ray-swe-rl:latest"

# ═══════════════════════════════════════════════════════════════
# Step 0: Verify prerequisites
# ═══════════════════════════════════════════════════════════════

echo "=== 0/10  Verifying prerequisites ==="
for cmd in docker kubectl helm git; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: $cmd not found" >&2
        exit 1
    fi
done

if [ ! -d /ephemeral ]; then
    echo "ERROR: /ephemeral not found — is this a Brev GPU instance?" >&2
    echo "       See https://docs.nvidia.com/brev/concepts/gpu-instances" >&2
    exit 1
fi

if [ ! -L /var/lib/docker ] && [ -d /var/lib/docker ]; then
    echo "WARNING: /var/lib/docker is not a symlink to /ephemeral."
    echo "         Run:  sudo bash training/swe_bench/deploy/setup-brev-storage.sh"
    echo "         Then re-run this script."
    exit 1
fi

echo "  Prerequisites OK."

# ═══════════════════════════════════════════════════════════════
# Step 1: Enable microk8s addons
# ═══════════════════════════════════════════════════════════════

echo "=== 1/10  Enabling microk8s addons ==="
# microk8s enable is idempotent — safe to re-run.
for addon in gpu dns helm3 hostpath-storage; do
    echo "  Ensuring $addon is enabled..."
    sudo microk8s enable "$addon" 2>&1 | tail -1
done

# ═══════════════════════════════════════════════════════════════
# Step 2: Local Docker registry on /ephemeral
# ═══════════════════════════════════════════════════════════════

echo "=== 2/10  Setting up local Docker registry ==="
if docker ps --format '{{.Names}}' | grep -q '^registry$'; then
    echo "  Registry container already running."
else
    echo "  Starting registry on ${REGISTRY} (data: /ephemeral/registry)..."
    docker run -d \
        --restart=always \
        -p 5000:5000 \
        -v /ephemeral/registry:/var/lib/registry \
        --name registry \
        registry:2
fi

# ═══════════════════════════════════════════════════════════════
# Step 3: Install KubeRay operator
# ═══════════════════════════════════════════════════════════════

echo "=== 3/10  Installing KubeRay operator ==="
helm repo add kuberay https://ray-project.github.io/kuberay-helm/ 2>/dev/null || true
helm repo update kuberay
if helm status kuberay-operator >/dev/null 2>&1; then
    echo "  KubeRay operator already installed."
else
    helm install kuberay-operator kuberay/kuberay-operator --version 1.3.0
fi

# ═══════════════════════════════════════════════════════════════
# Step 4: Apply RBAC
# ═══════════════════════════════════════════════════════════════

echo "=== 4/10  Applying RBAC ==="
kubectl apply -f training/swe_bench/deploy/rbac-training.yaml

# ═══════════════════════════════════════════════════════════════
# Step 5: Create PVCs
# ═══════════════════════════════════════════════════════════════

echo "=== 5/10  Creating PVCs ==="
kubectl apply -f training/swe_bench/deploy/pvcs.yaml

# ═══════════════════════════════════════════════════════════════
# Step 6: Build and push Docker images
# ═══════════════════════════════════════════════════════════════

echo "=== 6/10  Building Docker images ==="
if [ -n "${SKIP_IMAGE_BUILD}" ]; then
    echo "  SKIP_IMAGE_BUILD set, skipping."
else
    echo "  Building ray-swe-rl-base:latest (~10 min first time, cached after)..."
    docker build \
        -f infra/images/Containerfile.ray-swe-rl-base \
        -t ray-swe-rl-base:latest \
        .

    echo "  Building ray-swe-rl:latest (~1 min)..."
    docker build \
        -f infra/images/Containerfile.ray-swe-rl \
        -t ray-swe-rl:latest \
        .

    echo "  Tagging and pushing to ${REGISTRY}..."
    docker tag ray-swe-rl-base:latest "${REGISTRY}/ray-swe-rl-base:latest"
    docker push "${REGISTRY}/ray-swe-rl-base:latest"
    docker tag ray-swe-rl:latest "${REGISTRY}/ray-swe-rl:latest"
    docker push "${REGISTRY}/ray-swe-rl:latest"

    echo "  Cleaning up Docker build cache and dangling images..."
    docker builder prune -f 2>/dev/null || true
    docker image prune -f 2>/dev/null || true
fi

# ═══════════════════════════════════════════════════════════════
# Step 7: Install Ray CLI on host
# ═══════════════════════════════════════════════════════════════

echo "=== 7/10  Installing Ray CLI ==="
if command -v ray >/dev/null 2>&1 || [ -x "$HOME/.local/bin/ray" ]; then
    echo "  Ray CLI already installed."
else
    echo "  Installing ray[default]..."
    pip3 install --user 'ray[default]'
    echo "  NOTE: Add to your shell profile:"
    echo "    export PATH=\$HOME/.local/bin:\$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"

# ═══════════════════════════════════════════════════════════════
# Step 8: Deploy Ray cluster
# ═══════════════════════════════════════════════════════════════

echo "=== 8/10  Deploying Ray cluster ==="
RAY_IMAGE="${RAY_IMAGE}" \
    TRAINING_GPUS="${TRAINING_GPUS}" \
    TRAINING_GPUS_PER_NODE="${TRAINING_GPUS_PER_NODE}" \
    TRAINING_MEMORY="${TRAINING_MEMORY}" \
    bash train_swe_bench_grpo.sh --generate-cluster | kubectl apply -f -

echo "  Waiting for pods to be ready..."
kubectl wait --for=condition=Ready pod -l ray.io/cluster=swe-rl-cluster --timeout=300s

# ═══════════════════════════════════════════════════════════════
# Step 9: Build training dataset (writes directly to PVC)
# ═══════════════════════════════════════════════════════════════

echo "=== 9/10  Building training dataset ==="
DATASET_PATH="/hf-cache/swe_bench_train.jsonl"

# Check if dataset already exists on the PVC via the worker pod.
WORKER_POD=$(kubectl get pods -l ray.io/cluster=swe-rl-cluster,ray.io/node-type=worker -o jsonpath='{.items[0].metadata.name}')
if kubectl exec "$WORKER_POD" -- test -f "${DATASET_PATH}" 2>/dev/null; then
    echo "  Dataset already exists at ${DATASET_PATH}, skipping."
else
    BUILD_CMD="python3 /opt/code-agent/training/swe_bench/build_dataset.py --output ${DATASET_PATH} --model ${MODEL_PATH}"
    if [ "${INSTANCE_LIMIT}" -gt 0 ] 2>/dev/null; then
        BUILD_CMD="${BUILD_CMD} --instance-limit ${INSTANCE_LIMIT}"
    fi

    echo "  Running build_dataset.py via K8s Job (model: ${MODEL_PATH})..."
    cat <<JOB_YAML | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: build-swe-dataset
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: builder
          image: ${RAY_IMAGE}
          imagePullPolicy: Always
          command: ["/bin/bash", "-c", "${BUILD_CMD}"]
          volumeMounts:
            - name: hf-cache
              mountPath: /hf-cache
          env:
            - name: HF_HOME
              value: /hf-cache
      volumes:
        - name: hf-cache
          persistentVolumeClaim:
            claimName: hf-cache-0
JOB_YAML

    echo "  Waiting for dataset Job to complete..."
    kubectl wait --for=condition=complete job/build-swe-dataset --timeout=600s
    kubectl delete job build-swe-dataset
    echo "  Dataset built at ${DATASET_PATH}."
fi

# ═══════════════════════════════════════════════════════════════
# Step 10: Summary and next steps
# ═══════════════════════════════════════════════════════════════

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Ray cluster pods:"
kubectl get pods -l ray.io/cluster=swe-rl-cluster -o wide
echo ""
echo "Next steps:"
echo ""
echo "  1. Port-forward Ray dashboard:"
echo "     kubectl port-forward svc/swe-rl-cluster-head-svc 8265:8265 &"
echo ""
echo "  2. Run training:"
echo "     RAY_IMAGE=${RAY_IMAGE} \\"
echo "       TRAINING_GPUS=${TRAINING_GPUS} \\"
echo "       TRAINING_MEMORY=${TRAINING_MEMORY} \\"
echo "       DATASET=${DATASET_PATH} \\"
echo "       bash train_swe_bench_grpo.sh"
echo ""
echo "  For a quick smoke test (2 instances, small model):"
echo "     RAY_IMAGE=${RAY_IMAGE} \\"
echo "       TRAINING_GPUS=${TRAINING_GPUS} \\"
echo "       TRAINING_MEMORY=${TRAINING_MEMORY} \\"
echo "       MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct \\"
echo "       DATASET=${DATASET_PATH} \\"
echo "       bash train_swe_bench_grpo.sh"
