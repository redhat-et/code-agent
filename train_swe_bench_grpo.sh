#!/bin/bash
#
# GRPO training of Qwen3.5-9B on SWE-bench using OpenRLHF with multi-turn
# agent rollouts.
#
# The agent (SWEBenchAgentInstance) runs inside the vLLM rollout worker.
# Each step calls kubectl exec into a K8s Pod sandbox.
#
# Prerequisites:
#   - KubeRay operator installed (OperatorHub or Helm)
#   - RBAC applied:     oc apply -f training/swe_bench/deploy/rbac-training.yaml
#   - Cluster created:  bash train_swe_bench_grpo.sh --generate-cluster | oc apply -f -
#   - Port-forward:     oc port-forward svc/swe-rl-cluster-head-svc 8265:8265
#   - Dataset built:    python training/swe_bench/build_dataset.py \
#                         --output /tmp/swe_bench_train.jsonl
#   - Project installed in Ray image: pip install -e ".[training,eval]"
#
# Usage:
#   bash train_swe_bench_grpo.sh                     # run training
#   bash train_swe_bench_grpo.sh --generate-cluster  # print matching Ray cluster YAML
#
# ═══════════════════════════════════════════════════════════════
# HARDWARE CONFIGURATION — edit this section to change the setup.
# Everything below (OpenRLHF flags, Ray cluster YAML) derives from
# these values.  See HARDWARE_REQUIREMENTS.md for sizing guidance.
# ═══════════════════════════════════════════════════════════════

# Deployment mode: "colocated" or "separate"
#   colocated:  all models share GPUs; rollout and training take turns
#               via sleep/wake memory offload.  Minimum GPU count.
#   separate:   dedicated GPU sets for rollout and training; can run
#               in parallel (async pipelining).  Needs more GPUs.
DEPLOY_MODE="${DEPLOY_MODE:-colocated}"

# Training worker: total GPUs and how they are distributed across nodes.
#   TRAINING_GPUS:          total GPU count for DeepSpeed
#   TRAINING_GPUS_PER_NODE: GPUs per physical node (must divide TRAINING_GPUS evenly)
#   TRAINING_NODES:         derived automatically
TRAINING_GPUS="${TRAINING_GPUS:-4}"
TRAINING_GPUS_PER_NODE="${TRAINING_GPUS_PER_NODE:-${TRAINING_GPUS}}"

# Rollout worker (only used in "separate" mode).
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"

# vLLM tensor-parallel size (defaults to rollout GPU count).
# Must not exceed the GPUs on a single node (TP doesn't cross nodes efficiently).
VLLM_TP="${VLLM_TP:-}"

# DeepSpeed ZeRO stage.
#   3: model weights sharded (works with fewer GPUs, slower)
#   2: model replicated, optimizer/gradients sharded (faster, needs more GPUs)
ZERO_STAGE="${ZERO_STAGE:-3}"

# Worker pod memory requests.
TRAINING_MEMORY="${TRAINING_MEMORY:-180Gi}"
ROLLOUT_MEMORY="${ROLLOUT_MEMORY:-32Gi}"

# Shared-memory size for NCCL allreduce.
SHM_SIZE="${SHM_SIZE:-16Gi}"

# Container image for Ray workers.
RAY_IMAGE="${RAY_IMAGE:-image-registry.openshift-image-registry.svc:5000/code-agent/ray-swe-rl:latest}"

# Node selector (optional).  Set to target specific instance types.
# Example: NODE_SELECTOR="node.kubernetes.io/instance-type=g6e.12xlarge"
NODE_SELECTOR="${NODE_SELECTOR:-}"

# ═══════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════

RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
DATASET="${DATASET:-/tmp/swe_bench_train.jsonl}"
SAVE_PATH="${SAVE_PATH:-/checkpoints/qwen3.5-9b-grpo-swe}"

# Agent environment (forwarded to Ray workers via env vars)
SWE_MAX_STEPS="${SWE_MAX_STEPS:-100}"
SWE_IMAGE_REGISTRY="${SWE_IMAGE_REGISTRY:-}"
SWE_K8S_NAMESPACE="${SWE_K8S_NAMESPACE:-}"
SWE_SERVICE_ACCOUNT="${SWE_SERVICE_ACCOUNT:-swe-bench-training}"
SWE_MAX_ROLLOUT_RETRIES="${SWE_MAX_ROLLOUT_RETRIES:-2}"

# ═══════════════════════════════════════════════════════════════
# DERIVED SETTINGS — no need to edit below this line
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

TRAINING_NODES=$(( TRAINING_GPUS / TRAINING_GPUS_PER_NODE ))
if (( TRAINING_GPUS % TRAINING_GPUS_PER_NODE != 0 )); then
    echo "ERROR: TRAINING_GPUS (${TRAINING_GPUS}) must be divisible by TRAINING_GPUS_PER_NODE (${TRAINING_GPUS_PER_NODE})"
    exit 1
fi

if [ "${DEPLOY_MODE}" = "colocated" ]; then
    EFFECTIVE_ROLLOUT_GPUS="${TRAINING_GPUS_PER_NODE}"
    VLLM_ENGINES="${TRAINING_NODES}"
    COLOCATION_FLAGS="--train.colocate_all --vllm.enable_sleep --ds.enable_sleep"
else
    EFFECTIVE_ROLLOUT_GPUS="${ROLLOUT_GPUS}"
    VLLM_ENGINES=1
    COLOCATION_FLAGS="--train.colocate_actor_ref"
fi

VLLM_TP="${VLLM_TP:-${EFFECTIVE_ROLLOUT_GPUS}}"

# ── Helper: emit nodeSelector YAML if configured ──────────────

_node_selector_yaml() {
    local indent="${1:-10}"
    if [ -z "${NODE_SELECTOR}" ]; then
        return
    fi
    local key="${NODE_SELECTOR%%=*}"
    local val="${NODE_SELECTOR#*=}"
    printf '%*s%s\n' "$indent" "" "nodeSelector:"
    printf '%*s  %s: \"%s\"\n' "$indent" "" "$key" "$val"
}

# ── Generate Ray cluster YAML ──────────────────────────────────

generate_cluster_yaml() {
    local sa="${SWE_SERVICE_ACCOUNT}"
    local ns_yaml
    ns_yaml=$(_node_selector_yaml 10)

    cat <<YAML
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: kube-rbac-proxy-config-swe-rl-cluster
data:
  config-file.yaml: |+
    authorization:
      resourceAttributes:
        apiVersion: v1
        resource: services
        subresource: proxy
        name: swe-rl-cluster
---
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: swe-rl-cluster
spec:
  rayVersion: "2.53.0"
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
      num-cpus: "0"
    template:
      spec:
        serviceAccountName: ${sa}
        containers:
          - name: ray-head
            image: ${RAY_IMAGE}
            imagePullPolicy: Always
            ports:
              - containerPort: 6379
                name: gcs
              - containerPort: 8265
                name: dashboard
              - containerPort: 10001
                name: client
            resources:
              requests:
                cpu: "4"
                memory: "8Gi"
              limits:
                cpu: "4"
                memory: "8Gi"
            env:
              - name: NVIDIA_VISIBLE_DEVICES
                value: "void"
            volumeMounts:
              - name: ray-tmp
                mountPath: /tmp
        volumes:
          - name: ray-tmp
            emptyDir: {}
  workerGroupSpecs:
YAML

    if [ "${DEPLOY_MODE}" = "colocated" ]; then
        # Generate one worker group per node, each with its own PVC for HF cache.
        # EBS volumes are ReadWriteOnce, so each node needs a separate PVC.
        # Worker-0 also mounts the checkpoints PVC (DeepSpeed saves from rank 0).
        for idx in $(seq 0 $(( TRAINING_NODES - 1 ))); do
            local ckpt_mount=""
            local ckpt_vol=""
            if [ "$idx" -eq 0 ]; then
                ckpt_mount="                - name: checkpoints
                  mountPath: /checkpoints"
                ckpt_vol="            - name: checkpoints
              persistentVolumeClaim:
                claimName: training-checkpoints"
            fi
            cat <<YAML
    # Worker node ${idx} (${TRAINING_GPUS_PER_NODE} GPUs, colocated)
    - groupName: worker-${idx}
      replicas: 1
      minReplicas: 1
      maxReplicas: 1
      rayStartParams:
        num-gpus: "${TRAINING_GPUS_PER_NODE}"
      template:
        spec:
          serviceAccountName: ${sa}
          securityContext:
            runAsUser: 0
${ns_yaml}
          containers:
            - name: ray-worker
              image: ${RAY_IMAGE}
              imagePullPolicy: Always
              env:
                - name: HOME
                  value: /tmp
                - name: HF_HOME
                  value: /hf-cache
                - name: XDG_CACHE_HOME
                  value: /tmp/.cache
                - name: TRITON_CACHE_DIR
                  value: /tmp/.triton
                - name: NVTX_DISABLE
                  value: "1"
                - name: PYTORCH_CUDA_ALLOC_CONF
                  value: "expandable_segments:True"
              resources:
                requests:
                  cpu: "16"
                  memory: "${TRAINING_MEMORY}"
                  nvidia.com/gpu: "${TRAINING_GPUS_PER_NODE}"
                limits:
                  cpu: "16"
                  memory: "${TRAINING_MEMORY}"
                  nvidia.com/gpu: "${TRAINING_GPUS_PER_NODE}"
              volumeMounts:
                - name: ray-tmp
                  mountPath: /tmp
                - name: shm
                  mountPath: /dev/shm
                - name: hf-cache
                  mountPath: /hf-cache
${ckpt_mount}
          volumes:
            - name: ray-tmp
              emptyDir: {}
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "${SHM_SIZE}"
            - name: hf-cache
              persistentVolumeClaim:
                claimName: hf-cache-${idx}
${ckpt_vol}
YAML
        done
    else
        cat <<YAML
    # Rollout worker: vLLM engine + agent loop (${ROLLOUT_GPUS} GPUs, TP=${VLLM_TP})
    # Needs K8s API access for Pod management.
    - groupName: rollout
      replicas: 1
      minReplicas: 1
      maxReplicas: 1
      rayStartParams:
        num-gpus: "${ROLLOUT_GPUS}"
      template:
        spec:
          serviceAccountName: ${sa}
${ns_yaml}
          containers:
            - name: ray-worker
              image: ${RAY_IMAGE}
              imagePullPolicy: Always
              env:
                - name: HOME
                  value: /tmp
                - name: HF_HOME
                  value: /hf-cache
                - name: XDG_CACHE_HOME
                  value: /tmp/.cache
                - name: TRITON_CACHE_DIR
                  value: /tmp/.triton
                - name: NVTX_DISABLE
                  value: "1"
                - name: PYTORCH_CUDA_ALLOC_CONF
                  value: "expandable_segments:True"
              resources:
                requests:
                  cpu: "8"
                  memory: "${ROLLOUT_MEMORY}"
                  nvidia.com/gpu: "${ROLLOUT_GPUS}"
                limits:
                  cpu: "8"
                  memory: "${ROLLOUT_MEMORY}"
                  nvidia.com/gpu: "${ROLLOUT_GPUS}"
              volumeMounts:
                - name: ray-tmp
                  mountPath: /tmp
                - name: hf-cache
                  mountPath: /hf-cache
                - name: shm
                  mountPath: /dev/shm
          volumes:
            - name: ray-tmp
              emptyDir: {}
            - name: hf-cache
              emptyDir: {}
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "${SHM_SIZE}"
    # Training worker: DeepSpeed ZeRO-${ZERO_STAGE} (${TRAINING_GPUS_PER_NODE} GPUs/node × ${TRAINING_NODES} node(s))
    - groupName: training
      replicas: ${TRAINING_NODES}
      minReplicas: ${TRAINING_NODES}
      maxReplicas: ${TRAINING_NODES}
      rayStartParams:
        num-gpus: "${TRAINING_GPUS_PER_NODE}"
      template:
        spec:
${ns_yaml}
          containers:
            - name: ray-worker
              image: ${RAY_IMAGE}
              imagePullPolicy: Always
              env:
                - name: HOME
                  value: /tmp
                - name: HF_HOME
                  value: /hf-cache
                - name: XDG_CACHE_HOME
                  value: /tmp/.cache
                - name: TRITON_CACHE_DIR
                  value: /tmp/.triton
                - name: NVTX_DISABLE
                  value: "1"
                - name: PYTORCH_CUDA_ALLOC_CONF
                  value: "expandable_segments:True"
              resources:
                requests:
                  cpu: "16"
                  memory: "${TRAINING_MEMORY}"
                  nvidia.com/gpu: "${TRAINING_GPUS_PER_NODE}"
                limits:
                  cpu: "16"
                  memory: "${TRAINING_MEMORY}"
                  nvidia.com/gpu: "${TRAINING_GPUS_PER_NODE}"
              volumeMounts:
                - name: ray-tmp
                  mountPath: /tmp
                - name: shm
                  mountPath: /dev/shm
                - name: hf-cache
                  mountPath: /hf-cache
          volumes:
            - name: ray-tmp
              emptyDir: {}
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "${SHM_SIZE}"
YAML
    fi
}

if [ "${1:-}" = "--generate-cluster" ]; then
    generate_cluster_yaml
    exit 0
fi

# ── Submit training job ─────────────────────────────────────────

set -x

ray job submit \
    --address="${RAY_ADDRESS}" \
    --runtime-env-json="{
        \"working_dir\": \"training/swe_bench/\",
        \"env_vars\": {
            \"SWE_MAX_STEPS\": \"${SWE_MAX_STEPS}\",
            \"SWE_IMAGE_REGISTRY\": \"${SWE_IMAGE_REGISTRY}\",
            \"SWE_K8S_NAMESPACE\": \"${SWE_K8S_NAMESPACE}\",
            \"SWE_SERVICE_ACCOUNT\": \"${SWE_SERVICE_ACCOUNT}\",
            \"SWE_MAX_ROLLOUT_RETRIES\": \"${SWE_MAX_ROLLOUT_RETRIES}\"
        }
    }" \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    \
    --actor.model_name_or_path "${MODEL_PATH}" \
    --ckpt.output_dir "${SAVE_PATH}" \
    --ckpt.path "${SAVE_PATH}/ckpt" \
    --ckpt.save_hf \
    --ckpt.max_num 3 \
    --ckpt.save_steps 20 \
    \
    --train.agent_func_path "agent_func.py" \
    --data.prompt_dataset "${DATASET}" \
    --data.input_key input \
    --data.label_key label \
    --data.max_len 16384 \
    --rollout.max_new_tokens 12288 \
    \
    --rollout.batch_size 4 \
    --rollout.n_samples_per_prompt 4 \
    --train.batch_size 8 \
    --train.micro_batch_size 1 \
    --rollout.micro_batch_size 1 \
    --data.max_samples 40 \
    --train.max_epochs 1 \
    --train.num_episodes 1 \
    --train.dynamic_batch_enable \
    \
    --actor.num_nodes "${TRAINING_NODES}" \
    --actor.num_gpus_per_node "${TRAINING_GPUS_PER_NODE}" \
    --ref.num_nodes "${TRAINING_NODES}" \
    --ref.num_gpus_per_node "${TRAINING_GPUS_PER_NODE}" \
    ${COLOCATION_FLAGS} \
    --vllm.num_engines "${VLLM_ENGINES}" \
    --vllm.tensor_parallel_size "${VLLM_TP}" \
    --vllm.gpu_memory_utilization 0.45 \
    --vllm.sync_backend nccl \
    --vllm.enforce_eager \
    \
    --ds.zero_stage "${ZERO_STAGE}" \
    --actor.gradient_checkpointing_enable \
    --ds.adam_offload \
    --ds.param_dtype bf16 \
    \
    --algo.advantage.estimator group_norm \
    --actor.adam.lr 5e-7 \
    --algo.kl.init_coef 0.01 \
    --algo.kl.use_loss \
    --algo.kl.estimator k2 \
    \
    --logger.tensorboard_dir "${SAVE_PATH}/runs" \
    --logger.logging_steps 1 \
    --eval.steps -1
