#!/bin/bash
#
# GRPO training of Qwen3.5-9B on SWE-bench using OpenRLHF with multi-turn
# agent rollouts.
#
# The agent (SWEBenchAgentInstance) runs inside the vLLM rollout worker.
# Each step calls kubectl exec into a K8s Pod sandbox.
#
# Prerequisites:
#   - RBAC applied:     oc apply -f training/swe_bench/deploy/rbac-training.yaml
#   - Cluster created:  bash train_swe_bench_grpo.sh --generate-cluster | oc apply -f -
#   - Port-forward:     oc port-forward svc/swe-rl-cluster-head-svc 8265:8265
#   - Dataset built:    python training/swe_bench/build_dataset.py \
#                         --output /tmp/swe_bench_train.jsonl
#   - Project installed in Ray image: pip install -e ".[training,eval]"
#
# Usage:
#   bash train_swe_bench_grpo.sh                  # run training
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

# Number of GPUs for the training worker (DeepSpeed).
TRAINING_GPUS="${TRAINING_GPUS:-4}"

# Number of GPUs for the rollout worker (vLLM).
# In colocated mode this is ignored — rollout uses all TRAINING_GPUS.
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"

# vLLM tensor-parallel size (defaults to rollout GPU count).
VLLM_TP="${VLLM_TP:-}"

# DeepSpeed ZeRO stage.
#   3: model weights sharded (works with fewer GPUs, slower)
#   2: model replicated, optimizer/gradients sharded (faster, needs more GPUs)
ZERO_STAGE="${ZERO_STAGE:-3}"

# Worker pod memory requests.
TRAINING_MEMORY="${TRAINING_MEMORY:-128Gi}"
ROLLOUT_MEMORY="${ROLLOUT_MEMORY:-32Gi}"

# Shared-memory size for NCCL allreduce.
SHM_SIZE="${SHM_SIZE:-16Gi}"

# Container image for Ray workers.
RAY_IMAGE="${RAY_IMAGE:-quay.io/michaelclifford/ray-openrlhf:latest}"

# ═══════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════

RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
DATASET="${DATASET:-/tmp/swe_bench_train.jsonl}"
SAVE_PATH="${SAVE_PATH:-/tmp/checkpoints/qwen3.5-9b-grpo-swe}"

# Agent environment (forwarded to Ray workers via env vars)
SWE_MAX_STEPS="${SWE_MAX_STEPS:-100}"
SWE_IMAGE_REGISTRY="${SWE_IMAGE_REGISTRY:-}"
SWE_K8S_NAMESPACE="${SWE_K8S_NAMESPACE:-}"
SWE_SERVICE_ACCOUNT="${SWE_SERVICE_ACCOUNT:-swe-bench-training}"

# ═══════════════════════════════════════════════════════════════
# DERIVED SETTINGS — no need to edit below this line
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

if [ "${DEPLOY_MODE}" = "colocated" ]; then
    EFFECTIVE_ROLLOUT_GPUS="${TRAINING_GPUS}"
    COLOCATION_FLAGS="--colocate_all_models --vllm_enable_sleep --deepspeed_enable_sleep"
else
    EFFECTIVE_ROLLOUT_GPUS="${ROLLOUT_GPUS}"
    COLOCATION_FLAGS="--colocate_actor_ref"
fi

VLLM_TP="${VLLM_TP:-${EFFECTIVE_ROLLOUT_GPUS}}"

# ── Generate Ray cluster YAML ──────────────────────────────────

generate_cluster_yaml() {
    local sa="${SWE_SERVICE_ACCOUNT}"

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
        cat <<YAML
    # Single worker: all models colocated (${TRAINING_GPUS} GPUs)
    # vLLM and DeepSpeed take turns via sleep/wake memory offload.
    - groupName: worker
      replicas: 1
      minReplicas: 1
      maxReplicas: 1
      rayStartParams:
        num-gpus: "${TRAINING_GPUS}"
      template:
        spec:
          serviceAccountName: ${sa}
          containers:
            - name: ray-worker
              image: ${RAY_IMAGE}
              imagePullPolicy: Always
              resources:
                requests:
                  cpu: "16"
                  memory: "${TRAINING_MEMORY}"
                  nvidia.com/gpu: "${TRAINING_GPUS}"
                limits:
                  cpu: "16"
                  memory: "${TRAINING_MEMORY}"
                  nvidia.com/gpu: "${TRAINING_GPUS}"
              volumeMounts:
                - name: ray-tmp
                  mountPath: /tmp
                - name: shm
                  mountPath: /dev/shm
          volumes:
            - name: ray-tmp
              emptyDir: {}
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "${SHM_SIZE}"
YAML
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
          containers:
            - name: ray-worker
              image: ${RAY_IMAGE}
              imagePullPolicy: Always
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
          volumes:
            - name: ray-tmp
              emptyDir: {}
    # Training worker: DeepSpeed ZeRO-${ZERO_STAGE} (${TRAINING_GPUS} GPUs)
    - groupName: training
      replicas: 1
      minReplicas: 1
      maxReplicas: 1
      rayStartParams:
        num-gpus: "${TRAINING_GPUS}"
      template:
        spec:
          containers:
            - name: ray-worker
              image: ${RAY_IMAGE}
              imagePullPolicy: Always
              resources:
                requests:
                  cpu: "16"
                  memory: "${TRAINING_MEMORY}"
                  nvidia.com/gpu: "${TRAINING_GPUS}"
                limits:
                  cpu: "16"
                  memory: "${TRAINING_MEMORY}"
                  nvidia.com/gpu: "${TRAINING_GPUS}"
              volumeMounts:
                - name: ray-tmp
                  mountPath: /tmp
                - name: shm
                  mountPath: /dev/shm
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
            \"SWE_SERVICE_ACCOUNT\": \"${SWE_SERVICE_ACCOUNT}\"
        }
    }" \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    \
    --pretrain "${MODEL_PATH}" \
    --save_path "${SAVE_PATH}" \
    --ckpt_path "${SAVE_PATH}/ckpt" \
    --save_hf_ckpt \
    --max_ckpt_num 3 \
    --save_steps 20 \
    \
    --agent_func_path "agent_func.py" \
    --prompt_data "${DATASET}" \
    --input_key input \
    --label_key label \
    --apply_chat_template \
    \
    --prompt_max_len 4096 \
    --generate_max_len 131072 \
    \
    --rollout_batch_size 4 \
    --n_samples_per_prompt 4 \
    --train_batch_size 4 \
    --micro_train_batch_size 1 \
    --micro_rollout_batch_size 1 \
    --max_samples 40 \
    --max_epochs 1 \
    --num_episodes 1 \
    --use_dynamic_batch \
    \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node "${TRAINING_GPUS}" \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node "${TRAINING_GPUS}" \
    ${COLOCATION_FLAGS} \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size "${VLLM_TP}" \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_sync_backend nccl \
    --enforce_eager \
    \
    --zero_stage "${ZERO_STAGE}" \
    --gradient_checkpointing \
    --adam_offload \
    --param_dtype bf16 \
    \
    --advantage_estimator group_norm \
    --actor_learning_rate 5e-7 \
    --init_kl_coef 0.01 \
    --use_kl_loss \
    --kl_estimator k2 \
    \
    --use_tensorboard "${SAVE_PATH}/runs" \
    --logging_steps 1 \
    --eval_steps -1
