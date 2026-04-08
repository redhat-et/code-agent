#!/bin/bash
#
# GRPO training of Qwen3-1.7B on GSM8K using OpenRLHF.
#
# Cluster layout (3 total GPUs):
#   - Head pod:       0 GPUs  (Ray GCS + dashboard)
#   - Rollout worker: 1 GPU   (vLLM inference engine)
#   - Training worker: 2 GPUs (actor + ref model via DeepSpeed ZeRO-2)
#
# Prerequisites:
#   - RayCluster deployed: oc apply -f infra/deploy/raycluster.yaml
#   - Port-forward active: oc port-forward svc/rl-training-cluster-head-svc 8265:8265
#   - Custom image with OpenRLHF: built from infra/images/Containerfile.ray-openrlhf
#
# Usage:
#   bash training/gsm8k/train_grpo.sh

set -euo pipefail

# ── Configurable ────────────────────────────────────────────────
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
DATASET="${DATASET:-openai/gsm8k@main}"
REWARD_FUNC="reward_func.py"  # relative to working_dir uploaded to Ray workers
SAVE_PATH="${SAVE_PATH:-/tmp/checkpoints/qwen3-1.7b-grpo-gsm8k}"

set -x

ray job submit \
    --address="${RAY_ADDRESS}" \
    --runtime-env-json="{\"working_dir\": \"training/gsm8k/\"}" \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    \
    --pretrain "${MODEL_PATH}" \
    --save_path "${SAVE_PATH}" \
    --ckpt_path "${SAVE_PATH}/ckpt" \
    --save_hf_ckpt \
    --max_ckpt_num 2 \
    --save_steps 50 \
    \
    --remote_rm_url "${REWARD_FUNC}" \
    --prompt_data "${DATASET}" \
    --input_key question \
    --label_key answer \
    --apply_chat_template \
    --packing_samples \
    \
    --max_len 2048 \
    \
    --rollout_batch_size 8 \
    --n_samples_per_prompt 4 \
    --train_batch_size 8 \
    --micro_train_batch_size 2 \
    --micro_rollout_batch_size 2 \
    --max_samples 40 \
    --max_epochs 1 \
    --num_episodes 1 \
    --use_dynamic_batch \
    \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.75 \
    --colocate_actor_ref \
    --vllm_sync_backend nccl \
    --enforce_eager \
    \
    --zero_stage 2 \
    --gradient_checkpointing \
    --adam_offload \
    --param_dtype bf16 \
    \
    --advantage_estimator group_norm \
    --actor_learning_rate 1e-6 \
    --init_kl_coef 0.01 \
    --use_kl_loss \
    --kl_estimator k2 \
    \
    --use_tensorboard "${SAVE_PATH}/runs" \
    --logging_steps 1 \
    --eval_steps -1
