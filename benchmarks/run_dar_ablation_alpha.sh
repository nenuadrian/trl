#!/usr/bin/env bash
set -euo pipefail

# Ablation: sweep DAR alpha (reference regularization coefficient).
# alpha=0 reduces to pure advantage-weighted regression (no ref anchor).
# Higher alpha pushes the weights toward the reference policy.
#
# Usage:  bash run_dar_ablation_alpha.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MODEL="${MODEL:-Qwen/Qwen2-0.5B-Instruct}"
REWARD_MODEL="${REWARD_MODEL:-RLHFlow/ArmoRM-Llama3-8B-v0.1}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

for ALPHA in 0.0 0.01 0.1 0.5 1.0; do
  echo "=== Running DAR with alpha=${ALPHA} ==="
  OUTPUT_DIR="ablation-dar-alpha-${ALPHA}"

  accelerate launch \
    --config_file "${SCRIPT_DIR}/accelerate_4gpu_fp16.yaml" \
    --main_process_port 0 \
    --num_processes "${NUM_PROCESSES}" \
    "${REPO_ROOT}/trl/scripts/dar.py" \
    --dataset_name trl-lib/ultrafeedback-prompt \
    --model_name_or_path "${MODEL}" \
    --reward_model_name_or_path "${REWARD_MODEL}" \
    --attn_implementation eager \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing False \
    --alpha "${ALPHA}" \
    --dar_k 2 \
    --dar_wclip 50.0 \
    --adv_norm per_prompt \
    --max_new_tokens 256 \
    --temperature 0.7 \
    --eval_strategy steps \
    --eval_steps 200 \
    --logging_steps 10 \
    --output_dir "${OUTPUT_DIR}" \
    --no_remove_unused_columns \
    --report_to wandb \
    --fp16 True \
    --bf16 False

  echo "=== Finished alpha=${ALPHA} ==="
done
