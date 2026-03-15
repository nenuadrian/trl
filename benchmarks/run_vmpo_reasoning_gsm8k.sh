#!/usr/bin/env bash
set -euo pipefail

# VMPO on GSM8K with reasoning reward (exact-match + format bonuses).
# Standalone script that doesn't depend on external _4gpu scripts.
#
# Usage:  bash run_vmpo_reasoning_gsm8k.sh
#   or:   CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 bash run_vmpo_reasoning_gsm8k.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MODEL="${MODEL:-Qwen/Qwen2-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-Qwen2-0.5B-VMPO-gsm8k}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

accelerate launch \
  --config_file "${SCRIPT_DIR}/accelerate_4gpu_fp16.yaml" \
  --main_process_port 0 \
  --num_processes "${NUM_PROCESSES}" \
  "${REPO_ROOT}/trl/scripts/vmpo.py" \
  --dataset_name openai/gsm8k \
  --dataset_config main \
  --model_name_or_path "${MODEL}" \
  --reward_funcs "rewards.reasoning_reward.reasoning_em_reward" \
  --attn_implementation eager \
  --learning_rate 5.0e-7 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing False \
  --vmpo_k 4 \
  --vmpo_topk_fraction 0.5 \
  --vmpo_eps_eta 0.1 \
  --vmpo_eps_alpha 0.05 \
  --vmpo_kl_estimator old_policy_ref \
  --vmpo_old_policy_sync_steps 16 \
  --vmpo_ref_anchor_coef 0.1 \
  --vmpo_advantage_baseline per_prompt \
  --max_new_tokens 512 \
  --temperature 0.8 \
  --eval_strategy steps \
  --eval_steps 100 \
  --logging_steps 5 \
  --save_strategy steps \
  --save_steps 200 \
  --output_dir "${OUTPUT_DIR}" \
  --no_remove_unused_columns \
  --report_to wandb \
  --fp16 True \
  --bf16 False
