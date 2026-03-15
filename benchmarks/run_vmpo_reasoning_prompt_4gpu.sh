#!/usr/bin/env bash
set -euo pipefail

# VMPO on a math/reasoning dataset (prompt-only, online generation).
# Uses exact-match reasoning reward from benchmarks/rewards/.
#
# Overridable env vars:
#   CUDA_VISIBLE_DEVICES  (default: 0,1,2,3)
#   NUM_PROCESSES         (default: 4)
#   OUTPUT_DIR            (default: Qwen2-0.5B-VMPO-reasoning-4gpu-stable)
#   MODEL                 (default: Qwen/Qwen2-0.5B-Instruct)
#   DATASET               (default: openai/gsm8k)
#   DATASET_CONFIG        (default: main)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-Qwen2-0.5B-VMPO-reasoning-4gpu-stable}"
MODEL="${MODEL:-Qwen/Qwen2-0.5B-Instruct}"
DATASET="${DATASET:-openai/gsm8k}"
DATASET_CONFIG="${DATASET_CONFIG:-main}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

accelerate launch \
  --config_file "${SCRIPT_DIR}/accelerate_4gpu_fp16.yaml" \
  --main_process_port 0 \
  --num_processes "${NUM_PROCESSES}" \
  "${REPO_ROOT}/trl/scripts/vmpo.py" \
  --dataset_name "${DATASET}" \
  --dataset_config "${DATASET_CONFIG}" \
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
  --eval_steps 200 \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 400 \
  --output_dir "${OUTPUT_DIR}" \
  --no_remove_unused_columns \
  --report_to wandb \
  --fp16 True \
  --bf16 False
