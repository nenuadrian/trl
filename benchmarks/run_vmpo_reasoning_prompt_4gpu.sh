#!/usr/bin/env bash
set -euo pipefail

# VMPO reasoning preset:
# - Uses a local exact-match reward function:
#   benchmarks.rewards.reasoning_reward.reasoning_em_reward
# - Expects a prompt-only dataset with:
#   - prompt column: prompt
#   - answer column: answer (or solution/target/label/final_answer/ground_truth)
#
# Required:
#   export DATASET_NAME=<your_dataset>
#
# Optional:
#   export DATASET_CONFIG=<config_name>
#   export DATASET_TRAIN_SPLIT=train
#   export DATASET_TEST_SPLIT=test
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   export NUM_PROCESSES=4
#   export OUTPUT_DIR=Qwen2-0.5B-VMPO-reasoning-4gpu-stable

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [[ -z "${NUM_PROCESSES:-}" ]]; then
  IFS=',' read -r -a _gpus <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_PROCESSES="${#_gpus[@]}"
fi
if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "NUM_PROCESSES must be >= 1"
  exit 1
fi

DATASET_NAME="${DATASET_NAME:-}"
if [[ -z "${DATASET_NAME}" ]]; then
  echo "DATASET_NAME is required."
  echo "Example: export DATASET_NAME=my-org/my-reasoning-prompts"
  exit 1
fi

DATASET_TRAIN_SPLIT="${DATASET_TRAIN_SPLIT:-train}"
DATASET_TEST_SPLIT="${DATASET_TEST_SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-Qwen2-0.5B-VMPO-reasoning-${NUM_PROCESSES}gpu-stable}"

DATASET_CONFIG_ARG=()
if [[ -n "${DATASET_CONFIG:-}" ]]; then
  DATASET_CONFIG_ARG=(--dataset_config "${DATASET_CONFIG}")
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

accelerate launch \
  --config_file "${SCRIPT_DIR}/accelerate_4gpu_fp16.yaml" \
  --main_process_port 0 \
  --num_processes "${NUM_PROCESSES}" \
  "${REPO_ROOT}/trl/scripts/vmpo.py" \
  --dataset_name "${DATASET_NAME}" \
  "${DATASET_CONFIG_ARG[@]}" \
  --dataset_train_split "${DATASET_TRAIN_SPLIT}" \
  --dataset_test_split "${DATASET_TEST_SPLIT}" \
  --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
  --attn_implementation eager \
  --reward_funcs benchmarks.rewards.reasoning_reward.reasoning_em_reward \
  --learning_rate 2.0e-7 \
  --num_train_epochs 4 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing False \
  --eval_strategy steps \
  --eval_steps 200 \
  --output_dir "${OUTPUT_DIR}" \
  --no_remove_unused_columns \
  --vmpo_k 2 \
  --vmpo_kl_estimator old_policy_ref \
  --vmpo_old_policy_sync_steps 64 \
  --vmpo_ref_anchor_coef 0.5 \
  --vmpo_min_alpha 0.05 \
  --vmpo_alpha_lr 1e-3 \
  --vmpo_advantage_baseline ema \
  --vmpo_reward_ema_decay 0.98 \
  --vmpo_max_eta 100.0 \
  --vmpo_max_alpha 100.0 \
  --vmpo_kl_zero_tol 1e-6 \
  --vmpo_kl_warning_patience 20 \
  --temperature 0.7 \
  --top_p 0.95 \
  --top_k 50 \
  --max_new_tokens 256 \
  --report_to wandb \
  --fp16 True \
  --bf16 False
