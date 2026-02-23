#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash benchmarks/run_vmpo_ultrafeedback_prompt_4gpu.sh
#
# Optional overrides:
#   export CUDA_VISIBLE_DEVICES=4,5,6,7
#   export OUTPUT_DIR=Qwen2-0.5B-VMPO-online-4-prompts-k2
#   export NUM_PROCESSES=4

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
OUTPUT_DIR="${OUTPUT_DIR:-Qwen2-0.5B-VMPO-online-4-prompts-k2}"
if [[ -z "${NUM_PROCESSES:-}" ]]; then
  IFS=',' read -r -a _gpus <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_PROCESSES="${#_gpus[@]}"
fi
if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "NUM_PROCESSES must be >= 1"
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

accelerate launch \
  --config_file "${SCRIPT_DIR}/accelerate_4gpu_fp16.yaml" \
  --main_process_port 0 \
  --num_processes "${NUM_PROCESSES}" \
  "${REPO_ROOT}/trl/scripts/vmpo.py" \
  --dataset_name trl-lib/ultrafeedback-prompt \
  --reward_model_name_or_path Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback \
  --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
  --attn_implementation eager \
  --learning_rate 5.0e-7 \
  --num_train_epochs 4 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing False \
  --eval_strategy steps \
  --eval_steps 200 \
  --output_dir "${OUTPUT_DIR}" \
  --vmpo_k 2 \
  --vmpo_kl_estimator old_policy_ref \
  --vmpo_old_policy_sync_steps 16 \
  --vmpo_ref_anchor_coef 0.1 \
  --vmpo_advantage_baseline ema \
  --vmpo_reward_ema_decay 0.98 \
  --vmpo_max_eta 100.0 \
  --vmpo_max_alpha 100.0 \
  --vmpo_kl_zero_tol 1e-6 \
  --vmpo_kl_warning_patience 20 \
  --report_to wandb \
  --fp16 True \
  --bf16 False
