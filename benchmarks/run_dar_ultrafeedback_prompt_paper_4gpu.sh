#!/usr/bin/env bash
set -euo pipefail

# DAR paper-style preset (arXiv:2504.14177):
# - K-shot sampling: 4
# - weight clip: 20
# - total regularization alpha+beta: 0.05
# - alpha ratio: 10%
# - advantage normalization: batch
# - temperature: 0.9
# - learning rate: 1e-6
#
# Notes:
# - This launcher keeps `attn_implementation=eager` and disables gradient checkpointing
#   to avoid the SDPA shape error seen with online generation in this environment.

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

# Paper-ish optimization setup.
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-15}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
OPTIMIZER="${OPTIMIZER:-adafactor}"

# Paper-ish DAR knobs.
DAR_K="${DAR_K:-4}"
DAR_WCLIP="${DAR_WCLIP:-20}"
DAR_TOTAL_REG="${DAR_TOTAL_REG:-0.05}"
DAR_ALPHA_RATIO="${DAR_ALPHA_RATIO:-0.1}"
ADV_NORM="${ADV_NORM:-batch}"
TEMPERATURE="${TEMPERATURE:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

ALPHA="$(awk -v t="${DAR_TOTAL_REG}" -v r="${DAR_ALPHA_RATIO}" 'BEGIN {printf "%.6f", t*r}')"
BETA="$(awk -v t="${DAR_TOTAL_REG}" -v a="${ALPHA}" 'BEGIN {printf "%.6f", t-a}')"

# Paper reports effective batch size 128 with grad accumulation 16 in online AI-reward setting.
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-128}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
denom=$((NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))
if (( TARGET_EFFECTIVE_BATCH_SIZE % denom != 0 )); then
  echo "TARGET_EFFECTIVE_BATCH_SIZE (${TARGET_EFFECTIVE_BATCH_SIZE}) must be divisible by NUM_PROCESSES*GRADIENT_ACCUMULATION_STEPS (${denom})."
  exit 1
fi
PER_DEVICE_TRAIN_BATCH_SIZE=$((TARGET_EFFECTIVE_BATCH_SIZE / denom))
if (( PER_DEVICE_TRAIN_BATCH_SIZE < 1 )); then
  echo "Computed PER_DEVICE_TRAIN_BATCH_SIZE is < 1. Increase TARGET_EFFECTIVE_BATCH_SIZE or reduce NUM_PROCESSES/GRADIENT_ACCUMULATION_STEPS."
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-Qwen2-0.5B-DAR-online-paper-${NUM_PROCESSES}gpu-prompts-k${DAR_K}}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-4}"
EVAL_STEPS="${EVAL_STEPS:-200}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

accelerate launch \
  --config_file "${SCRIPT_DIR}/accelerate_4gpu_fp16.yaml" \
  --main_process_port 0 \
  --num_processes "${NUM_PROCESSES}" \
  "${REPO_ROOT}/trl/scripts/dar.py" \
  --dataset_name trl-lib/ultrafeedback-prompt \
  --reward_model_name_or_path Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback \
  --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
  --attn_implementation eager \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --optim "${OPTIMIZER}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --gradient_checkpointing False \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --output_dir "${OUTPUT_DIR}" \
  --dar_k "${DAR_K}" \
  --dar_wclip "${DAR_WCLIP}" \
  --alpha "${ALPHA}" \
  --beta "${BETA}" \
  --adv_norm "${ADV_NORM}" \
  --temperature "${TEMPERATURE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --report_to wandb \
  --fp16 True \
  --bf16 False
