#!/usr/bin/env bash
set -euo pipefail

# DPO 2-GPU benchmark preset for comparison against DAR/VMPO runs.
#
# Optional overrides:
#   export CUDA_VISIBLE_DEVICES=2,3
#   export NUM_PROCESSES=2
#   export DATASET_NAME=trl-lib/ultrafeedback_binarized
#   export DATASET_CONFIG=<config_name>
#   export OUTPUT_DIR=Qwen2-0.5B-DPO-2gpu-ufbinarized

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "NUM_PROCESSES must be >= 1"
  exit 1
fi

DATASET_NAME="${DATASET_NAME:-trl-lib/ultrafeedback_binarized}"
OUTPUT_DIR="${OUTPUT_DIR:-Qwen2-0.5B-DPO-2gpu-ufbinarized}"

DATASET_CONFIG_ARG=()
if [[ -n "${DATASET_CONFIG:-}" ]]; then
  DATASET_CONFIG_ARG=(--dataset_config "${DATASET_CONFIG}")
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

accelerate launch \
  --config_file "${SCRIPT_DIR}/accelerate_4gpu_fp16.yaml" \
  --main_process_port 0 \
  --num_processes "${NUM_PROCESSES}" \
  "${REPO_ROOT}/trl/scripts/dpo.py" \
  --dataset_name "${DATASET_NAME}" \
  "${DATASET_CONFIG_ARG[@]}" \
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
  --no_remove_unused_columns \
  --report_to wandb \
  --fp16 True \
  --bf16 False
