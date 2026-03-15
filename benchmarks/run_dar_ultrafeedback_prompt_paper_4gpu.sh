#!/usr/bin/env bash
set -euo pipefail

# DAR on UltraFeedback with paper-aligned hyperparameters (k=4, higher alpha).
# This configuration matches the settings reported in the GEMPI paper.
#
# Overridable env vars:
#   CUDA_VISIBLE_DEVICES  (default: 0,1,2,3)
#   NUM_PROCESSES         (default: 4)
#   OUTPUT_DIR            (default: Qwen2-0.5B-DAR-online-paper-4gpu-prompts-k4)
#   REWARD_MODEL          (default: RLHFlow/ArmoRM-Llama3-8B-v0.1)
#   MODEL                 (default: Qwen/Qwen2-0.5B-Instruct)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-Qwen2-0.5B-DAR-online-paper-4gpu-prompts-k4}"
MODEL="${MODEL:-Qwen/Qwen2-0.5B-Instruct}"
REWARD_MODEL="${REWARD_MODEL:-RLHFlow/ArmoRM-Llama3-8B-v0.1}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

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
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing False \
  --alpha 0.5 \
  --dar_k 4 \
  --dar_wclip 50.0 \
  --adv_norm per_prompt \
  --max_new_tokens 256 \
  --temperature 0.7 \
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
