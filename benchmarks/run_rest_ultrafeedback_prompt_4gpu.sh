#!/usr/bin/env bash
set -euo pipefail

# ReST_EM on UltraFeedback (prompt-only) with ArmoRM reward model.
# Each EM iteration: generate 8 completions/prompt, filter by reward threshold,
# then SFT from the base model on the best completions.
#
# Uses the same dataset and reward model as run_dar_ultrafeedback_prompt_4gpu.sh
# for direct algorithm comparison.
#
# Overridable env vars:
#   CUDA_VISIBLE_DEVICES  (default: 0,1,2,3)
#   NUM_PROCESSES         (default: 4)
#   OUTPUT_DIR            (default: Qwen2-0.5B-ReST-ultrafeedback-4gpu)
#   REWARD_MODEL          (default: RLHFlow/ArmoRM-Llama3-8B-v0.1)
#   MODEL                 (default: Qwen/Qwen2-0.5B-Instruct)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-Qwen2-0.5B-ReST-ultrafeedback-4gpu}"
MODEL="${MODEL:-Qwen/Qwen2-0.5B-Instruct}"
REWARD_MODEL="${REWARD_MODEL:-RLHFlow/ArmoRM-Llama3-8B-v0.1}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

accelerate launch \
  --config_file "${SCRIPT_DIR}/accelerate_4gpu_fp16.yaml" \
  --main_process_port 0 \
  --num_processes "${NUM_PROCESSES}" \
  "${REPO_ROOT}/trl/scripts/rest.py" \
  --dataset_name trl-lib/ultrafeedback-prompt \
  --model_name_or_path "${MODEL}" \
  --reward_model_name_or_path "${REWARD_MODEL}" \
  --attn_implementation eager \
  --num_iterations 3 \
  --num_samples_per_prompt 8 \
  --max_solutions_per_problem 4 \
  --reward_threshold 0.5 \
  --generation_temperature 0.7 \
  --generation_top_k 40 \
  --generation_top_p 1.0 \
  --generation_batch_size 4 \
  --reset_model_each_iteration True \
  --max_new_tokens 256 \
  --learning_rate 5.0e-6 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing False \
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
