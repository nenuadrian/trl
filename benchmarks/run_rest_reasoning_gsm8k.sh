#!/usr/bin/env bash
set -euo pipefail

# ReST_EM on GSM8K with exact-match reasoning reward.
# Each EM iteration: generate 32 completions/prompt, filter correct ones (reward > 0),
# cap 10 solutions/problem, then SFT from the base model.
#
# Matches the hyperparameters from "Beyond Human Data" (arXiv:2312.06585):
#   - 32 samples/prompt, top-K=40, temperature=0.7
#   - Binary filter (reward_threshold=0.0 keeps reward > 0, i.e. exact-match correct)
#   - Cap 10 solutions per problem
#   - 3 EM iterations, each M-step resets to base model
#
# Usage:  bash run_rest_reasoning_gsm8k.sh
#   or:   CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 bash run_rest_reasoning_gsm8k.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MODEL="${MODEL:-Qwen/Qwen2-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-Qwen2-0.5B-ReST-gsm8k}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

accelerate launch \
  --config_file "${SCRIPT_DIR}/accelerate_4gpu_fp16.yaml" \
  --main_process_port 0 \
  --num_processes "${NUM_PROCESSES}" \
  "${REPO_ROOT}/trl/scripts/rest.py" \
  --dataset_name openai/gsm8k \
  --dataset_config main \
  --model_name_or_path "${MODEL}" \
  --reward_funcs "rewards.reasoning_reward.reasoning_em_reward" \
  --attn_implementation eager \
  --num_iterations 3 \
  --num_samples_per_prompt 32 \
  --max_solutions_per_problem 10 \
  --reward_threshold 0.0 \
  --generation_temperature 0.7 \
  --generation_top_k 40 \
  --generation_top_p 1.0 \
  --generation_batch_size 4 \
  --reset_model_each_iteration True \
  --max_new_tokens 512 \
  --learning_rate 1.0e-5 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing False \
  --eval_strategy no \
  --logging_steps 5 \
  --save_strategy epoch \
  --output_dir "${OUTPUT_DIR}" \
  --no_remove_unused_columns \
  --report_to wandb \
  --fp16 True \
  --bf16 False
