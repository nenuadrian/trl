#!/usr/bin/env bash
set -euo pipefail

# Head-to-head comparison: ReST_EM vs VMPO vs DAR on GSM8K reasoning.
#
# Runs all three algorithms sequentially on the same dataset/model/hardware.
# Results can be compared in wandb under the same project.
#
# Usage:  bash run_rest_vs_vmpo_vs_dar_gsm8k.sh
#   or:   MODEL=Qwen/Qwen2-0.5B-Instruct WANDB_PROJECT=gsm8k-comparison bash ...
#
# Optional env vars:
#   MODEL              (default: Qwen/Qwen2-0.5B-Instruct)
#   CUDA_VISIBLE_DEVICES (default: 0,1)
#   NUM_PROCESSES      (default: 2)
#   WANDB_PROJECT      (default: trl-gsm8k-comparison)

export MODEL="${MODEL:-Qwen/Qwen2-0.5B-Instruct}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_PROCESSES="${NUM_PROCESSES:-2}"
export WANDB_PROJECT="${WANDB_PROJECT:-trl-gsm8k-comparison}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_NAME="${MODEL##*/}"

echo "============================================="
echo "  Algorithm comparison on GSM8K"
echo "  Model: ${MODEL}"
echo "  GPUs: ${CUDA_VISIBLE_DEVICES} (${NUM_PROCESSES} processes)"
echo "  Results logged to wandb project: ${WANDB_PROJECT}"
echo "============================================="

# ── 1. ReST_EM ──────────────────────────────────────────────────────────────
echo ""
echo ">>> [1/3] Running ReST_EM..."
OUTPUT_DIR="${BASE_NAME}-ReST-gsm8k" \
  bash "${SCRIPT_DIR}/run_rest_reasoning_gsm8k.sh"
echo "<<< ReST_EM done."

# ── 2. VMPO ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> [2/3] Running VMPO..."
OUTPUT_DIR="${BASE_NAME}-VMPO-gsm8k" \
  bash "${SCRIPT_DIR}/run_vmpo_reasoning_gsm8k.sh"
echo "<<< VMPO done."

# ── 3. DAR ──────────────────────────────────────────────────────────────────
# DAR uses a reward function rather than a reward model for GSM8K; we reuse
# the same reasoning reward.
echo ""
echo ">>> [3/3] Running DAR..."
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

OUTPUT_DIR="${BASE_NAME}-DAR-gsm8k" \
  accelerate launch \
    --config_file "${SCRIPT_DIR}/accelerate_4gpu_fp16.yaml" \
    --main_process_port 0 \
    --num_processes "${NUM_PROCESSES}" \
    "${REPO_ROOT}/trl/scripts/dar.py" \
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
    --dar_k 4 \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --eval_strategy steps \
    --eval_steps 100 \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 200 \
    --output_dir "${BASE_NAME}-DAR-gsm8k" \
    --no_remove_unused_columns \
    --report_to wandb \
    --fp16 True \
    --bf16 False
echo "<<< DAR done."

echo ""
echo "============================================="
echo "  All runs complete. Compare results in:"
echo "  https://wandb.ai/${WANDB_ENTITY:-}/${WANDB_PROJECT}"
echo "============================================="
