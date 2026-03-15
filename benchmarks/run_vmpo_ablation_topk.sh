#!/usr/bin/env bash
set -euo pipefail

# Ablation: sweep vmpo_topk_fraction (E-step selection pressure).
# Tests fraction in {0.25, 0.5, 0.75, 1.0} with k=4.
# At 1.0 the E-step uses all samples (reduces toward soft AWR).
#
# Usage:  bash run_vmpo_ablation_topk.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MODEL="${MODEL:-Qwen/Qwen2-0.5B-Instruct}"
REWARD_MODEL="${REWARD_MODEL:-RLHFlow/ArmoRM-Llama3-8B-v0.1}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

for FRAC in 0.25 0.5 0.75 1.0; do
  echo "=== Running VMPO with topk_fraction=${FRAC} ==="
  OUTPUT_DIR="ablation-vmpo-topk-${FRAC}"

  accelerate launch \
    --config_file "${SCRIPT_DIR}/accelerate_4gpu_fp16.yaml" \
    --main_process_port 0 \
    --num_processes "${NUM_PROCESSES}" \
    "${REPO_ROOT}/trl/scripts/vmpo.py" \
    --dataset_name trl-lib/ultrafeedback-prompt \
    --model_name_or_path "${MODEL}" \
    --reward_model_name_or_path "${REWARD_MODEL}" \
    --attn_implementation eager \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing False \
    --vmpo_k 4 \
    --vmpo_topk_fraction "${FRAC}" \
    --vmpo_eps_eta 0.1 \
    --vmpo_eps_alpha 0.1 \
    --vmpo_kl_estimator old_policy_ref \
    --vmpo_old_policy_sync_steps 16 \
    --vmpo_advantage_baseline ema \
    --max_new_tokens 256 \
    --temperature 0.7 \
    --eval_strategy steps \
    --eval_steps 200 \
    --logging_steps 10 \
    --output_dir "${OUTPUT_DIR}" \
    --no_remove_unused_columns \
    --report_to wandb \
    --fp16 True \
    --bf16 False

  echo "=== Finished topk_fraction=${FRAC} ==="
done
