#!/usr/bin/env bash
set -euo pipefail

# Ablation: sweep KL estimator types.
# Compares ref, behavior, old_policy, old_policy_ref.
# This tests the GEMPI paper claim that old_policy_ref is the most stable.
#
# Usage:  bash run_vmpo_ablation_kl_estimator.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MODEL="${MODEL:-Qwen/Qwen2-0.5B-Instruct}"
REWARD_MODEL="${REWARD_MODEL:-RLHFlow/ArmoRM-Llama3-8B-v0.1}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

for KL_EST in ref behavior old_policy old_policy_ref; do
  echo "=== Running VMPO with kl_estimator=${KL_EST} ==="
  OUTPUT_DIR="ablation-vmpo-kl-${KL_EST}"

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
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing False \
    --vmpo_k 2 \
    --vmpo_topk_fraction 0.5 \
    --vmpo_eps_eta 0.1 \
    --vmpo_eps_alpha 0.1 \
    --vmpo_kl_estimator "${KL_EST}" \
    --vmpo_old_policy_sync_steps 16 \
    --vmpo_ref_anchor_coef 0.1 \
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

  echo "=== Finished kl_estimator=${KL_EST} ==="
done
