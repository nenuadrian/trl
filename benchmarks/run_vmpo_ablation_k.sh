#!/usr/bin/env bash
set -euo pipefail

# Ablation: sweep vmpo_k (number of completions per prompt).
# Tests k in {1, 2, 4, 8} to measure sample-efficiency vs compute trade-off.
#
# Usage:  bash run_vmpo_ablation_k.sh
#   or:   CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 bash run_vmpo_ablation_k.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MODEL="${MODEL:-Qwen/Qwen2-0.5B-Instruct}"
REWARD_MODEL="${REWARD_MODEL:-RLHFlow/ArmoRM-Llama3-8B-v0.1}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

for K in 1 2 4 8; do
  echo "=== Running VMPO with k=${K} ==="
  OUTPUT_DIR="ablation-vmpo-k${K}"

  # Adjust batch size inversely with k to keep memory ~constant
  if   [[ ${K} -le 2 ]]; then BS=2; GA=4
  elif [[ ${K} -le 4 ]]; then BS=1; GA=8
  else                         BS=1; GA=4
  fi

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
    --per_device_train_batch_size ${BS} \
    --gradient_accumulation_steps ${GA} \
    --gradient_checkpointing False \
    --vmpo_k ${K} \
    --vmpo_topk_fraction 0.5 \
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

  echo "=== Finished k=${K} ==="
done
