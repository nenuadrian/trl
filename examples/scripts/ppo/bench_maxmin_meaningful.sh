#!/bin/bash
# =============================================================================
# MaxMin-RLHF Meaningful Benchmark (2 GPUs)
#
# Uses Qwen2.5-1.5B-Instruct (small, fast) with two reward models that have
# DIFFERENT score heads (different random seeds), so the MaxMin mechanism
# actually triggers. Compares against single-RM PPO baseline.
#
# Key differences from bench_maxmin_2gpu.sh:
#   - Smaller model (1.5B vs 7B) → faster iteration, fits easily on 2 GPUs
#   - Different RM initializations → rm_0 and rm_1 give different scores
#   - MaxMin actually selects between RMs (selected_rm_idx varies)
#   - Includes comparison run with single PPO
#
# Hardware: 2x GPUs (any with >=24GB VRAM)
# Runtime: ~20-40 min total (both runs)
# =============================================================================

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/bench_maxmin_meaningful}"
TOTAL_EPISODES="${TOTAL_EPISODES:-5000}"
LR="${LR:-3e-6}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# Use ZeRO-2 by default (1.5B model is small enough)
DS_CONFIG="${DS_CONFIG:-examples/accelerate_configs/deepspeed_zero2_2gpu.yaml}"

echo ""
echo "============================================================"
echo " MaxMin-RLHF Meaningful Benchmark (2 GPUs)"
echo " Policy:    ${BASE_MODEL}"
echo " RMs:       2x ${BASE_MODEL} (different score head seeds)"
echo " Config:    ${DS_CONFIG}"
echo " Batch:     ${BATCH_SIZE}/GPU"
echo " Episodes:  ${TOTAL_EPISODES}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: MaxMin PPO with 2 differently-initialized reward models
# ---------------------------------------------------------------------------
echo "[1/2] MaxMin PPO (2 RMs with different seeds, min strategy)"

accelerate launch --config_file "${DS_CONFIG}" \
    examples/scripts/ppo/maxmin_ppo_bench.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir "${OUTPUT_ROOT}/maxmin_ppo" \
    --learning_rate ${LR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 4 \
    --total_episodes ${TOTAL_EPISODES} \
    --model_name_or_path "${BASE_MODEL}" \
    --sft_model_path "${BASE_MODEL}" \
    --reward_model_path "${BASE_MODEL}" \
    --reward_model_paths "${BASE_MODEL}" "${BASE_MODEL}" \
    --maxmin_strategy min \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --local_rollout_forward_batch_size ${BATCH_SIZE} \
    --kl_coef 0.05 \
    --response_length 128 \
    --temperature 0.7 \
    --missing_eos_penalty 1.0 \
    --logging_steps 1 \
    --num_sample_generations 3 \
    --bf16 \
    --gradient_checkpointing \
    --report_to wandb \
    --run_name "maxmin_ppo_meaningful"

echo "  -> Saved to ${OUTPUT_ROOT}/maxmin_ppo"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Single PPO baseline (same model, single RM, for comparison)
# ---------------------------------------------------------------------------
echo "[2/2] Single Reward PPO baseline (same model, 1 RM)"

accelerate launch --config_file "${DS_CONFIG}" \
    examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir "${OUTPUT_ROOT}/single_ppo" \
    --learning_rate ${LR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 4 \
    --total_episodes ${TOTAL_EPISODES} \
    --model_name_or_path "${BASE_MODEL}" \
    --sft_model_path "${BASE_MODEL}" \
    --reward_model_path "${BASE_MODEL}" \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --local_rollout_forward_batch_size ${BATCH_SIZE} \
    --kl_coef 0.05 \
    --response_length 128 \
    --temperature 0.7 \
    --missing_eos_penalty 1.0 \
    --logging_steps 1 \
    --num_sample_generations 3 \
    --bf16 \
    --gradient_checkpointing \
    --report_to wandb \
    --run_name "single_ppo_meaningful"

echo "  -> Saved to ${OUTPUT_ROOT}/single_ppo"
echo ""

echo "============================================================"
echo " Benchmark complete! Compare runs on wandb:"
echo "   - maxmin_ppo_meaningful  (2 RMs, min aggregation)"
echo "   - single_ppo_meaningful  (1 RM baseline)"
echo ""
echo " Key metrics to compare:"
echo "   - objective/scores       (should differ if MaxMin is working)"
echo "   - objective/kl           (MaxMin may be more conservative)"
echo "   - policy/entropy_avg     (MaxMin should preserve diversity)"
echo "   - maxmin/selected_rm_idx (should vary if RMs disagree)"
echo "============================================================"
