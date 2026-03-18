#!/bin/bash
# =============================================================================
# MaxMin-RLHF Smoke Test
#
# Quick validation that the full MaxMin pipeline works end-to-end.
# Uses tiny models (pythia-160m) and minimal episodes.
#
# Hardware: CPU or 1x small GPU
# Runtime: ~5-10 minutes
# =============================================================================

set -euo pipefail

BASE_MODEL="EleutherAI/pythia-160m"
DATASET="trl-internal-testing/descriptiveness-sentiment-trl-style"
DATASET_SPLIT="descriptiveness"
OUTPUT_ROOT="outputs/bench_maxmin_smoke"

echo "============================================================"
echo " MaxMin-RLHF Smoke Test (pythia-160m)"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Test 1: MaxMin PPO with 2 reward models (min strategy)
# ---------------------------------------------------------------------------
echo "[Test 1/3] MaxMin PPO (min strategy, 2 reward models)"
python examples/scripts/ppo/maxmin_ppo.py \
    --dataset_name "${DATASET}" \
    --dataset_train_split "${DATASET_SPLIT}" \
    --learning_rate 3e-6 \
    --output_dir "${OUTPUT_ROOT}/maxmin_min" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --total_episodes 32 \
    --model_name_or_path "${BASE_MODEL}" \
    --sft_model_path "${BASE_MODEL}" \
    --reward_model_path "${BASE_MODEL}" \
    --reward_model_paths "${BASE_MODEL}" "${BASE_MODEL}" \
    --maxmin_strategy min \
    --response_length 16 \
    --missing_eos_penalty 1.0 \
    --logging_steps 1 \
    --num_sample_generations 1 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1
echo "  PASSED"
echo ""

# ---------------------------------------------------------------------------
# Test 2: MaxMin PPO with softmin strategy
# ---------------------------------------------------------------------------
echo "[Test 2/3] MaxMin PPO (softmin strategy)"
python examples/scripts/ppo/maxmin_ppo.py \
    --dataset_name "${DATASET}" \
    --dataset_train_split "${DATASET_SPLIT}" \
    --learning_rate 3e-6 \
    --output_dir "${OUTPUT_ROOT}/maxmin_softmin" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --total_episodes 32 \
    --model_name_or_path "${BASE_MODEL}" \
    --sft_model_path "${BASE_MODEL}" \
    --reward_model_path "${BASE_MODEL}" \
    --reward_model_paths "${BASE_MODEL}" "${BASE_MODEL}" \
    --maxmin_strategy softmin \
    --softmin_temperature 0.1 \
    --response_length 16 \
    --missing_eos_penalty 1.0 \
    --logging_steps 1 \
    --num_sample_generations 1 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1
echo "  PASSED"
echo ""

# ---------------------------------------------------------------------------
# Test 3: MaxMin PPO with 3 reward models
# ---------------------------------------------------------------------------
echo "[Test 3/3] MaxMin PPO (3 reward models)"
python examples/scripts/ppo/maxmin_ppo.py \
    --dataset_name "${DATASET}" \
    --dataset_train_split "${DATASET_SPLIT}" \
    --learning_rate 3e-6 \
    --output_dir "${OUTPUT_ROOT}/maxmin_3rm" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --total_episodes 32 \
    --model_name_or_path "${BASE_MODEL}" \
    --sft_model_path "${BASE_MODEL}" \
    --reward_model_path "${BASE_MODEL}" \
    --reward_model_paths "${BASE_MODEL}" "${BASE_MODEL}" "${BASE_MODEL}" \
    --maxmin_strategy min \
    --response_length 16 \
    --missing_eos_penalty 1.0 \
    --logging_steps 1 \
    --num_sample_generations 1 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1
echo "  PASSED"
echo ""

echo "============================================================"
echo " All smoke tests passed!"
echo "============================================================"

# Cleanup
rm -rf "${OUTPUT_ROOT}"
echo "Cleaned up ${OUTPUT_ROOT}"
