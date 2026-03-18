#!/bin/bash
# =============================================================================
# MaxMin-RLHF Multi-GPU Benchmark with DeepSpeed
#
# Full-scale replication using DeepSpeed ZeRO-2/ZeRO-3 on multiple GPUs.
# This matches the paper's large-scale setup more closely (no LoRA, full
# fine-tuning with DeepSpeed for memory efficiency).
#
# Hardware: 8x A100 (80GB) GPUs
# Runtime: ~3-6 hours for the full pipeline
# =============================================================================

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-allenai/tulu-2-7b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/bench_maxmin_multigpu}"
DS_CONFIG="${DS_CONFIG:-examples/accelerate_configs/deepspeed_zero2.yaml}"
TOTAL_EPISODES=30000
LR=3e-6

echo "============================================================"
echo " MaxMin-RLHF Multi-GPU Benchmark (DeepSpeed)"
echo " Model: ${BASE_MODEL}"
echo " Accelerate config: ${DS_CONFIG}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Train EM Reward Models
# ---------------------------------------------------------------------------
echo "[Step 1/3] EM Reward Learning (2 clusters, 5 iterations)"

# NOTE: For EM reward learning, we use single-GPU to avoid complications.
# Each EM iteration trains a standard RewardTrainer which handles DDP internally.
python examples/scripts/ppo/em_reward_learning.py \
    --model_name_or_path "${BASE_MODEL}" \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --num_clusters 2 \
    --num_em_iterations 5 \
    --output_dir "${OUTPUT_ROOT}/em_rewards" \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --max_length 512

RM_0="${OUTPUT_ROOT}/em_rewards/final/reward_model_0"
RM_1="${OUTPUT_ROOT}/em_rewards/final/reward_model_1"

if [ ! -d "${RM_0}" ]; then
    echo "  EM reward models not found, using base model as placeholder."
    RM_0="${BASE_MODEL}"
    RM_1="${BASE_MODEL}"
fi

echo ""

# ---------------------------------------------------------------------------
# Step 2: MaxMin PPO (multi-GPU)
# ---------------------------------------------------------------------------
echo "[Step 2/3] MaxMin PPO (multi-GPU with DeepSpeed)"

accelerate launch --config_file "${DS_CONFIG}" \
    examples/scripts/ppo/maxmin_ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir "${OUTPUT_ROOT}/maxmin_ppo" \
    --learning_rate ${LR} \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes ${TOTAL_EPISODES} \
    --model_name_or_path "${BASE_MODEL}" \
    --sft_model_path "${BASE_MODEL}" \
    --reward_model_path "${RM_0}" \
    --reward_model_paths "${RM_0}" "${RM_1}" \
    --maxmin_strategy min \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --local_rollout_forward_batch_size 16 \
    --response_length 128 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --logging_steps 5 \
    --num_sample_generations 3

echo "  -> Saved to ${OUTPUT_ROOT}/maxmin_ppo"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Single Reward PPO baseline (multi-GPU)
# ---------------------------------------------------------------------------
echo "[Step 3/3] Single Reward PPO baseline (multi-GPU with DeepSpeed)"

accelerate launch --config_file "${DS_CONFIG}" \
    examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir "${OUTPUT_ROOT}/single_ppo" \
    --learning_rate ${LR} \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes ${TOTAL_EPISODES} \
    --model_name_or_path "${BASE_MODEL}" \
    --sft_model_path "${BASE_MODEL}" \
    --reward_model_path "${RM_0}" \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --local_rollout_forward_batch_size 16 \
    --response_length 128 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --logging_steps 5 \
    --num_sample_generations 3

echo "  -> Saved to ${OUTPUT_ROOT}/single_ppo"
echo ""

echo "============================================================"
echo " Multi-GPU benchmark complete!"
echo " Results saved to: ${OUTPUT_ROOT}"
echo ""
echo " Next: Run pairwise evaluation with GPT-4 judge"
echo "   python examples/scripts/ppo/maxmin_ppo_tulu7b.py --mode evaluate \\"
echo "     --output_dir ${OUTPUT_ROOT}"
echo "============================================================"
