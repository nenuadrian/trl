#!/bin/bash
# =============================================================================
# MaxMin-RLHF GPT-2 Benchmark (Section 5.1 of the paper)
#
# Replicates the small-scale experiment:
#   - Base model: GPT-2
#   - Two user groups: majority (positive sentiment) + minority (conciseness)
#   - Compares: Single reward RLHF vs MaxMin-RLHF
#
# Prerequisite: Train two reward models via EM or use pre-existing ones.
#   For quick testing, we use the same base model as placeholder reward models.
#   For proper replication, first run em_reward_learning.py to get real RMs.
#
# Hardware: 1x GPU (A100/A10G/V100 with >=16GB VRAM)
# Runtime: ~30 min per run on A100
# =============================================================================

set -euo pipefail

BASE_MODEL="gpt2"
DATASET="trl-internal-testing/descriptiveness-sentiment-trl-style"
DATASET_SPLIT="descriptiveness"
OUTPUT_ROOT="outputs/bench_maxmin_gpt2"
TOTAL_EPISODES=10000
BATCH_SIZE=64
LR=3e-6

echo "============================================================"
echo " MaxMin-RLHF GPT-2 Benchmark"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: EM Reward Learning (learn cluster-specific reward models)
# ---------------------------------------------------------------------------
echo "[Step 1/4] EM Reward Learning"
echo "  For a quick test, we skip EM and use the base model as placeholder RMs."
echo "  For full replication, run:"
echo "    python examples/scripts/ppo/em_reward_learning.py \\"
echo "      --model_name_or_path ${BASE_MODEL} \\"
echo "      --dataset_name trl-lib/ultrafeedback_binarized \\"
echo "      --num_clusters 2 --num_em_iterations 5 \\"
echo "      --output_dir ${OUTPUT_ROOT}/em_rewards \\"
echo "      --per_device_train_batch_size 8 --num_train_epochs 1 \\"
echo "      --learning_rate 1e-5 --max_length 512"
echo ""

# After EM, reward models would be at:
#   ${OUTPUT_ROOT}/em_rewards/final/reward_model_0
#   ${OUTPUT_ROOT}/em_rewards/final/reward_model_1
# For now, use the base model as both RMs for pipeline testing:
RM_0="${BASE_MODEL}"
RM_1="${BASE_MODEL}"

# ---------------------------------------------------------------------------
# Step 2: Single Reward PPO (baseline — demonstrates impossibility)
# ---------------------------------------------------------------------------
echo "[Step 2/4] Single Reward PPO (baseline)"
python examples/scripts/ppo/ppo.py \
    --dataset_name "${DATASET}" \
    --dataset_train_split "${DATASET_SPLIT}" \
    --learning_rate ${LR} \
    --output_dir "${OUTPUT_ROOT}/single_ppo" \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --total_episodes ${TOTAL_EPISODES} \
    --model_name_or_path "${BASE_MODEL}" \
    --sft_model_path "${BASE_MODEL}" \
    --reward_model_path "${RM_0}" \
    --missing_eos_penalty 1.0 \
    --logging_steps 10 \
    --num_sample_generations 5 \
    --report_to wandb \
    --run_name "single_ppo_gpt2"
echo "  -> Saved to ${OUTPUT_ROOT}/single_ppo"
echo ""

# ---------------------------------------------------------------------------
# Step 3: MaxMin PPO (proposed approach)
# ---------------------------------------------------------------------------
echo "[Step 3/4] MaxMin PPO"
python examples/scripts/ppo/maxmin_ppo.py \
    --dataset_name "${DATASET}" \
    --dataset_train_split "${DATASET_SPLIT}" \
    --learning_rate ${LR} \
    --output_dir "${OUTPUT_ROOT}/maxmin_ppo" \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --total_episodes ${TOTAL_EPISODES} \
    --model_name_or_path "${BASE_MODEL}" \
    --sft_model_path "${BASE_MODEL}" \
    --reward_model_path "${RM_0}" \
    --reward_model_paths "${RM_0}" "${RM_1}" \
    --maxmin_strategy min \
    --missing_eos_penalty 1.0 \
    --logging_steps 10 \
    --num_sample_generations 5 \
    --report_to wandb \
    --run_name "maxmin_ppo_gpt2"
echo "  -> Saved to ${OUTPUT_ROOT}/maxmin_ppo"
echo ""

# ---------------------------------------------------------------------------
# Step 4: Evaluation
# ---------------------------------------------------------------------------
echo "[Step 4/4] Evaluation"
echo "  Compare generations from both models on sentiment and conciseness."
echo "  Run the evaluation script:"
echo "    python examples/scripts/ppo/maxmin_ppo_gpt2_sentiment.py --mode evaluate \\"
echo "      --output_dir ${OUTPUT_ROOT}"
echo ""

python examples/scripts/ppo/maxmin_ppo_gpt2_sentiment.py \
    --mode evaluate \
    --base_model "${BASE_MODEL}" \
    --output_dir "${OUTPUT_ROOT}"

echo ""
echo "============================================================"
echo " Benchmark complete!"
echo " Results saved to: ${OUTPUT_ROOT}"
echo "============================================================"
echo ""
echo "Expected results (from paper Figure 5):"
echo "  Single Reward PPO: High sentiment, LOW conciseness (ignores minority)"
echo "  MaxMin PPO (ours): High sentiment, HIGH conciseness (serves both groups)"
