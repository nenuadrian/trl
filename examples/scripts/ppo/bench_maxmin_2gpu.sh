#!/bin/bash
# =============================================================================
# MaxMin-RLHF on 2 GPUs
#
# Two options depending on your VRAM:
#   - 2x A100 80GB  → ZeRO-2 (faster, keeps full weights on each GPU)
#   - 2x A100 40GB  → ZeRO-3 (shards weights, fits in less VRAM)
#   - 2x A10G 24GB  → ZeRO-3 + LoRA (only viable option for small GPUs)
#
# Why 2 GPUs help for MaxMin:
#   MaxMin PPO loads N reward models (vs 1 for standard PPO).
#   With 2 RMs on a 7B model, total VRAM ≈ 70GB in bf16 (policy + ref +
#   value + 2 RMs). A single 80GB GPU barely fits this.
#   2 GPUs with ZeRO-2 cut optimizer memory in half (~35GB saved).
#   2 GPUs with ZeRO-3 shard weights too, so even 40GB GPUs work.
#
# Speed gain:
#   - Throughput roughly 1.5-1.8x (not 2x due to communication overhead)
#   - Generation step is the bottleneck and parallelizes well
#   - Reward model inference (the MaxMin-specific cost) is embarrassingly
#     parallel across the batch, so scales linearly with GPUs
#
# Hardware: 2x GPUs (A100/A10G/H100)
# Runtime: ~1-3 hours depending on model and GPU
# =============================================================================

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-allenai/tulu-2-7b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/bench_maxmin_2gpu}"
TOTAL_EPISODES="${TOTAL_EPISODES:-5000}"
LR=1.41e-5

# Auto-detect GPU VRAM to pick the right config
GPU_MEM_MB=$(python3 -c "
import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_properties(0).total_mem // (1024*1024))
else:
    print(0)
" 2>/dev/null || echo "0")

if [ "${GPU_MEM_MB}" -ge 70000 ]; then
    DS_CONFIG="examples/accelerate_configs/deepspeed_zero2_2gpu.yaml"
    USE_PEFT=""
    BATCH_SIZE=16
    echo "Detected >=70GB VRAM per GPU → using ZeRO-2 (no LoRA)"
elif [ "${GPU_MEM_MB}" -ge 35000 ]; then
    DS_CONFIG="examples/accelerate_configs/deepspeed_zero3_2gpu.yaml"
    USE_PEFT=""
    BATCH_SIZE=8
    echo "Detected >=35GB VRAM per GPU → using ZeRO-3 (no LoRA)"
else
    DS_CONFIG="examples/accelerate_configs/deepspeed_zero3_2gpu.yaml"
    USE_PEFT="--use_peft --lora_r 16"
    BATCH_SIZE=4
    echo "Detected <35GB VRAM per GPU → using ZeRO-3 + LoRA"
fi

echo ""
echo "============================================================"
echo " MaxMin-RLHF 2-GPU Benchmark"
echo " Model:     ${BASE_MODEL}"
echo " Config:    ${DS_CONFIG}"
echo " Batch:     ${BATCH_SIZE}/GPU"
echo " Episodes:  ${TOTAL_EPISODES}"
echo "============================================================"
echo ""

# Use base model as placeholder RMs (replace with EM-trained models for real runs)
RM_0="${RM_0:-${BASE_MODEL}}"
RM_1="${RM_1:-${BASE_MODEL}}"

# ---------------------------------------------------------------------------
# MaxMin PPO (2 GPUs)
# ---------------------------------------------------------------------------
echo "[1/2] MaxMin PPO (2 RMs, min strategy)"

accelerate launch --config_file "${DS_CONFIG}" \
    examples/scripts/ppo/maxmin_ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir "${OUTPUT_ROOT}/maxmin_ppo" \
    --learning_rate ${LR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 4 \
    --total_episodes ${TOTAL_EPISODES} \
    --model_name_or_path "${BASE_MODEL}" \
    --sft_model_path "${BASE_MODEL}" \
    --reward_model_path "${RM_0}" \
    --reward_model_paths "${RM_0}" "${RM_1}" \
    --maxmin_strategy min \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --local_rollout_forward_batch_size ${BATCH_SIZE} \
    --response_length 128 \
    --missing_eos_penalty 1.0 \
    --logging_steps 5 \
    --num_sample_generations 3 \
    --report_to wandb \
    --run_name "maxmin_ppo_2gpu" \
    ${USE_PEFT}

echo "  -> Saved to ${OUTPUT_ROOT}/maxmin_ppo"
echo ""

# ---------------------------------------------------------------------------
# Single PPO baseline (2 GPUs, for comparison)
# ---------------------------------------------------------------------------
echo "[2/2] Single Reward PPO baseline"

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
    --reward_model_path "${RM_0}" \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --local_rollout_forward_batch_size ${BATCH_SIZE} \
    --response_length 128 \
    --missing_eos_penalty 1.0 \
    --logging_steps 5 \
    --num_sample_generations 3 \
    --report_to wandb \
    --run_name "single_ppo_2gpu" \
    ${USE_PEFT}

echo "  -> Saved to ${OUTPUT_ROOT}/single_ppo"
echo ""

echo "============================================================"
echo " 2-GPU benchmark complete! Results in: ${OUTPUT_ROOT}"
echo "============================================================"
