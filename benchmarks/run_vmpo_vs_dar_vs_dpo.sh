#!/usr/bin/env bash
set -euo pipefail

# Head-to-head comparison: VMPO vs DAR vs DPO on UltraFeedback.
# Runs all three methods sequentially with matched compute budget.
# All use the same model, 2 GPUs, and equivalent effective batch size.
#
# Usage:  bash run_vmpo_vs_dar_vs_dpo.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  Experiment 1/3: DPO baseline"
echo "============================================"
OUTPUT_DIR="comparison-dpo-uf" \
  bash "${SCRIPT_DIR}/run_dpo_ultrafeedback_binarized_2gpu.sh"

echo "============================================"
echo "  Experiment 2/3: DAR (k=2)"
echo "============================================"
OUTPUT_DIR="comparison-dar-uf-k2" \
  bash "${SCRIPT_DIR}/run_dar_ultrafeedback_prompt_2gpu.sh"

echo "============================================"
echo "  Experiment 3/3: VMPO (k=2)"
echo "============================================"
OUTPUT_DIR="comparison-vmpo-uf-k2" \
  bash "${SCRIPT_DIR}/run_vmpo_ultrafeedback_prompt_2gpu.sh"

echo "============================================"
echo "  All three experiments completed."
echo "  Compare runs in WandB."
echo "============================================"
