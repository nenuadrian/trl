#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_PROCESSES="${NUM_PROCESSES:-2}"
export OUTPUT_DIR="${OUTPUT_DIR:-Qwen2-0.5B-DAR-online-2gpu-prompts-k2}"

bash "${SCRIPT_DIR}/run_dar_ultrafeedback_prompt_4gpu.sh"
