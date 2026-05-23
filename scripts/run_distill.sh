#!/usr/bin/env bash
# Launch group-wise distillation: Qwen2.5-14B-Instruct -> Qwen2.5-3B
set -euo pipefail

CONFIG="${1:-configs/distill/qwen2_5_14b_to_3b.yaml}"
NUM_GPUS="${NUM_GPUS:-8}"

if [ "$NUM_GPUS" -gt 1 ]; then
  torchrun --nproc_per_node="$NUM_GPUS" \
    --master_port="${MASTER_PORT:-29500}" \
    -m qwen_compress.cli distill --config "$CONFIG"
else
  qwen-compress distill --config "$CONFIG"
fi
