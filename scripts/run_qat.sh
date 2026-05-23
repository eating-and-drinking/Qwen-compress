#!/usr/bin/env bash
# Launch INT8 QAT with QAD.
set -euo pipefail
CONFIG="${1:-configs/qat/int8_qad.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"

if [ "$NUM_GPUS" -gt 1 ]; then
  torchrun --nproc_per_node="$NUM_GPUS" \
    --master_port="${MASTER_PORT:-29501}" \
    -m qwen_compress.cli qat --config "$CONFIG"
else
  qwen-compress qat --config "$CONFIG"
fi
