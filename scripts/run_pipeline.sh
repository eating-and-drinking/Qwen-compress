#!/usr/bin/env bash
# End-to-end Qwen2.5-14B-Instruct -> 3B INT8 pipeline.
set -euo pipefail
CONFIG="${1:-configs/pipeline/full.yaml}"
qwen-compress pipeline --config "$CONFIG"
