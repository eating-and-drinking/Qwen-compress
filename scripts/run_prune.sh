#!/usr/bin/env bash
# Launch SparseGPT pruning + recovery FT.
set -euo pipefail
CONFIG="${1:-configs/prune/sparsegpt_50pct.yaml}"
qwen-compress prune --config "$CONFIG"
