# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0 (see LICENSE for details).
"""qwen-compress: production-grade LLM compression toolkit for Qwen models.

The package exposes three independently-usable stages and an orchestrator:

* ``qwen_compress.distill``  - Group-wise distillation (14B -> 3B).
* ``qwen_compress.prune``    - SparseGPT non-structured pruning.
* ``qwen_compress.qat``      - Quantization-aware training with optional QAD.

Typical entry points::

    from qwen_compress.distill import GroupwiseDistillTrainer
    from qwen_compress.prune import SparseGPTPruner
    from qwen_compress.qat import QADTrainer, prepare_qat_model

Or use the CLI::

    qwen-compress distill --config configs/distill/qwen_14b_to_3b.yaml
    qwen-compress prune   --config configs/prune/sparsegpt_50pct.yaml
    qwen-compress qat     --config configs/qat/int8_qad.yaml
"""

from qwen_compress._version import __version__

__all__ = ["__version__"]
