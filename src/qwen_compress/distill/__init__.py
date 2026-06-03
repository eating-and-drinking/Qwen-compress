# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""MOT-FD: Monotonic Optimal Transport Functional Distillation.

Teacher decomposition (48 → 12 groups) with optimal transport alignment
and monotonic semantic-progression constraint.

The headline API is :class:`GroupwiseDistillTrainer`. Loss components are
factored out into :mod:`qwen_compress.distill.losses` so they can be reused
by the QAT-with-distillation trainer in :mod:`qwen_compress.qat`.
"""

from qwen_compress.distill.groupwise import GroupAssignment, build_group_assignment
from qwen_compress.distill.losses import (
    DistillationLoss,
    HiddenCosineLoss,
    KDLoss,
    OptimalTransportAlignLoss,
    sinkhorn,
)
from qwen_compress.distill.trainer import GroupwiseDistillTrainer

__all__ = [
    "DistillationLoss",
    "GroupAssignment",
    "GroupwiseDistillTrainer",
    "HiddenCosineLoss",
    "KDLoss",
    "OptimalTransportAlignLoss",
    "build_group_assignment",
    "sinkhorn",
]
