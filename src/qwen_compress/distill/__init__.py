# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Group-wise distillation: 14B teacher -> 3B student.

The headline API is :class:`GroupwiseDistillTrainer`. Loss components are
factored out into :mod:`qwen_compress.distill.losses` so they can be reused by
the QAT-with-distillation trainer in :mod:`qwen_compress.qat`.
"""

from qwen_compress.distill.groupwise import GroupAssignment, build_group_assignment
from qwen_compress.distill.losses import (
    DistillationLoss,
    HiddenStateMSELoss,
    KDDivergenceLoss,
)
from qwen_compress.distill.trainer import GroupwiseDistillTrainer

__all__ = [
    "DistillationLoss",
    "GroupAssignment",
    "GroupwiseDistillTrainer",
    "HiddenStateMSELoss",
    "KDDivergenceLoss",
    "build_group_assignment",
]
