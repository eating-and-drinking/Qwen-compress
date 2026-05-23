# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic_cudnn: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Parameters
    ----------
    seed:
        Global seed.
    deterministic_cudnn:
        If ``True``, force deterministic cuDNN algorithms. Significantly slower;
        use only for debugging.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
