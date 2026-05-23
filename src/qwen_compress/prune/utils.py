# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Pruning helpers: sparsity statistics and N:M mask enforcement."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def compute_sparsity(model: nn.Module) -> Dict[str, float]:
    """Return the per-module sparsity ratio (zeros / total) and the aggregate."""
    total = 0
    zeros = 0
    per_module: Dict[str, float] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            mod_total = w.numel()
            mod_zeros = int((w == 0).sum().item())
            per_module[name] = mod_zeros / max(1, mod_total)
            total += mod_total
            zeros += mod_zeros
    per_module["__overall__"] = zeros / max(1, total)
    return per_module


def apply_mask(weight: torch.Tensor, mask: torch.Tensor) -> None:
    """In-place ``weight *= mask``. ``mask`` is expected to be ``{0, 1}`` valued."""
    if mask.shape != weight.shape:
        raise ValueError(f"mask shape {mask.shape} != weight shape {weight.shape}")
    weight.mul_(mask.to(weight.dtype))


def enforce_n_m_sparsity(weight: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """Return an ``{0,1}`` mask enforcing N:M sparsity along the *input* dim.

    For each group of ``m`` consecutive input weights, the ``n`` smallest by
    absolute value are zeroed (so the remaining ``m - n`` are kept).

    Example: ``n=2, m=4`` is the Ampere/Hopper "2:4" pattern.
    """
    if n <= 0 or m <= 0 or n >= m:
        raise ValueError(f"Invalid N:M pattern with N={n} M={m}")
    if weight.dim() != 2:
        raise ValueError(f"Expected 2-D weight, got {weight.dim()}-D")

    out_dim, in_dim = weight.shape
    if in_dim % m != 0:
        raise ValueError(f"in_dim={in_dim} is not divisible by M={m}")

    mask = torch.ones_like(weight, dtype=torch.bool)
    reshaped = weight.abs().view(out_dim, in_dim // m, m)
    # Indices of the ``n`` smallest entries per group.
    _, idx = torch.topk(reshaped, k=n, dim=-1, largest=False)
    flat_mask = mask.view(out_dim, in_dim // m, m)
    flat_mask.scatter_(-1, idx, False)
    return flat_mask.view(out_dim, in_dim).to(weight.dtype)
