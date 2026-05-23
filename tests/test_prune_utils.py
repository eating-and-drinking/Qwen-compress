# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Unit tests for pruning utilities."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from qwen_compress.prune.utils import apply_mask, compute_sparsity, enforce_n_m_sparsity


class TestComputeSparsity:
    def test_dense_is_zero(self):
        m = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        stats = compute_sparsity(m)
        assert stats["__overall__"] == 0.0

    def test_fully_pruned_is_one(self):
        lin = nn.Linear(8, 4)
        lin.weight.data.zero_()
        stats = compute_sparsity(lin)
        assert stats["__overall__"] == 1.0


class TestApplyMask:
    def test_zeroes_match_mask(self):
        w = torch.randn(4, 8)
        m = torch.zeros_like(w)
        m[::2] = 1
        apply_mask(w, m)
        assert (w[1::2] == 0).all()
        assert (w[::2] != 0).any()

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            apply_mask(torch.zeros(4, 8), torch.zeros(4, 4))


class TestNMSparsity:
    def test_2_4_pattern(self):
        w = torch.tensor([[4.0, 1.0, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0]])
        m = enforce_n_m_sparsity(w, n=2, m=4)
        # In each group of 4, the 2 smallest are zeroed.
        # Group 0: [4,1,3,2] -> keep 4 and 3, zero 1 and 2.
        # Group 1: [5,6,7,8] -> keep 7 and 8, zero 5 and 6.
        assert m[0].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]

    def test_invalid_pattern(self):
        with pytest.raises(ValueError):
            enforce_n_m_sparsity(torch.zeros(4, 8), n=0, m=4)
        with pytest.raises(ValueError):
            enforce_n_m_sparsity(torch.zeros(4, 8), n=4, m=4)
        with pytest.raises(ValueError):
            enforce_n_m_sparsity(torch.zeros(4, 6), n=2, m=4)  # 6 % 4 != 0
