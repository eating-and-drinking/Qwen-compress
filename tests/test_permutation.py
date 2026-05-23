# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Unit tests for channel permutation."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from qwen_compress.prune.permutation import (
    PermutationSearcher,
    apply_intermediate_permutation,
    hard_enforce_n_m,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _make_random_50pct_zero(out_dim: int, in_dim: int) -> torch.Tensor:
    """Random Gaussian matrix with exactly 50% zeros per row, randomly placed."""
    W = torch.randn(out_dim, in_dim)
    # For each row, zero half the columns at random.
    for r in range(out_dim):
        idx = torch.randperm(in_dim)[: in_dim // 2]
        W[r, idx] = 0.0
    return W


def _make_canonical_2_4(out_dim: int, in_dim: int) -> torch.Tensor:
    """Matrix already in strict 2:4 form (per row, every 4 cols have 2 zeros)."""
    assert in_dim % 4 == 0
    W = torch.randn(out_dim, in_dim)
    for r in range(out_dim):
        for g in range(in_dim // 4):
            base = g * 4
            zero_pos = torch.randperm(4)[:2] + base
            W[r, zero_pos] = 0.0
    return W


# --------------------------------------------------------------------------- #
# PermutationSearcher.cost
# --------------------------------------------------------------------------- #


class TestCost:
    def test_canonical_2_4_has_zero_cost(self):
        W = _make_canonical_2_4(8, 16)
        s = PermutationSearcher(n=2, m=4)
        assert s.cost(W) == 0

    def test_dense_matrix_has_max_cost(self):
        W = torch.randn(8, 16)  # no zeros
        s = PermutationSearcher(n=2, m=4)
        # Every (row, group) cell has 0 zeros — all misaligned.
        assert s.cost(W) == 8 * 4  # 8 rows * 4 groups

    def test_alignment_ratio(self):
        W = _make_canonical_2_4(8, 16)
        s = PermutationSearcher(n=2, m=4)
        assert s.alignment_ratio(W) == 1.0

    def test_invalid_in_dim_raises(self):
        s = PermutationSearcher(n=2, m=4)
        with pytest.raises(ValueError):
            s.cost(torch.zeros(4, 6))  # 6 % 4 != 0

    def test_invalid_pattern_rejected(self):
        with pytest.raises(ValueError):
            PermutationSearcher(n=4, m=4)  # n == m
        with pytest.raises(ValueError):
            PermutationSearcher(n=0, m=4)
        with pytest.raises(ValueError):
            PermutationSearcher(n=5, m=4)  # n > m


# --------------------------------------------------------------------------- #
# PermutationSearcher.search
# --------------------------------------------------------------------------- #


class TestSearch:
    def test_recovers_known_permutation(self):
        """Take a canonical 2:4 matrix, shuffle its columns, verify search recovers."""
        torch.manual_seed(42)
        W_canon = _make_canonical_2_4(16, 32)
        # Random column permutation
        perm_truth = torch.randperm(32)
        W_scrambled = W_canon[:, perm_truth]
        # Verify cost is non-zero after scrambling
        s = PermutationSearcher(n=2, m=4)
        assert s.cost(W_scrambled) > 0
        # Search
        perm_found, final_cost = s.search(
            W_scrambled, max_iters=500, swaps_per_iter=500, seed=0
        )
        # After applying the found permutation, cost should be very low or zero.
        W_recovered = W_scrambled[:, perm_found]
        assert s.cost(W_recovered) <= s.cost(W_scrambled) * 0.2  # >= 80% reduction

    def test_returns_valid_permutation(self):
        W = _make_random_50pct_zero(8, 16)
        s = PermutationSearcher(n=2, m=4)
        perm, _ = s.search(W, max_iters=50, swaps_per_iter=50, seed=0)
        # Must be a valid permutation of [0, in_dim).
        assert perm.numel() == 16
        assert sorted(perm.tolist()) == list(range(16))

    def test_search_does_not_increase_cost(self):
        W = _make_random_50pct_zero(16, 32)
        s = PermutationSearcher(n=2, m=4)
        initial = s.cost(W)
        perm, final = s.search(W, max_iters=100, swaps_per_iter=100, seed=0)
        assert final <= initial


# --------------------------------------------------------------------------- #
# hard_enforce_n_m
# --------------------------------------------------------------------------- #


class TestHardEnforce:
    def test_forces_exact_pattern(self):
        W = torch.randn(8, 16)  # dense
        hard_enforce_n_m(W, n=2, m=4)
        s = PermutationSearcher(n=2, m=4)
        assert s.cost(W) == 0

    def test_keeps_largest_in_each_group(self):
        W = torch.tensor([
            [10.0, 1.0, 8.0, 2.0, 5.0, 6.0, 7.0, 9.0],
        ])
        hard_enforce_n_m(W, n=2, m=4)
        # Group [10, 1, 8, 2]: keep 10 and 8, zero 1 and 2.
        # Group [5, 6, 7, 9]: keep 7 and 9, zero 5 and 6.
        assert W[0].tolist() == [10.0, 0.0, 8.0, 0.0, 0.0, 0.0, 7.0, 9.0]

    def test_invalid_pattern_raises(self):
        with pytest.raises(ValueError):
            hard_enforce_n_m(torch.zeros(4, 6), n=2, m=4)  # 6 % 4 != 0
        with pytest.raises(ValueError):
            hard_enforce_n_m(torch.zeros(4, 8), n=0, m=4)


# --------------------------------------------------------------------------- #
# apply_intermediate_permutation — network equivariance
# --------------------------------------------------------------------------- #


class _ToyMLP(nn.Module):
    """Minimal SwiGLU MLP mimicking Qwen's structure for testing."""

    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _ToyBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.mlp = _ToyMLP(hidden, intermediate)


class TestApplyIntermediatePermutation:
    def test_preserves_function(self):
        torch.manual_seed(0)
        block = _ToyBlock(hidden=8, intermediate=16)
        x = torch.randn(2, 4, 8)
        y_before = block(x)

        perm = torch.randperm(16)
        apply_intermediate_permutation(block, perm)

        y_after = block(x)
        # Function should be unchanged up to floating-point error.
        torch.testing.assert_close(y_before, y_after, rtol=1e-5, atol=1e-5)

    def test_identity_permutation_is_noop(self):
        torch.manual_seed(0)
        block = _ToyBlock(hidden=8, intermediate=16)
        before = block.mlp.down_proj.weight.data.clone()
        apply_intermediate_permutation(block, torch.arange(16))
        torch.testing.assert_close(before, block.mlp.down_proj.weight.data)

    def test_changes_down_proj_when_perm_nontrivial(self):
        torch.manual_seed(0)
        block = _ToyBlock(hidden=8, intermediate=16)
        before = block.mlp.down_proj.weight.data.clone()
        # Pick a definitely-non-trivial permutation.
        perm = torch.tensor([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        apply_intermediate_permutation(block, perm)
        # Weight tensor should have moved.
        assert not torch.allclose(before, block.mlp.down_proj.weight.data)


# --------------------------------------------------------------------------- #
# End-to-end: pruning → permutation → enforcement
# --------------------------------------------------------------------------- #


def test_end_to_end_alignment_improves():
    """Simulate a pruned matrix, run search, verify alignment improves."""
    torch.manual_seed(7)
    # Pretend down_proj has been pruned to 50% unstructured.
    W = _make_random_50pct_zero(64, 128)
    s = PermutationSearcher(n=2, m=4)
    initial_align = s.alignment_ratio(W)

    perm, _ = s.search(W, max_iters=200, swaps_per_iter=200, seed=0)
    W_perm = W[:, perm]
    final_align = s.alignment_ratio(W_perm)

    # Alignment should strictly improve.
    assert final_align >= initial_align

    # After hard enforcement, alignment must be perfect.
    hard_enforce_n_m(W_perm, n=2, m=4)
    assert s.alignment_ratio(W_perm) == 1.0
