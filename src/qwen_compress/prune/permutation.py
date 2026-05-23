# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Channel permutation for N:M sparsity alignment.

After unstructured SparseGPT pruning, this module finds a column permutation
``π`` that maximises the alignment of zeros to N:M groups, then applies it in
a network-equivariant way and (optionally) hard-enforces the N:M pattern.

Three permutation domains exist in a transformer:

* **Intermediate dim** (rows of ``gate_proj`` / ``up_proj`` paired with columns
  of ``down_proj``) — fully local to one MLP block. Permuting freely changes
  ``down_proj``'s sparsity pattern without affecting anything else.
* **Hidden dim** — shared globally across all layers via residual connections.
  Helps ``q_proj`` / ``k_proj`` / ``v_proj`` / ``gate_proj`` / ``up_proj`` get
  to N:M but requires propagating the permutation through every embedding,
  norm and lm_head. Not implemented in this module (future work).
* **Head dim** — local to an attention block, but only "free" when ``num_heads
  == num_kv_heads`` (i.e. MHA, not GQA). Qwen2.5 uses GQA so head permutation
  is not exposed by default.

The greedy swap search follows the design of Pool & Yu (NeurIPS 2021)
"Channel Permutations for N:M Sparsity": iteratively pick a misaligned
4-column group and try swapping one of its columns with a column from another
group, accepting the swap if it reduces total misalignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from qwen_compress.models.qwen_wrapper import get_decoder_layers
from qwen_compress.utils.logging import get_logger

_logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Cost / search primitives
# --------------------------------------------------------------------------- #


class PermutationSearcher:
    """Greedy column-swap search for N:M sparsity alignment.

    Parameters
    ----------
    n, m:
        N:M pattern; each group of ``m`` consecutive columns must have exactly
        ``n`` zeros per row. Typical: ``(2, 4)``.
    """

    def __init__(self, n: int = 2, m: int = 4) -> None:
        if not (0 < n < m):
            raise ValueError(f"Invalid N:M = {n}:{m}; require 0 < N < M")
        self.n = n
        self.m = m

    # --------------------------------------------------------------- cost
    def cost(self, W: torch.Tensor) -> int:
        """Return the number of (row, group) cells violating the N:M pattern.

        A cell is misaligned when the number of zeros in that group on that row
        is not exactly :attr:`n`. Lower is better; 0 means perfectly N:M.
        """
        if W.dim() != 2:
            raise ValueError(f"Expected 2-D weight tensor, got {W.dim()}-D")
        out_dim, in_dim = W.shape
        if in_dim % self.m != 0:
            raise ValueError(f"in_dim={in_dim} not divisible by m={self.m}")
        zero_mask = (W == 0)
        groups = zero_mask.view(out_dim, in_dim // self.m, self.m)
        zeros_per_group = groups.sum(dim=-1)  # (out, num_groups)
        return int((zeros_per_group != self.n).sum().item())

    def alignment_ratio(self, W: torch.Tensor) -> float:
        """Fraction of (row, group) cells that *are* in N:M form (1.0 = perfect)."""
        out_dim, in_dim = W.shape
        total = out_dim * (in_dim // self.m)
        if total == 0:
            return 1.0
        return 1.0 - self.cost(W) / total

    # ------------------------------------------------------------ search
    def search(
        self,
        W: torch.Tensor,
        max_iters: int = 300,
        swaps_per_iter: int = 200,
        patience: int = 30,
        seed: int = 0,
    ) -> Tuple[torch.Tensor, int]:
        """Greedy column-swap search.

        Parameters
        ----------
        W:
            ``[out_dim, in_dim]`` weight tensor (zeros already in place).
        max_iters:
            Maximum number of outer-loop iterations.
        swaps_per_iter:
            Number of random swap candidates evaluated per iteration.
        patience:
            Stop early if no iteration produces an improvement for this many
            iterations.
        seed:
            RNG seed for reproducible swap sampling.

        Returns
        -------
        perm:
            ``LongTensor`` of shape ``(in_dim,)``. Apply via
            ``W_perm = W[:, perm]``.
        final_cost:
            Misalignment after search.
        """
        out_dim, in_dim = W.shape
        device = W.device
        num_groups = in_dim // self.m

        zero_mask = (W == 0).clone()
        perm = torch.arange(in_dim, device=device)

        initial_cost = self.cost(W)
        current_cost = initial_cost
        rng = torch.Generator(device="cpu").manual_seed(seed)
        no_improve = 0

        for it in range(max_iters):
            # Per-group misalignment for biased sampling.
            groups = zero_mask.view(out_dim, num_groups, self.m)
            zeros_per_group = groups.sum(dim=-1)  # (out, num_groups)
            mis_per_group = (zeros_per_group != self.n).sum(dim=0)  # (num_groups,)
            total_mis = int(mis_per_group.sum().item())
            if total_mis == 0:
                break

            probs = mis_per_group.float()
            probs = probs / probs.sum().clamp_min(1e-12)

            src_groups = torch.multinomial(
                probs.cpu(), swaps_per_iter, replacement=True, generator=rng
            )
            dst_groups = torch.randint(0, num_groups, (swaps_per_iter,), generator=rng)
            src_inner = torch.randint(0, self.m, (swaps_per_iter,), generator=rng)
            dst_inner = torch.randint(0, self.m, (swaps_per_iter,), generator=rng)

            improvements = 0
            for k in range(swaps_per_iter):
                ga = int(src_groups[k].item())
                gb = int(dst_groups[k].item())
                if ga == gb:
                    continue
                col_a = ga * self.m + int(src_inner[k].item())
                col_b = gb * self.m + int(dst_inner[k].item())

                # Delta cost of swapping col_a <-> col_b.
                ma = zero_mask[:, col_a].long()
                mb = zero_mask[:, col_b].long()
                old_za = zero_mask[:, ga * self.m : (ga + 1) * self.m].sum(dim=-1)
                old_zb = zero_mask[:, gb * self.m : (gb + 1) * self.m].sum(dim=-1)
                new_za = old_za - ma + mb
                new_zb = old_zb - mb + ma

                # Sum of independent misalignments for groups a and b.
                old_mis = (
                    int((old_za != self.n).sum().item())
                    + int((old_zb != self.n).sum().item())
                )
                new_mis = (
                    int((new_za != self.n).sum().item())
                    + int((new_zb != self.n).sum().item())
                )
                delta = new_mis - old_mis

                if delta < 0:
                    # Accept the swap.
                    tmp = zero_mask[:, col_a].clone()
                    zero_mask[:, col_a] = zero_mask[:, col_b]
                    zero_mask[:, col_b] = tmp
                    pa = perm[col_a].clone()
                    perm[col_a] = perm[col_b]
                    perm[col_b] = pa
                    current_cost += delta
                    improvements += 1

            if improvements > 0:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    _logger.debug(
                        f"Permutation search converged after {it + 1} iters "
                        f"(no improvement for {patience})"
                    )
                    break

        _logger.debug(
            f"Permutation search: cost {initial_cost} → {current_cost} "
            f"(reduction = {initial_cost - current_cost})"
        )
        return perm, current_cost


# --------------------------------------------------------------------------- #
# Hard enforcement
# --------------------------------------------------------------------------- #


def hard_enforce_n_m(W: torch.Tensor, n: int = 2, m: int = 4) -> torch.Tensor:
    """In-place: zero the ``n`` smallest-|W| entries in each group of ``m`` columns.

    Applies the canonical N:M mask, overriding the existing pattern. Use this
    *after* a permutation search to clean up the residual misalignment.

    Returns the same ``W`` (modified in place) for chaining.
    """
    if W.dim() != 2:
        raise ValueError(f"Expected 2-D tensor, got {W.dim()}-D")
    out_dim, in_dim = W.shape
    if in_dim % m != 0:
        raise ValueError(f"in_dim={in_dim} not divisible by m={m}")
    if not (0 < n < m):
        raise ValueError(f"Invalid N:M = {n}:{m}")

    abs_groups = W.abs().view(out_dim, in_dim // m, m)
    # Indices of the n smallest entries per group (these get zeroed).
    _, smallest_idx = torch.topk(abs_groups, k=n, dim=-1, largest=False)
    mask = torch.ones_like(abs_groups, dtype=torch.bool)
    mask.scatter_(-1, smallest_idx, False)
    W.mul_(mask.view(out_dim, in_dim).to(W.dtype))
    return W


# --------------------------------------------------------------------------- #
# Network-equivariant application
# --------------------------------------------------------------------------- #


def apply_intermediate_permutation(block: nn.Module, perm: torch.Tensor) -> None:
    """Apply an intermediate-dim permutation to one Qwen MLP block.

    Mathematically equivalent to inserting ``P`` after gate/up and ``P^T``
    before down — the block's input/output behaviour is unchanged.

    Modifies:
        ``block.mlp.gate_proj`` — rows permuted by ``perm``
        ``block.mlp.up_proj``  — rows permuted by ``perm``
        ``block.mlp.down_proj`` — columns permuted by ``perm``

    And the corresponding bias terms if present.
    """
    if not hasattr(block, "mlp"):
        raise AttributeError("Block has no `mlp` attribute (not a Qwen decoder layer?)")
    mlp = block.mlp
    perm_dev = perm.to(mlp.down_proj.weight.device)
    with torch.no_grad():
        # gate_proj.weight: (intermediate, hidden) -> permute rows
        mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[perm_dev, :].contiguous()
        if mlp.gate_proj.bias is not None:
            mlp.gate_proj.bias.data = mlp.gate_proj.bias.data[perm_dev].contiguous()

        # up_proj.weight: (intermediate, hidden) -> permute rows
        mlp.up_proj.weight.data = mlp.up_proj.weight.data[perm_dev, :].contiguous()
        if mlp.up_proj.bias is not None:
            mlp.up_proj.bias.data = mlp.up_proj.bias.data[perm_dev].contiguous()

        # down_proj.weight: (hidden, intermediate) -> permute columns
        mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, perm_dev].contiguous()


# --------------------------------------------------------------------------- #
# Block-level / model-level orchestration
# --------------------------------------------------------------------------- #


@dataclass
class BlockPermutationStats:
    """Per-block diagnostics returned by :func:`permute_block_for_2_4`."""

    layer_idx: int
    initial_cost: int
    after_perm_cost: int
    after_enforce_cost: int
    total_cells: int

    @property
    def initial_alignment(self) -> float:
        return 1.0 - self.initial_cost / max(1, self.total_cells)

    @property
    def post_perm_alignment(self) -> float:
        return 1.0 - self.after_perm_cost / max(1, self.total_cells)


def permute_block_for_2_4(
    block: nn.Module,
    layer_idx: int = 0,
    n: int = 2,
    m: int = 4,
    enforce: bool = True,
    max_iters: int = 300,
    swaps_per_iter: int = 200,
    seed: int = 0,
) -> BlockPermutationStats:
    """Search + apply intermediate-dim permutation on one decoder block.

    Optimises the column permutation that maximises ``down_proj``'s N:M
    alignment, applies it to ``gate_proj`` / ``up_proj`` rows and
    ``down_proj`` columns (preserving the block's function), and optionally
    hard-enforces N:M on ``down_proj``.
    """
    if not hasattr(block, "mlp"):
        raise AttributeError(f"Layer {layer_idx} has no MLP attribute.")
    mlp = block.mlp
    W = mlp.down_proj.weight.data

    searcher = PermutationSearcher(n=n, m=m)
    initial_cost = searcher.cost(W)
    total_cells = W.shape[0] * (W.shape[1] // m)

    perm, _ = searcher.search(
        W, max_iters=max_iters, swaps_per_iter=swaps_per_iter, seed=seed
    )
    apply_intermediate_permutation(block, perm)
    after_perm_cost = searcher.cost(mlp.down_proj.weight.data)

    if enforce:
        hard_enforce_n_m(mlp.down_proj.weight.data, n=n, m=m)
        after_enforce_cost = searcher.cost(mlp.down_proj.weight.data)
    else:
        after_enforce_cost = after_perm_cost

    return BlockPermutationStats(
        layer_idx=layer_idx,
        initial_cost=initial_cost,
        after_perm_cost=after_perm_cost,
        after_enforce_cost=after_enforce_cost,
        total_cells=total_cells,
    )


def permute_model_for_2_4(
    model: nn.Module,
    n: int = 2,
    m: int = 4,
    enforce: bool = True,
    max_iters: int = 300,
    swaps_per_iter: int = 200,
    seed: int = 0,
) -> Dict[str, float]:
    """Apply :func:`permute_block_for_2_4` to every decoder block in ``model``.

    Returns aggregate statistics across all layers.
    """
    layers = get_decoder_layers(model)
    per_layer: List[BlockPermutationStats] = []

    _logger.info(
        f"Channel permutation for {n}:{m} alignment across {len(layers)} blocks "
        f"(enforce={enforce})"
    )
    for idx, block in enumerate(layers):
        stats = permute_block_for_2_4(
            block,
            layer_idx=idx,
            n=n,
            m=m,
            enforce=enforce,
            max_iters=max_iters,
            swaps_per_iter=swaps_per_iter,
            seed=seed + idx,
        )
        per_layer.append(stats)
        _logger.info(
            f"  layer {idx:2d}: alignment "
            f"{stats.initial_alignment * 100:.1f}% → "
            f"{stats.post_perm_alignment * 100:.1f}% "
            f"(misalign {stats.initial_cost} → {stats.after_perm_cost}"
            + (f" → 0 after enforce" if enforce else "")
            + ")"
        )

    total_initial = sum(s.initial_cost for s in per_layer)
    total_after_perm = sum(s.after_perm_cost for s in per_layer)
    total_after_enforce = sum(s.after_enforce_cost for s in per_layer)
    total_cells = sum(s.total_cells for s in per_layer)

    return {
        "num_layers": len(per_layer),
        "total_cells": total_cells,
        "total_initial_misalignment": total_initial,
        "total_after_perm_misalignment": total_after_perm,
        "total_after_enforce_misalignment": total_after_enforce,
        "initial_alignment_pct": 100.0 * (1.0 - total_initial / max(1, total_cells)),
        "after_perm_alignment_pct": 100.0 * (1.0 - total_after_perm / max(1, total_cells)),
        "after_enforce_alignment_pct": 100.0 * (1.0 - total_after_enforce / max(1, total_cells)),
    }
