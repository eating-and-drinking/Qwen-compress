# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""SparseGPT one-shot pruner for Qwen models.

Implements the algorithm from Frantar & Alistarh (2023), "SparseGPT: Massive
Language Models Can Be Accurately Pruned in One-Shot". The core update for a
linear layer with weight :math:`W \\in \\mathbb{R}^{d_{\\text{out}} \\times d_{\\text{in}}}`
and inputs :math:`X \\in \\mathbb{R}^{n \\times d_{\\text{in}}}` is:

.. code-block::

   H = 2 X^T X / n  + lambda I            # damped Hessian
   for each column j (in blocks):
       w_j = W[:, j]
       # Decide which rows to zero in this column (mask M_j)
       err = w_j / H_inv[j, j]
       w_j[~M_j] = 0
       # Propagate error to remaining columns
       for k > j (within block):
           W[:, k] -= H_inv[j, k] / H_inv[j, j] * err * (~M_j)

The implementation walks decoder blocks sequentially: for each block, we
forward calibration samples up to that block, collect input activations to
every Linear inside, prune all of them, then move on. Memory peaks at a single
block's worth of activations.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from tqdm import tqdm

from qwen_compress.models.qwen_wrapper import (
    QWEN_ALL_LINEARS,
    get_decoder_layers,
    get_linear_layers_in_block,
)
from qwen_compress.prune.utils import enforce_n_m_sparsity
from qwen_compress.utils.logging import get_logger

_logger = get_logger(__name__)


class _SparseGPTLayer:
    """Accumulates Hessian for one ``nn.Linear`` and runs the SparseGPT update."""

    def __init__(self, linear: nn.Linear) -> None:
        self.linear = linear
        device = linear.weight.device
        self.rows = linear.weight.shape[0]
        self.cols = linear.weight.shape[1]
        # H = 2 / n * X^T X accumulated incrementally.
        self.H = torch.zeros((self.cols, self.cols), device=device, dtype=torch.float32)
        self.nsamples = 0
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def register(self) -> None:
        def _hook(_mod, inp, _out):  # noqa: ANN001
            x = inp[0]
            if x.dim() == 3:
                # [B, T, C] -> [B*T, C]
                x = x.reshape(-1, x.shape[-1])
            x = x.to(torch.float32)
            # Running average of Hessian: scale previous H, add new contribution.
            n_new = x.shape[0]
            if self.nsamples + n_new == 0:
                return
            self.H.mul_(self.nsamples / (self.nsamples + n_new))
            self.nsamples += n_new
            self.H.addmm_(x.T, x, alpha=2.0 / self.nsamples)

        self._handle = self.linear.register_forward_hook(_hook)

    def unregister(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def prune(
        self,
        sparsity: float,
        sparsity_type: str,
        block_size: int,
        percdamp: float,
    ) -> Tuple[torch.Tensor, float]:
        """Run SparseGPT on this layer.

        Returns ``(mask, reconstruction_error)``.
        """
        W = self.linear.weight.data.clone().to(torch.float32)
        H = self.H

        # 1) Damped Hessian inversion.
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        damp = percdamp * torch.mean(torch.diag(H))
        diag_idx = torch.arange(self.cols, device=H.device)
        H[diag_idx, diag_idx] += damp

        # Cholesky of inverse via stable inversion-by-solve.
        # We compute the upper-triangular Cholesky factor of H^{-1}.
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        # 2) Determine N:M parameters for structured pruning, if applicable.
        if sparsity_type == "2:4":
            n, m = 2, 4
        elif sparsity_type == "4:8":
            n, m = 4, 8
        else:
            n, m = 0, 0

        # 3) Iterate columns in blocks.
        mask = torch.ones_like(W, dtype=torch.bool)
        for i1 in range(0, self.cols, block_size):
            i2 = min(i1 + block_size, self.cols)
            block_cols = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if n > 0 and m > 0:
                # N:M structured: mask is fixed per output-row across the whole block.
                # Standard SparseGPT does N:M per group-of-m columns; do that here.
                pass  # mask computed per column window below.

            mask_block = torch.ones_like(W1, dtype=torch.bool)

            if n == 0:
                # Unstructured: derive a global threshold from |W| / sqrt(diag(H⁻¹))
                # within this block, choose lowest-importance entries to zero.
                # Hinv1 is the upper-triangular Cholesky factor (H⁻¹ = UᵀU).
                # The true diagonal of H⁻¹ is column-wise ‖U_{:,j}‖².
                diag_hinv = (Hinv1 ** 2).sum(dim=0)  # [block_cols]
                tmp = (W1 ** 2) / diag_hinv.reshape((1, -1))
                k = int(round(sparsity * tmp.numel()))
                if k > 0:
                    thresh = torch.kthvalue(tmp.flatten(), k).values
                    mask_block = tmp > thresh

            for j in range(block_cols):
                w = W1[:, j]
                d = Hinv1[j, j]

                if n > 0 and m > 0:
                    # For N:M, recompute the per-row N:M mask using the running W1.
                    j_local = j
                    base = j_local - (j_local % m)
                    if j_local % m == 0 and base + m <= block_cols:
                        # Choose which n of the next m columns to zero per row.
                        chunk = W1[:, base : base + m]
                        chunk_hinv = Hinv1[base : base + m, base : base + m]
                        chunk_diag = (chunk_hinv ** 2).sum(dim=0)  # [m]
                        score = (chunk ** 2) / chunk_diag.unsqueeze(0)
                        _, idx = torch.topk(score, k=n, dim=-1, largest=False)
                        m_sub = torch.ones_like(chunk, dtype=torch.bool)
                        m_sub.scatter_(-1, idx, False)
                        mask_block[:, base : base + m] = m_sub
                q = w.clone()
                q[~mask_block[:, j]] = 0
                Q1[:, j] = q
                err = (w - q) / d
                # Update remaining columns within this block.
                W1[:, j:] -= err.unsqueeze(1) * Hinv1[j, j:].unsqueeze(0)
                Err1[:, j] = err

            W[:, i1:i2] = Q1
            mask[:, i1:i2] = mask_block
            # Propagate error to columns *after* this block.
            if i2 < self.cols:
                W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        # 4) Write back into the underlying linear in its original dtype.
        recon_err = float(((W - self.linear.weight.data.to(torch.float32)) ** 2).mean().item())
        self.linear.weight.data.copy_(W.to(self.linear.weight.dtype))
        if sparsity_type == "2:4":
            # Final clean-up: re-enforce structured pattern (small drift can creep in).
            mask = enforce_n_m_sparsity(self.linear.weight.data, n=2, m=4).bool()
            self.linear.weight.data.mul_(mask)
        elif sparsity_type == "4:8":
            mask = enforce_n_m_sparsity(self.linear.weight.data, n=4, m=8).bool()
            self.linear.weight.data.mul_(mask)
        else:
            self.linear.weight.data.mul_(mask)

        return mask, recon_err


class SparseGPTPruner:
    """Block-by-block SparseGPT pruner for Qwen-family causal LMs."""

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        sparsity_type: str = "unstructured",
        block_size: int = 128,
        percdamp: float = 0.01,
        target_linears: Tuple[str, ...] = QWEN_ALL_LINEARS,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        if sparsity_type not in {"unstructured", "2:4", "4:8"}:
            raise ValueError(f"Unsupported sparsity_type: {sparsity_type!r}")
        if not (0.0 <= sparsity < 1.0):
            raise ValueError("sparsity must be in [0, 1)")
        self.model = model
        self.sparsity = sparsity
        self.sparsity_type = sparsity_type
        self.block_size = block_size
        self.percdamp = percdamp
        self.target_linears = target_linears
        self.device = torch.device(device)

    @torch.no_grad()
    def prune(
        self,
        calibration_loader: torch.utils.data.DataLoader,
        save_dir: Optional[Path] = None,
    ) -> Dict[str, float]:
        """Prune the model in place using ``calibration_loader``.

        Returns a dict mapping ``"layer_idx.module"`` to reconstruction error.
        """
        self.model.eval()
        layers = get_decoder_layers(self.model)
        num_layers = len(layers)
        _logger.info(
            f"SparseGPT: pruning {num_layers} blocks @ {self.sparsity * 100:.1f}% "
            f"({self.sparsity_type}), block_size={self.block_size}"
        )

        # Cache the input to the *first* decoder layer for all calibration samples.
        # We swap the first layer for a no-op catcher to harvest the inputs.
        device = self.device
        try:
            self.model.model.embed_tokens.to(device)
        except AttributeError as e:
            raise RuntimeError("Model does not have `model.embed_tokens` (not a Qwen-like LM?).") from e

        cached_inputs: List[torch.Tensor] = []
        cached_attn_masks: List[torch.Tensor] = []
        cached_pos_ids: List[Optional[torch.Tensor]] = []

        class _Catcher(nn.Module):
            def __init__(self, inner: nn.Module) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, hidden_states, *args, **kwargs):
                cached_inputs.append(hidden_states.detach())
                cached_attn_masks.append(
                    kwargs.get("attention_mask", torch.ones(1, hidden_states.size(1)))
                )
                cached_pos_ids.append(kwargs.get("position_ids"))
                raise _StopIterationSignal

        # Wrap first layer for input capture with exception safety.
        original_layer0 = layers[0]
        layers[0] = _Catcher(original_layer0)
        try:
            for batch in calibration_loader:
                input_ids = batch.to(device) if isinstance(batch, torch.Tensor) else batch["input_ids"].to(device)
                try:
                    self.model(input_ids=input_ids, use_cache=False)
                except _StopIterationSignal:
                    continue
        finally:
            layers[0] = original_layer0

        if not cached_inputs:
            raise RuntimeError("No calibration inputs were captured.")

        hidden_states = torch.cat(cached_inputs, dim=0)
        # Concatenate per-sample attention masks for per-sample replay.
        attn_masks = torch.cat([m.to(device) for m in cached_attn_masks], dim=0)
        pos_ids = cached_pos_ids[0] if cached_pos_ids else None
        _logger.info(f"Captured {hidden_states.size(0)} calibration sequences.")

        recon_errors: Dict[str, float] = {}

        # Block-by-block pruning.
        for layer_idx, block in enumerate(tqdm(layers, desc="SparseGPT blocks")):
            block.to(device)

            linears = {
                name: mod
                for name, mod in get_linear_layers_in_block(block).items()
                if name in self.target_linears
            }
            wrappers = {name: _SparseGPTLayer(linear) for name, linear in linears.items()}
            for w in wrappers.values():
                w.register()

            # Collect activations for this block by replaying the cached inputs.
            # Each sample uses its own attention_mask for correct attention.
            # Outputs are discarded — we only need the forward to trigger Hessian hooks.
            for i in range(hidden_states.size(0)):
                inp = hidden_states[i : i + 1]
                sample_mask = attn_masks[i : i + 1]
                block(
                    inp,
                    attention_mask=sample_mask,
                    position_ids=pos_ids,
                )
            for w in wrappers.values():
                w.unregister()

            for name, w in wrappers.items():
                _, err = w.prune(
                    sparsity=self.sparsity,
                    sparsity_type=self.sparsity_type,
                    block_size=self.block_size,
                    percdamp=self.percdamp,
                )
                recon_errors[f"{layer_idx}.{name}"] = err

            # Re-forward with the now-pruned weights to update hidden_states for next block.
            new_states = []
            for i in range(hidden_states.size(0)):
                inp = hidden_states[i : i + 1]
                sample_mask = attn_masks[i : i + 1]
                out = block(
                    inp,
                    attention_mask=sample_mask,
                    position_ids=pos_ids,
                )
                new_states.append((out[0] if isinstance(out, tuple) else out).detach())
            hidden_states = torch.cat(new_states, dim=0)

            block.to("cpu")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        _logger.info(f"SparseGPT done. mean recon err = {sum(recon_errors.values()) / max(1, len(recon_errors)):.4e}")
        return recon_errors


class _StopIterationSignal(Exception):
    """Internal sentinel used by the first-layer catcher."""
