# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Fake-quant building blocks for quantization-aware training.

The math: a fake-quantize module simulates the quantize -> dequantize round trip
during the forward pass so the model's training signal sees realistic quantization
noise, while gradients flow through unchanged (Straight-Through Estimator).

For an asymmetric quantizer with bit-width ``b``, scale ``s`` and zero-point ``z``::

    q = clamp(round(x / s) + z, qmin, qmax)
    x_dq = (q - z) * s

Symmetric quantizers omit ``z`` (fixed at 0) and use ``qmin = -(2**(b-1))``,
``qmax = 2**(b-1) - 1``. We support per-tensor and per-channel weight
quantization, and per-tensor or per-token activation quantization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from qwen_compress.models.qwen_wrapper import (
    QWEN_ALL_LINEARS,
    get_decoder_layers,
    get_linear_layers_in_block,
)
from qwen_compress.utils.logging import get_logger

_logger = get_logger(__name__)


def _q_range(bits: int, symmetric: bool) -> Tuple[int, int]:
    if symmetric:
        return -(2 ** (bits - 1)) + 1, 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


class _STEQuantize(torch.autograd.Function):
    """quantize-dequantize with straight-through gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor,
                qmin: int, qmax: int) -> torch.Tensor:  # noqa: ANN001
        ctx.save_for_backward(scale)
        q = torch.round(x / scale.clamp_min(1e-8) + zero_point).clamp_(qmin, qmax)
        return (q - zero_point) * scale

    @staticmethod
    def backward(ctx, grad_out):  # noqa: ANN001
        # STE: gradient w.r.t. x is identity. Scale/zero_point treated as buffers (no grad here);
        # if they are learnable, use a separate learnable wrapper.
        return grad_out, None, None, None, None


class FakeQuantize(nn.Module):
    """Drop-in fake-quantizer module.

    Parameters
    ----------
    bits:
        Bit-width (typically ``8``).
    symmetric:
        Use symmetric quantization (no zero-point).
    granularity:
        ``"per_tensor"``, ``"per_channel"`` (channel = ``dim=0``), or
        ``"per_token"`` (channel = ``dim=-2``, for activations).
    ch_axis:
        Override the channel axis. ``None`` -> inferred from ``granularity``.
    learnable:
        Make ``scale`` a learnable parameter (LSQ-style).
    """

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
        granularity: Literal["per_tensor", "per_channel", "per_token"] = "per_tensor",
        ch_axis: Optional[int] = None,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.granularity = granularity
        self.qmin, self.qmax = _q_range(bits, symmetric)
        if ch_axis is None:
            ch_axis = 0 if granularity == "per_channel" else (-2 if granularity == "per_token" else -1)
        self.ch_axis = ch_axis

        self.register_buffer("initialized", torch.tensor(False), persistent=False)
        if learnable:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("scale", torch.ones(1))
        self.register_buffer("zero_point", torch.zeros(1, dtype=torch.float32))
        self._learnable = learnable

        # Calibration accumulators (populated by `observe` -> `compute_qparams`).
        self.register_buffer("running_min", torch.tensor(float("inf")), persistent=False)
        self.register_buffer("running_max", torch.tensor(float("-inf")), persistent=False)
        self._calibrating = False
        self._observed: List[torch.Tensor] = []
        self._calib_method: Literal["minmax", "percentile", "mse"] = "percentile"
        self._calib_percentile: float = 99.99

    # ------------------------------------------------------------ calibration
    def start_calibration(
        self,
        method: Literal["minmax", "percentile", "mse"] = "percentile",
        percentile: float = 99.99,
    ) -> None:
        self._calibrating = True
        self._observed = []
        self._calib_method = method
        self._calib_percentile = percentile

    def finish_calibration(self) -> None:
        if not self._calibrating:
            return
        if not self._observed:
            _logger.warning("FakeQuantize.finish_calibration called with no observations.")
            self._calibrating = False
            return

        x = torch.cat([o.flatten(0, -2) if o.dim() >= 2 else o.unsqueeze(0) for o in self._observed], dim=0)
        if self.granularity == "per_channel":
            x_ch = x.transpose(0, self.ch_axis % x.dim()).reshape(x.shape[self.ch_axis], -1)
            scales, zps = self._fit_per_row(x_ch)
        else:
            scales, zps = self._fit_per_row(x.reshape(1, -1))

        # Replace buffer/parameter with the right shape.
        if self._learnable:
            with torch.no_grad():
                self.scale = nn.Parameter(scales)
        else:
            self.scale = scales
        self.zero_point = zps
        self.initialized.fill_(True)
        self._observed = []
        self._calibrating = False

    def _fit_per_row(self, mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``mat`` is ``[C, N]``; returns ``([C], [C])`` (scale, zero_point)."""
        if self._calib_method == "minmax":
            lo = mat.min(dim=1).values
            hi = mat.max(dim=1).values
        elif self._calib_method == "percentile":
            p = self._calib_percentile
            lo = torch.quantile(mat.float(), q=(100 - p) / 100, dim=1)
            hi = torch.quantile(mat.float(), q=p / 100, dim=1)
        elif self._calib_method == "mse":
            # Simple grid search over a few percentiles, pick the one with best MSE.
            best_lo = mat.min(dim=1).values
            best_hi = mat.max(dim=1).values
            best_err = torch.full_like(best_lo, float("inf"))
            for p in (99.0, 99.5, 99.9, 99.99, 99.999):
                lo = torch.quantile(mat.float(), q=(100 - p) / 100, dim=1)
                hi = torch.quantile(mat.float(), q=p / 100, dim=1)
                scale, zp = self._compute_scale_zp(lo, hi)
                q = torch.round(mat / scale.unsqueeze(1).clamp_min(1e-8) + zp.unsqueeze(1)).clamp(self.qmin, self.qmax)
                dq = (q - zp.unsqueeze(1)) * scale.unsqueeze(1)
                err = (mat - dq).pow(2).mean(dim=1)
                better = err < best_err
                best_lo = torch.where(better, lo, best_lo)
                best_hi = torch.where(better, hi, best_hi)
                best_err = torch.where(better, err, best_err)
            lo, hi = best_lo, best_hi
        else:
            raise ValueError(self._calib_method)

        return self._compute_scale_zp(lo, hi)

    def _compute_scale_zp(self, lo: torch.Tensor, hi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.symmetric:
            absmax = torch.max(lo.abs(), hi.abs()).clamp_min(1e-8)
            scale = absmax / self.qmax
            zp = torch.zeros_like(scale)
        else:
            scale = (hi - lo).clamp_min(1e-8) / (self.qmax - self.qmin)
            zp = (self.qmin - torch.round(lo / scale)).clamp(self.qmin, self.qmax)
        return scale, zp

    # ------------------------------------------------------------ forward
    def _broadcast(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.numel() == 1:
            return t
        shape = [1] * x.dim()
        shape[self.ch_axis] = -1
        return t.view(*shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._calibrating:
            # Record but pass through.
            self._observed.append(x.detach().to("cpu"))
            return x
        if not bool(self.initialized.item()):
            # Uninitialised — fall back to identity (e.g. during model surgery).
            return x
        s = self._broadcast(x, self.scale.to(x.device))
        z = self._broadcast(x, self.zero_point.to(x.device))
        return _STEQuantize.apply(x, s, z, self.qmin, self.qmax)


@dataclass
class QuantSpec:
    """Bundle of quantizer hyperparameters."""

    weight_bits: int = 8
    activation_bits: int = 8
    weight_granularity: Literal["per_tensor", "per_channel"] = "per_channel"
    activation_granularity: Literal["per_tensor", "per_token"] = "per_tensor"
    weight_symmetric: bool = True
    activation_symmetric: bool = False
    learnable_weight_scale: bool = False


class QuantizedLinear(nn.Module):
    """``nn.Linear`` wrapper with weight + input fake-quantization.

    Drop-in replacement: same ``forward`` semantics, same parameters; just adds
    fake-quant nodes before the matmul. KV-cache projections (``k_proj``,
    ``v_proj``) can be marked via ``quantize_output`` to also wrap the output.
    """

    def __init__(self, linear: nn.Linear, spec: QuantSpec, quantize_output: bool = False) -> None:
        super().__init__()
        self.linear = linear
        self.spec = spec
        self.weight_quantizer = FakeQuantize(
            bits=spec.weight_bits,
            symmetric=spec.weight_symmetric,
            granularity=spec.weight_granularity,
            learnable=spec.learnable_weight_scale,
        )
        self.act_quantizer = FakeQuantize(
            bits=spec.activation_bits,
            symmetric=spec.activation_symmetric,
            granularity=spec.activation_granularity,
        )
        self.output_quantizer: Optional[FakeQuantize]
        if quantize_output:
            self.output_quantizer = FakeQuantize(
                bits=spec.activation_bits,
                symmetric=spec.activation_symmetric,
                granularity=spec.activation_granularity,
            )
        else:
            self.output_quantizer = None

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialise weight quantizer lazily from the weight itself
        # (calibration over weights is just observing the weight tensor).
        if not bool(self.weight_quantizer.initialized.item()) and not self.weight_quantizer._calibrating:
            self.weight_quantizer.start_calibration(method="minmax")
            _ = self.weight_quantizer(self.linear.weight.detach())
            self.weight_quantizer.finish_calibration()

        w_q = self.weight_quantizer(self.linear.weight)
        x_q = self.act_quantizer(x)
        out = F.linear(x_q, w_q, self.linear.bias)
        if self.output_quantizer is not None:
            out = self.output_quantizer(out)
        return out


def prepare_qat_model(
    model: nn.Module,
    spec: QuantSpec,
    skip_layers: Optional[List[str]] = None,
    quantize_kv_cache: bool = True,
) -> nn.Module:
    """Walk every decoder block, replace target Linears with ``QuantizedLinear``.

    Returns the same model, modified in place (also returned for chaining).
    """
    skip_layers = skip_layers or []
    layers = get_decoder_layers(model)
    n_replaced = 0
    for layer_idx, block in enumerate(layers):
        for dotted, linear in get_linear_layers_in_block(block).items():
            if any(s in dotted for s in skip_layers):
                continue
            quantize_output = quantize_kv_cache and dotted in {
                "self_attn.k_proj",
                "self_attn.v_proj",
            }
            wrapped = QuantizedLinear(linear, spec=spec, quantize_output=quantize_output)
            wrapped.to(linear.weight.device).to(linear.weight.dtype)
            # Re-mount onto the parent module.
            parent = block
            parts = dotted.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], wrapped)
            n_replaced += 1

    _logger.info(
        f"Inserted FakeQuant into {n_replaced} linears "
        f"(weight={spec.weight_bits}b/{spec.weight_granularity}, "
        f"act={spec.activation_bits}b/{spec.activation_granularity}, "
        f"kv_cache={quantize_kv_cache})"
    )
    # Make sure we don't quantize layers in skip_layers (e.g. lm_head, embeddings).
    for skip in skip_layers:
        if hasattr(model, skip):
            _logger.info(f"Skipping quantization for top-level module: {skip}")
    return model
