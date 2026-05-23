# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Unit tests for fake-quantize modules."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from qwen_compress.qat.fake_quant import (
    FakeQuantize,
    QuantSpec,
    QuantizedLinear,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


class TestFakeQuantize:
    def test_passthrough_before_calibration(self):
        fq = FakeQuantize(bits=8)
        x = torch.randn(2, 4, 8)
        # Pre-calibration: should pass through unchanged.
        assert torch.allclose(fq(x), x)

    def test_quantizes_after_calibration(self):
        fq = FakeQuantize(bits=8, granularity="per_tensor", symmetric=True)
        fq.start_calibration(method="minmax")
        x = torch.randn(64, 32)
        _ = fq(x)
        fq.finish_calibration()
        y = fq(x)
        # Output should be close to x but not identical (quantization noise).
        diff = (x - y).abs().max().item()
        assert 0 < diff < 0.5

    def test_per_channel(self):
        fq = FakeQuantize(bits=8, granularity="per_channel", ch_axis=0)
        fq.start_calibration(method="minmax")
        # Channels with very different scales should each get their own scale.
        x = torch.cat([torch.randn(1, 32) * 0.01, torch.randn(1, 32) * 100.0], dim=0)
        _ = fq(x)
        fq.finish_calibration()
        assert fq.scale.numel() == 2
        assert fq.scale[1] > fq.scale[0] * 100

    def test_ste_gradient(self):
        fq = FakeQuantize(bits=8, symmetric=True)
        fq.start_calibration(method="minmax")
        x = torch.randn(8, 16)
        _ = fq(x)
        fq.finish_calibration()
        x = torch.randn(8, 16, requires_grad=True)
        y = fq(x).sum()
        y.backward()
        # STE: gradient w.r.t. x should be all ones (since sum's grad is ones).
        assert torch.allclose(x.grad, torch.ones_like(x.grad))


class TestQuantizedLinear:
    def test_forward_shape(self):
        spec = QuantSpec(weight_bits=8, activation_bits=8)
        lin = nn.Linear(16, 8, bias=True)
        wrap = QuantizedLinear(lin, spec=spec)
        x = torch.randn(2, 4, 16)
        out = wrap(x)
        assert out.shape == (2, 4, 8)

    def test_close_to_fp_when_calibrated(self):
        spec = QuantSpec(weight_bits=8, activation_bits=8)
        lin = nn.Linear(32, 16, bias=False)
        wrap = QuantizedLinear(lin, spec=spec)
        # Calibrate activation quantizer.
        wrap.act_quantizer.start_calibration(method="minmax")
        x = torch.randn(64, 32)
        _ = wrap(x)
        wrap.act_quantizer.finish_calibration()

        fp_out = lin(x)
        q_out = wrap(x)
        rel = (fp_out - q_out).norm() / fp_out.norm()
        # INT8 dynamic range should yield <2% relative error on smooth Gaussian data.
        assert rel.item() < 0.05

    def test_kv_cache_output_quant(self):
        spec = QuantSpec()
        lin = nn.Linear(8, 8)
        wrap = QuantizedLinear(lin, spec=spec, quantize_output=True)
        assert wrap.output_quantizer is not None
        out = wrap(torch.randn(2, 4, 8))
        assert out.shape == (2, 4, 8)
