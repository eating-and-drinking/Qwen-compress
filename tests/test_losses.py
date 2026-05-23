# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Unit tests for distillation loss components."""

from __future__ import annotations

import pytest
import torch

from qwen_compress.distill.losses import (
    DistillationLoss,
    HiddenStateMSELoss,
    KDDivergenceLoss,
)


@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(0)


class TestKDDivergence:
    def test_zero_when_identical(self):
        loss = KDDivergenceLoss(temperature=2.0)
        logits = torch.randn(2, 4, 16)
        out = loss(logits, logits.clone())
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_distinct(self):
        loss = KDDivergenceLoss(temperature=2.0)
        s = torch.randn(2, 4, 16)
        t = torch.randn(2, 4, 16)
        out = loss(s, t)
        assert out.item() > 0.0

    def test_respects_mask(self):
        loss = KDDivergenceLoss(temperature=2.0)
        s = torch.randn(2, 4, 16)
        t = torch.randn(2, 4, 16)
        full_mask = torch.ones(2, 4, dtype=torch.bool)
        partial = full_mask.clone()
        partial[:, 2:] = False
        v_full = loss(s, t, valid_mask=full_mask)
        v_part = loss(s, t, valid_mask=partial)
        # Different normalisations -> different values (usually).
        assert v_full.item() != v_part.item()

    def test_shape_mismatch_raises(self):
        loss = KDDivergenceLoss()
        with pytest.raises(ValueError):
            loss(torch.randn(2, 4, 16), torch.randn(2, 4, 32))

    def test_invalid_temperature(self):
        with pytest.raises(ValueError):
            KDDivergenceLoss(temperature=0.0)


class TestHiddenStateMSE:
    def test_identity_when_same_dim(self):
        loss = HiddenStateMSELoss(student_dim=8, teacher_dim=8)
        # Same input on both sides -> zero (within projector noise).
        # The identity projector ensures s == student_input exactly.
        h = torch.randn(2, 4, 8)
        out = loss(h, h)
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_projects_dim_mismatch(self):
        loss = HiddenStateMSELoss(student_dim=8, teacher_dim=16)
        s = torch.randn(2, 4, 8)
        t = torch.randn(2, 4, 16)
        # Should not throw; output is a finite scalar.
        out = loss(s, t)
        assert out.dim() == 0
        assert torch.isfinite(out)


class TestDistillationLoss:
    def test_composite_returns_breakdown(self):
        s_logits = torch.randn(2, 6, 100, requires_grad=True)
        t_logits = torch.randn(2, 6, 100)
        labels = torch.randint(0, 100, (2, 6))
        labels[:, :2] = -100  # prompt positions
        s_hidden = [torch.randn(2, 6, 8) for _ in range(3)]
        t_hidden = [torch.randn(2, 6, 8) for _ in range(3)]

        loss = DistillationLoss(
            student_hidden_size=8,
            teacher_hidden_size=8,
            num_groups=3,
            alpha_ce=1.0,
            beta_kd=1.0,
            gamma_hidden=1.0,
            delta_attn=0.0,
        )
        out = loss(
            student_logits=s_logits,
            teacher_logits=t_logits,
            labels=labels,
            student_hidden_states=s_hidden,
            teacher_hidden_states=t_hidden,
        )
        assert "ce" in out.breakdown
        assert "kd" in out.breakdown
        assert "hidden" in out.breakdown
        assert "total" in out.breakdown
        assert torch.isfinite(out.total)
        out.total.backward()  # should not raise

    def test_hidden_count_mismatch_raises(self):
        loss = DistillationLoss(8, 8, num_groups=2)
        with pytest.raises(ValueError):
            loss(
                student_logits=torch.randn(1, 2, 10),
                teacher_logits=torch.randn(1, 2, 10),
                labels=torch.zeros(1, 2, dtype=torch.long),
                student_hidden_states=[torch.randn(1, 2, 8)],
                teacher_hidden_states=[torch.randn(1, 2, 8), torch.randn(1, 2, 8)],
            )
