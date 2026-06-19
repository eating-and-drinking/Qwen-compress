# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Unit tests for MOT-FD distillation loss components."""

from __future__ import annotations

import pytest
import torch

from qwen_compress.distill.losses import (
    DistillationLoss,
    HiddenCosineLoss,
    KDLoss,
    OptimalTransportAlignLoss,
    AttentionDistillationLoss,
    sinkhorn,
)


@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(0)


# ============================================================================
# Sinkhorn OT
# ============================================================================


class TestSinkhorn:
    def test_output_shape(self):
        C = torch.rand(5, 3)
        gamma = sinkhorn(C, eps=0.1, num_iters=20)
        assert gamma.shape == (5, 3)
        assert torch.all(gamma >= 0)
        assert gamma.sum().item() == pytest.approx(1.0, abs=1e-4)

    def test_diagonal_recovery(self):
        """With a diagonal cost matrix, transport should be diagonal."""
        C = torch.eye(4) * -10  # lower cost on diagonal
        gamma = sinkhorn(C, eps=1.0, num_iters=100)
        diag_sum = gamma.diag().sum()
        assert diag_sum > 0.9  # most mass on diagonal


# ============================================================================
# KD Loss
# ============================================================================


class TestKDLoss:
    def test_zero_when_identical(self):
        loss = KDLoss(temperature=2.0)
        logits = torch.randn(2, 4, 16)
        out = loss(logits, logits.clone())
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_distinct(self):
        loss = KDLoss(temperature=2.0)
        s = torch.randn(2, 4, 16)
        t = torch.randn(2, 4, 16)
        out = loss(s, t)
        assert out.item() > 0.0

    def test_respects_mask(self):
        loss = KDLoss(temperature=2.0)
        s = torch.randn(2, 4, 16)
        t = torch.randn(2, 4, 16)
        full_mask = torch.ones(2, 4, dtype=torch.bool)
        partial = full_mask.clone()
        partial[:, 2:] = False
        v_full = loss(s, t, valid_mask=full_mask)
        v_part = loss(s, t, valid_mask=partial)
        assert v_full.item() != v_part.item()

    def test_invalid_temperature(self):
        with pytest.raises(ValueError):
            KDLoss(temperature=0.0)


# ============================================================================
# Hidden Cosine Loss (legacy)
# ============================================================================


class TestHiddenCosineLoss:
    def test_identity_when_same_dim(self):
        loss_fn = HiddenCosineLoss(student_dim=8, teacher_dim=8)
        h = torch.randn(2, 4, 8)
        out = loss_fn(h, h)
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_projects_dim_mismatch(self):
        loss_fn = HiddenCosineLoss(student_dim=8, teacher_dim=16)
        s = torch.randn(2, 4, 8)
        t = torch.randn(2, 4, 16)
        out = loss_fn(s, t)
        assert out.dim() == 0
        assert torch.isfinite(out)


# ============================================================================
# OT Alignment Loss
# ============================================================================


class TestOTAlignLoss:
    def test_returns_components(self):
        loss_fn = OptimalTransportAlignLoss(
            ot_temperature=0.1,
            sinkhorn_iters=20,
        )
        s_hidden = [torch.randn(2, 6, 8) for _ in range(8)]  # 8 student layers
        t_groups = torch.randn(4, 8)  # 4 teacher groups
        valid_mask = torch.ones(2, 6, dtype=torch.bool)

        ot_loss, mono_loss, gamma, expected_pos = loss_fn(
            s_hidden, t_groups, valid_mask,
        )
        assert ot_loss.item() > 0
        assert mono_loss.item() >= 0
        assert gamma.shape == (8, 4)
        assert expected_pos.shape == (8,)

    def test_adaptive_temperature(self):
        """Test adaptive OT temperature."""
        loss_fn = OptimalTransportAlignLoss(
            ot_temperature=0.1,
            sinkhorn_iters=20,
            adaptive_temperature=True,
            adaptive_temp_min=0.05,
            adaptive_temp_max=0.5,
        )
        s_hidden = [torch.randn(2, 6, 8) for _ in range(8)]
        t_groups = torch.randn(4, 8) * 10.0  # Make alignment harder
        valid_mask = torch.ones(2, 6, dtype=torch.bool)

        ot_loss, mono_loss, gamma, expected_pos = loss_fn(
            s_hidden, t_groups, valid_mask,
        )
        assert ot_loss.item() > 0
        assert torch.isfinite(ot_loss)

    def test_mono_zero_when_monotonic(self):
        """When student reps are naturally ordered to match groups, mono loss ~ 0."""
        loss_fn = OptimalTransportAlignLoss(ot_temperature=0.05, sinkhorn_iters=50)
        # Create student reps that are monotonically aligned to groups:
        # early student layers close to early groups, late to late groups
        group_centers = torch.stack([
            torch.ones(8) * i * 10.0 for i in range(4)
        ])  # 4 groups, D=8
        s_hidden = []
        for i in range(12):
            # Student layers 0-2 → near group 0, 3-5 → group 1, etc.
            g_idx = i // 3
            h = torch.randn(2, 6, 8) * 0.01 + group_centers[g_idx].unsqueeze(0).unsqueeze(0)
            s_hidden.append(h)
        valid_mask = torch.ones(2, 6, dtype=torch.bool)

        _, mono_loss, _, _ = loss_fn(s_hidden, group_centers, valid_mask)
        # Expected positions should be roughly increasing
        assert mono_loss.item() < 0.5  # low penalty


# ============================================================================
# Attention Distillation Loss
# ============================================================================


class TestAttentionDistillationLoss:
    def test_kl_strategy(self):
        loss_fn = AttentionDistillationLoss(strategy="kl")
        s_attn = [torch.softmax(torch.randn(2, 4, 6, 6), dim=-1)]
        t_attn = [torch.softmax(torch.randn(2, 4, 6, 6), dim=-1)]
        loss = loss_fn(s_attn, t_attn)
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_cosine_strategy(self):
        loss_fn = AttentionDistillationLoss(strategy="cosine")
        s_attn = [torch.randn(2, 4, 6, 6)]
        t_attn = [torch.randn(2, 4, 6, 6)]
        loss = loss_fn(s_attn, t_attn)
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_mse_strategy(self):
        loss_fn = AttentionDistillationLoss(strategy="mse")
        s_attn = [torch.randn(2, 4, 6, 6)]
        t_attn = [torch.randn(2, 4, 6, 6)]
        loss = loss_fn(s_attn, t_attn)
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_ot_strategy(self):
        """Test OT-based attention distillation."""
        loss_fn = AttentionDistillationLoss(strategy="ot", ot_temperature=0.1)
        s_attn = [torch.softmax(torch.randn(2, 4, 6, 6), dim=-1)]  # 4 student heads
        t_attn = [torch.softmax(torch.randn(2, 8, 6, 6), dim=-1)]  # 8 teacher heads
        loss = loss_fn(s_attn, t_attn)
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_head_mismatch(self):
        """Test handling of different number of heads."""
        loss_fn = AttentionDistillationLoss(strategy="kl")
        s_attn = [torch.softmax(torch.randn(2, 4, 6, 6), dim=-1)]  # 4 heads
        t_attn = [torch.softmax(torch.randn(2, 8, 6, 6), dim=-1)]  # 8 heads
        loss = loss_fn(s_attn, t_attn)
        assert torch.isfinite(loss)


# ============================================================================
# Composite Distillation Loss
# ============================================================================


class TestDistillationLossMOTFD:
    def test_mot_fd_mode(self):
        """MOT-FD mode with teacher_group_reps."""
        teacher_group_reps = torch.randn(8, 16)
        loss_fn = DistillationLoss(
            student_hidden_size=16,
            teacher_hidden_size=16,
            teacher_group_reps=teacher_group_reps,
            alpha_ce=1.0,
            beta_kd=1.0,
            lambda_ot=1.0,
            lambda_mono=0.1,
            ot_temperature=0.1,
            sinkhorn_iters=20,
        )
        s_logits = torch.randn(2, 6, 100, requires_grad=True)
        t_logits = torch.randn(2, 6, 100)
        labels = torch.randint(0, 100, (2, 6))
        labels[:, :2] = -100
        s_hidden = [torch.randn(2, 6, 16) for _ in range(6)]

        out = loss_fn(
            student_logits=s_logits,
            teacher_logits=t_logits,
            labels=labels,
            student_hidden_states=s_hidden,
            teacher_hidden_states=None,
        )
        assert "ce" in out.breakdown
        assert "kd" in out.breakdown
        assert "ot" in out.breakdown
        assert "mono" in out.breakdown
        assert "total" in out.breakdown
        assert torch.isfinite(out.total)
        out.total.backward()

    def test_mot_fd_dim_mismatch(self):
        """MOT-FD with teacher_hidden_size != student_hidden_size uses ot_projector."""
        teacher_group_reps = torch.randn(8, 32)  # teacher D=32
        loss_fn = DistillationLoss(
            student_hidden_size=16,
            teacher_hidden_size=32,
            teacher_group_reps=teacher_group_reps,
            alpha_ce=1.0,
            beta_kd=1.0,
            lambda_ot=1.0,
            lambda_mono=0.1,
            ot_temperature=0.1,
            sinkhorn_iters=20,
        )
        assert isinstance(loss_fn.ot_projector, torch.nn.Linear)
        s_logits = torch.randn(2, 6, 100, requires_grad=True)
        t_logits = torch.randn(2, 6, 100)
        labels = torch.randint(0, 100, (2, 6))
        labels[:, :2] = -100
        s_hidden = [torch.randn(2, 6, 16) for _ in range(6)]  # student D=16

        out = loss_fn(
            student_logits=s_logits,
            teacher_logits=t_logits,
            labels=labels,
            student_hidden_states=s_hidden,
            teacher_hidden_states=None,
        )
        assert torch.isfinite(out.total)
        out.total.backward()

    def test_legacy_qat_mode(self):
        """Legacy QAT mode without teacher_group_reps (backward compat)."""
        loss_fn = DistillationLoss(
            student_hidden_size=8,
            teacher_hidden_size=8,
            teacher_group_reps=None,
            alpha_ce=1.0,
            beta_kd=1.0,
            gamma_hidden=1.0,
        )
        s_logits = torch.randn(1, 4, 50, requires_grad=True)
        t_logits = torch.randn(1, 4, 50)
        labels = torch.randint(0, 50, (1, 4))
        labels[:, :1] = -100
        s_hidden = [torch.randn(1, 4, 8)]
        t_hidden = [torch.randn(1, 4, 8)]

        out = loss_fn(
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
        out.total.backward()
