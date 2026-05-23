# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Distillation loss components.

Designed to be composable: ``DistillationLoss.forward`` returns both the
aggregate loss and a per-component breakdown for logging/diagnostics.

Numerical notes
---------------
* KD divergence uses ``F.kl_div`` with ``log_softmax(student)`` and
  ``softmax(teacher)`` and ``reduction='batchmean'``. The temperature-square
  rescaling follows Hinton et al. (2015).
* Hidden-state MSE uses normalised tensors so different hidden sizes
  (teacher 5120 vs. student 2048) don't blow up the loss scale. A learned
  linear projector handles the dimensionality mismatch.
* The ignore-mask carries through to all components: positions where
  ``labels == -100`` are excluded from every term (including KD/hidden MSE).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LossOutput:
    """Container returned by :class:`DistillationLoss`."""

    total: torch.Tensor
    breakdown: Dict[str, torch.Tensor]

    def to_log_dict(self) -> Dict[str, float]:
        return {k: float(v.detach()) for k, v in self.breakdown.items()}


class KDDivergenceLoss(nn.Module):
    """Hinton-style soft-target KL divergence with temperature ``T``."""

    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute KD loss.

        Parameters
        ----------
        student_logits, teacher_logits:
            ``[B, T, V]`` tensors. ``V`` must match.
        valid_mask:
            ``[B, T]`` boolean tensor; ``True`` keeps the position. If ``None``,
            all positions count.
        """
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"Logit shape mismatch: student {student_logits.shape} vs "
                f"teacher {teacher_logits.shape}"
            )
        T = self.temperature
        s_log_probs = F.log_softmax(student_logits / T, dim=-1)
        t_probs = F.softmax(teacher_logits / T, dim=-1)

        # Per-token KL: sum over vocab dim.
        per_token = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)  # [B, T]

        if valid_mask is not None:
            per_token = per_token * valid_mask.to(per_token.dtype)
            denom = valid_mask.sum().clamp_min(1).to(per_token.dtype)
            loss = per_token.sum() / denom
        else:
            loss = per_token.mean()

        # Temperature^2 rescaling so gradient magnitudes are T-invariant.
        return loss * (T * T)


class HiddenStateMSELoss(nn.Module):
    """MSE between (projected) student hidden states and teacher hidden states.

    A linear projector handles hidden-size mismatch. Both sides are RMS-normalised
    before comparison to keep the loss scale-invariant.
    """

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        if student_dim != teacher_dim:
            self.projector: nn.Module = nn.Linear(student_dim, teacher_dim, bias=False)
        else:
            self.projector = nn.Identity()
        self.eps = 1e-6

    @staticmethod
    def _rms_norm(x: torch.Tensor) -> torch.Tensor:
        return x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6)

    def forward(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        s = self.projector(student_hidden)
        s = self._rms_norm(s)
        t = self._rms_norm(teacher_hidden.detach())
        per_token = (s - t).pow(2).mean(dim=-1)  # [B, T]
        if valid_mask is not None:
            per_token = per_token * valid_mask.to(per_token.dtype)
            denom = valid_mask.sum().clamp_min(1).to(per_token.dtype)
            return per_token.sum() / denom
        return per_token.mean()


class DistillationLoss(nn.Module):
    """Composite teacher-student loss.

    ``L = alpha * CE + beta * KD + gamma * sum_g MSE(h_s^g, h_t^g)``

    Where the sum runs over group anchors. Attention-map alignment can be added
    via the ``delta`` term but defaults to ``0`` (attention export adds memory).
    """

    def __init__(
        self,
        student_hidden_size: int,
        teacher_hidden_size: int,
        num_groups: int,
        alpha_ce: float = 1.0,
        beta_kd: float = 1.0,
        gamma_hidden: float = 1.0,
        delta_attn: float = 0.0,
        kd_temperature: float = 2.0,
    ) -> None:
        super().__init__()
        self.alpha_ce = alpha_ce
        self.beta_kd = beta_kd
        self.gamma_hidden = gamma_hidden
        self.delta_attn = delta_attn

        self.kd = KDDivergenceLoss(temperature=kd_temperature)
        # One projector per group anchor.
        self.hidden_losses = nn.ModuleList(
            [HiddenStateMSELoss(student_hidden_size, teacher_hidden_size) for _ in range(num_groups)]
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_hidden_states: Sequence[torch.Tensor],
        teacher_hidden_states: Sequence[torch.Tensor],
        student_attentions: Optional[Sequence[torch.Tensor]] = None,
        teacher_attentions: Optional[Sequence[torch.Tensor]] = None,
    ) -> LossOutput:
        if len(student_hidden_states) != len(teacher_hidden_states):
            raise ValueError(
                f"Got {len(student_hidden_states)} student anchors vs "
                f"{len(teacher_hidden_states)} teacher anchors."
            )

        breakdown: Dict[str, torch.Tensor] = {}

        # 1) Cross-entropy on labels (next-token prediction).
        # `labels` is the standard causal-LM target tensor with -100 on prompt positions.
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        breakdown["ce"] = ce_loss

        # `valid_mask` for KD/hidden uses the same -100 ignore convention.
        valid_mask = (labels != -100).to(student_logits.device)

        # 2) KD logit divergence.
        if self.beta_kd > 0 and teacher_logits is not None:
            kd_loss = self.kd(student_logits, teacher_logits, valid_mask=valid_mask)
            breakdown["kd"] = kd_loss
        else:
            kd_loss = student_logits.new_zeros(())

        # 3) Hidden-state MSE (per group).
        if self.gamma_hidden > 0 and len(student_hidden_states) > 0:
            hidden_terms = []
            for module, s_h, t_h in zip(
                self.hidden_losses, student_hidden_states, teacher_hidden_states
            ):
                hidden_terms.append(module(s_h, t_h, valid_mask=valid_mask))
            hidden_loss = torch.stack(hidden_terms).mean()
            breakdown["hidden"] = hidden_loss
        else:
            hidden_loss = student_logits.new_zeros(())

        # 4) (Optional) Attention-map MSE.
        if (
            self.delta_attn > 0
            and student_attentions is not None
            and teacher_attentions is not None
            and len(student_attentions) > 0
        ):
            attn_terms = []
            for s_a, t_a in zip(student_attentions, teacher_attentions):
                # Average over heads when head counts differ (e.g. GQA).
                s_avg = s_a.mean(dim=1)  # [B, T, T]
                t_avg = t_a.detach().mean(dim=1)
                attn_terms.append(F.mse_loss(s_avg, t_avg))
            attn_loss = torch.stack(attn_terms).mean()
            breakdown["attn"] = attn_loss
        else:
            attn_loss = student_logits.new_zeros(())

        total = (
            self.alpha_ce * ce_loss
            + self.beta_kd * kd_loss
            + self.gamma_hidden * hidden_loss
            + self.delta_attn * attn_loss
        )
        breakdown["total"] = total
        return LossOutput(total=total, breakdown=breakdown)
