from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from qwen_compress.utils.logging import get_logger

_logger = get_logger(__name__)


# ============================================================================
# Utilities
# ============================================================================


def masked_mean(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Mean over valid (non-masked) positions.

    x: [B, T] or [B, T, D]
    mask: [B, T] with True=valid
    """
    if mask is None:
        return x.mean()
    mask = mask.to(x.dtype)
    if x.dim() == 3:
        mask = mask.unsqueeze(-1)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def whiten(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Feature whitening across hidden dimension."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)


# ============================================================================
# Sinkhorn Optimal Transport
# ============================================================================


def sinkhorn(
    C: torch.Tensor,
    eps: float = 0.1,
    num_iters: int = 50,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Entropy-regularized Optimal Transport via Sinkhorn algorithm.

    γ* = argmin_{γ∈Π(a,b)} Σ_{l,k} γ_{l,k} C_{l,k} + ε Σ γ_{l,k} log γ_{l,k}

    Parameters
    ----------
    C:
        Cost matrix of shape ``[N, M]`` (student layers × teacher groups).
    eps:
        Entropy regularization strength (smaller → harder assignment).
    num_iters:
        Number of Sinkhorn iterations.
    a:
        Source marginals ``[N]`` (default: uniform).
    b:
        Target marginals ``[M]`` (default: uniform).

    Returns
    -------
    Transport plan γ of shape ``[N, M]``.
    """
    N, M = C.shape
    device, dtype = C.device, C.dtype

    if a is None:
        a = torch.ones(N, device=device, dtype=dtype) / N
    if b is None:
        b = torch.ones(M, device=device, dtype=dtype) / M

    # Gibbs kernel: K = exp(-C / ε)
    K = torch.exp(-C / eps)

    # Sinkhorn iterations
    v = torch.ones(M, device=device, dtype=dtype) / M
    for _ in range(num_iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)

    # Transport plan
    gamma = u.unsqueeze(1) * K * v.unsqueeze(0)
    return gamma


@dataclass
class LossOutput:
    total: torch.Tensor
    breakdown: Dict[str, torch.Tensor]

    def to_log_dict(self) -> Dict[str, float]:
        return {k: float(v.detach()) for k, v in self.breakdown.items()}


# ============================================================================
# KD Loss (shared between MOT-FD and QAT modes)
# ============================================================================


class KDLoss(nn.Module):
    """KL-divergence knowledge distillation on logits."""

    def __init__(self, temperature: float = 2.0):
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
        T = self.temperature
        s_log_probs = F.log_softmax(student_logits / T, dim=-1)
        t_probs = F.softmax(teacher_logits.detach() / T, dim=-1)

        per_token = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)
        return masked_mean(per_token, valid_mask) * (T * T)


# ============================================================================
# Hidden State Cosine Loss (for QAT legacy mode)
# ============================================================================


class HiddenCosineLoss(nn.Module):
    """Lightweight semantic alignment via cosine distance.

    Teacher → student projection (NOT student → teacher).
    """

    def __init__(
        self,
        teacher_dim: int,
        student_dim: int,
        whiten_features: bool = True,
    ):
        super().__init__()
        self.whiten_features = whiten_features
        if teacher_dim != student_dim:
            self.projector = nn.Linear(teacher_dim, student_dim, bias=False)
        else:
            self.projector = nn.Identity()

    def forward(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t = self.projector(teacher_hidden.detach())
        s = student_hidden
        if self.whiten_features:
            s = whiten(s)
            t = whiten(t)
        per_token = 1.0 - F.cosine_similarity(s, t, dim=-1)
        return masked_mean(per_token, valid_mask)


# ============================================================================
# MOT-FD: Optimal Transport Alignment Loss
# ============================================================================


class OptimalTransportAlignLoss(nn.Module):
    """Optimal Transport alignment between student layers and teacher groups.

    For each batch:
    1. Compute layer-wise mean representations from student hidden states.
    2. Build cost matrix C_{l,k} = ||h_l^S - g_k^T||^2.
    3. Solve transport plan γ via Sinkhorn.
    4. Compute OT loss: L_OT = Σ_{l,k} γ_{l,k} * C_{l,k}.
    5. Compute expected functional position μ_l = Σ_k k * π_{l,k}
       where π_{l,k} = softmax(-C_{l,k} / τ).
    6. Compute monotonic regularization: L_mono = Σ max(0, μ_l - μ_{l+1}).

    Returns (L_OT, L_mono, transport_plan, expected_positions).
    """

    def __init__(
        self,
        ot_temperature: float = 0.1,
        sinkhorn_iters: int = 50,
        soft_assign_temperature: float = 1.0,
    ):
        super().__init__()
        if ot_temperature <= 0:
            raise ValueError("ot_temperature must be > 0")
        self.ot_temperature = ot_temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.soft_assign_temperature = soft_assign_temperature

    def forward(
        self,
        student_hidden_states: Sequence[torch.Tensor],
        teacher_group_reps: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        student_hidden_states:
            List of tensors, each ``[B, T, D_s]``, one per student layer.
        teacher_group_reps:
            Pre-computed group representations ``[num_groups, D_t]``.
        valid_mask:
            ``[B, T]`` boolean mask (True = valid token).

        Returns
        -------
        ot_loss, mono_loss, transport_plan, expected_positions:
            - ot_loss: scalar, Σ γ·C.
            - mono_loss: scalar, monotonic penalty.
            - transport_plan: ``[L, G]`` tensor.
            - expected_positions: ``[L]`` tensor (μ_l).
        """
        num_layers = len(student_hidden_states)
        num_groups = teacher_group_reps.shape[0]
        device = student_hidden_states[0].device
        dtype = student_hidden_states[0].dtype

        # 1. Compute mean representation per student layer
        student_reps = []
        for h in student_hidden_states:
            # h: [B, T, D]
            if valid_mask is not None:
                mask = valid_mask.unsqueeze(-1).to(h.dtype)
                rep = (h * mask).sum(dim=(0, 1)) / mask.sum().clamp_min(1.0)
            else:
                rep = h.mean(dim=(0, 1))
            student_reps.append(rep)
        s_reps = torch.stack(student_reps, dim=0)  # [L, D]

        # Align teacher group reps to student device/dtype
        t_reps = teacher_group_reps.to(device=device, dtype=dtype)  # [G, D]

        # Handle dimension mismatch (teacher vs student hidden sizes).
        # This happens when teacher and student have different hidden dims.
        # We use a lightweight projection: teacher reps are projected down to
        # student dim via a learned Linear if needed.
        if s_reps.shape[-1] != t_reps.shape[-1]:
            t_dim = t_reps.shape[-1]
            s_dim = s_reps.shape[-1]
            _logger.warning(
                f"Hidden dimension mismatch in OT cost: student={s_dim}, "
                f"teacher_group={t_dim}. Using dimension truncation; consider "
                f"using a learned projector for better alignment."
            )
            min_dim = min(s_dim, t_dim)
            s_reps = s_reps[..., :min_dim]
            t_reps = t_reps[..., :min_dim]

        # 2. Cost matrix: C_{l,k} = ||s_l - g_k||^2 / D
        # [L, D] vs [G, D] → [L, G]
        s_norm = s_reps.pow(2).sum(dim=-1, keepdim=True)  # [L, 1]
        t_norm = t_reps.pow(2).sum(dim=-1).unsqueeze(0)   # [1, G]
        s_t_dot = torch.mm(s_reps, t_reps.T)              # [L, G]
        C = (s_norm + t_norm - 2 * s_t_dot) / s_reps.shape[-1]  # [L, G]

        # 3. Transport plan via Sinkhorn
        gamma = sinkhorn(
            C,
            eps=self.ot_temperature,
            num_iters=self.sinkhorn_iters,
        )  # [L, G]

        # 4. OT loss
        ot_loss = (gamma * C).sum()

        # 5. Soft assignment and expected functional position
        # π_{l,k} = softmax(-C_{l,k} / τ)
        pi = F.softmax(-C / self.soft_assign_temperature, dim=-1)  # [L, G]
        group_indices = torch.arange(num_groups, device=device, dtype=dtype)  # [G]
        expected_pos = (pi * group_indices.unsqueeze(0)).sum(dim=-1)  # [L]

        # 6. Monotonic regularization
        mono_penalty = F.relu(expected_pos[:-1] - expected_pos[1:])
        mono_loss = mono_penalty.sum()

        return ot_loss, mono_loss, gamma.detach(), expected_pos.detach()


# ============================================================================
# Main Distillation Loss (MOT-FD)
# ============================================================================


class DistillationLoss(nn.Module):
    """Composite distillation loss supporting MOT-FD and legacy QAT modes.

    MOT-FD mode (teacher_group_reps is provided):
        L = α·CE + β·KD + λ_ot·L_OT + λ_mono·L_mono

    Legacy QAT mode (teacher_group_reps is None):
        L = α·CE + β·KD + γ·hidden_cosine
    """

    def __init__(
        self,
        student_hidden_size: int,
        teacher_hidden_size: int,
        teacher_group_reps: Optional[torch.Tensor] = None,
        # Loss weights
        alpha_ce: float = 1.0,
        beta_kd: float = 1.0,
        gamma_hidden: float = 1.0,
        delta_attn: float = 0.0,
        # MOT-FD specific
        lambda_ot: float = 1.0,
        lambda_mono: float = 0.1,
        # Temperature / hyperparams
        kd_temperature: float = 2.0,
        ot_temperature: float = 0.1,
        sinkhorn_iters: int = 50,
    ):
        super().__init__()
        self.alpha_ce = alpha_ce
        self.beta_kd = beta_kd
        self.gamma_hidden = gamma_hidden
        self.delta_attn = delta_attn
        self.lambda_ot = lambda_ot
        self.lambda_mono = lambda_mono

        # KD loss (always used)
        self.kd = KDLoss(temperature=kd_temperature)

        # Hidden cosine loss (used in legacy QAT mode)
        self.hidden_loss = HiddenCosineLoss(
            teacher_dim=teacher_hidden_size,
            student_dim=student_hidden_size,
            whiten_features=True,
        )

        # OT alignment (used in MOT-FD mode when teacher_group_reps is provided)
        self.ot_loss_fn: Optional[OptimalTransportAlignLoss] = None
        if teacher_group_reps is not None:
            self.register_buffer("teacher_group_reps", teacher_group_reps)
            self.ot_loss_fn = OptimalTransportAlignLoss(
                ot_temperature=ot_temperature,
                sinkhorn_iters=sinkhorn_iters,
            )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_hidden_states: Sequence[torch.Tensor],
        teacher_hidden_states: Optional[Sequence[torch.Tensor]] = None,
        student_attentions: Optional[Sequence[torch.Tensor]] = None,
        teacher_attentions: Optional[Sequence[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """Compute composite distillation loss.

        Parameters
        ----------
        student_logits, teacher_logits:
            Logits from student and teacher ``[B, T, V]``.
        labels:
            Ground-truth token labels ``[B, T]`` (with -100 for prompt).
        student_hidden_states:
            Captured hidden states from student layers.
        teacher_hidden_states:
            Captured hidden states from teacher layers (legacy QAT mode).
        student_attentions, teacher_attentions:
            Optional attention maps (currently unused in MOT-FD).
        attention_mask:
            ``[B, T]`` attention mask (not used for loss computation).

        Returns
        -------
        LossOutput with total and breakdown dict.
        """
        breakdown: Dict[str, torch.Tensor] = {}

        # ================================================================
        # 1. CE Loss
        # ================================================================
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        breakdown["ce"] = ce_loss

        # ================================================================
        # Response-only mask
        # ================================================================
        valid_mask = labels != -100  # [B, T]

        # ================================================================
        # 2. KD Loss
        # ================================================================
        kd_loss = self.kd(student_logits, teacher_logits, valid_mask)
        breakdown["kd"] = kd_loss

        total = self.alpha_ce * ce_loss + self.beta_kd * kd_loss

        # ================================================================
        # 3. MOT-FD: OT Alignment + Monotonic Regularization
        # ================================================================
        if self.ot_loss_fn is not None and len(student_hidden_states) > 0:
            ot_loss, mono_loss, _, __ = self.ot_loss_fn(
                student_hidden_states=student_hidden_states,
                teacher_group_reps=self.teacher_group_reps,  # type: ignore[arg-type]
                valid_mask=valid_mask,
            )
            breakdown["ot"] = ot_loss
            breakdown["mono"] = mono_loss
            total = total + self.lambda_ot * ot_loss + self.lambda_mono * mono_loss
        elif teacher_hidden_states is not None:
            # ============================================================
            # Legacy QAT mode: hidden-state cosine loss
            # ============================================================
            hidden_terms: List[torch.Tensor] = []
            for s_h, t_h in zip(student_hidden_states, teacher_hidden_states):
                hidden_terms.append(self.hidden_loss(s_h, t_h, valid_mask))
            if hidden_terms:
                hidden_loss_val = torch.stack(hidden_terms).mean()
            else:
                hidden_loss_val = student_logits.new_zeros(())
            breakdown["hidden"] = hidden_loss_val
            total = total + self.gamma_hidden * hidden_loss_val

        # ================================================================
        # 4. Attention loss (deprecated, kept for config compat)
        # ================================================================
        if self.delta_attn > 0 and student_attentions is not None and teacher_attentions is not None:
            # Simple mean-attention KL for backward compat
            attn_terms: List[torch.Tensor] = []
            for s_a, t_a in zip(student_attentions, teacher_attentions):
                s = s_a.mean(dim=1)
                t = t_a.detach().mean(dim=1)
                attn_terms.append(F.kl_div(
                    F.log_softmax(s, dim=-1),
                    F.softmax(t, dim=-1),
                    reduction="batchmean",
                ))
            attn_loss = torch.stack(attn_terms).mean()
        else:
            attn_loss = student_logits.new_zeros(())
        breakdown["attn"] = attn_loss
        total = total + self.delta_attn * attn_loss

        breakdown["total"] = total
        return LossOutput(total=total, breakdown=breakdown)
