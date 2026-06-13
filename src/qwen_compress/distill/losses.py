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
# Attention Distillation Loss
# ============================================================================


class AttentionDistillationLoss(nn.Module):
    """Attention pattern distillation between student and teacher.

    Supports multiple attention distillation strategies:
    - KL divergence between attention distributions
    - Cosine similarity matching
    - Mean squared error
    - OT: Optimal Transport based head alignment
    """

    def __init__(
        self,
        strategy: str = "kl",
        normalize: bool = True,
        ot_temperature: float = 0.1,
        sinkhorn_iters: int = 50,
    ):
        super().__init__()
        self.strategy = strategy.lower()
        self.normalize = normalize
        self.ot_temperature = ot_temperature
        self.sinkhorn_iters = sinkhorn_iters
        if self.strategy not in ["kl", "cosine", "mse", "ot"]:
            raise ValueError(f"Unknown attention distillation strategy: {strategy}")

    def _extract_head_representations(
        self, attn: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract representation vector for each attention head.
        
        Parameters
        ----------
        attn: [B, H, T, T] attention weights
        
        Returns
        -------
        head_reps: [H, D] where D = T (mean attention pattern per head)
        """
        # attn: [B, H, T, T]
        if valid_mask is not None:
            # Apply mask: only consider valid positions
            mask = valid_mask.unsqueeze(1).unsqueeze(-1).to(attn.dtype)  # [B, 1, T, 1]
            attn = attn * mask
            denom = mask.sum(dim=(0, 2, 3)).clamp_min(1.0)  # [1]
        else:
            denom = attn.shape[0] * attn.shape[2]  # B * T
        
        # Mean over batch and query positions: [H, T]
        # Each head gets a "signature" vector representing its attention pattern
        head_reps = attn.sum(dim=(0, 2)) / denom  # [H, T]
        return head_reps

    def _compute_ot_attention_loss(
        self,
        s_attn: torch.Tensor,
        t_attn: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute OT-based attention distillation loss.
        
        Uses optimal transport to find the best matching between student and teacher
        attention heads, then computes the aligned attention loss.
        
        Parameters
        ----------
        s_attn: [B, H_s, T_s, T_s] student attention
        t_attn: [B, H_t, T_t, T_t] teacher attention
        
        Returns
        -------
        OT-aligned attention distillation loss
        """
        t_attn = t_attn.detach()
        
        # Handle sequence length mismatch
        min_len = min(s_attn.shape[-1], t_attn.shape[-1])
        s_attn = s_attn[..., :min_len, :min_len]
        t_attn = t_attn[..., :min_len, :min_len]
        
        # Normalize attention
        if self.normalize:
            s_attn = F.softmax(s_attn, dim=-1)
            t_attn = F.softmax(t_attn, dim=-1)
        
        # Extract head representations: [H_s, T], [H_t, T]
        s_head_reps = self._extract_head_representations(s_attn, valid_mask)  # [H_s, T]
        t_head_reps = self._extract_head_representations(t_attn, valid_mask)  # [H_t, T]
        
        num_s_heads = s_head_reps.shape[0]
        num_t_heads = t_head_reps.shape[0]
        
        # Build cost matrix between student and teacher heads
        # C_{i,j} = ||s_head_i - t_head_j||^2
        s_norm = s_head_reps.pow(2).sum(dim=-1, keepdim=True)  # [H_s, 1]
        t_norm = t_head_reps.pow(2).sum(dim=-1).unsqueeze(0)   # [1, H_t]
        s_t_dot = torch.mm(s_head_reps, t_head_reps.T)         # [H_s, H_t]
        C = (s_norm + t_norm - 2 * s_t_dot) / s_head_reps.shape[-1]  # [H_s, H_t]
        
        # Solve OT to find optimal head matching
        gamma = sinkhorn(
            C,
            eps=self.ot_temperature,
            num_iters=self.sinkhorn_iters,
        )  # [H_s, H_t]
        
        # Compute aligned attention loss
        # For each student head i, it learns from teacher heads weighted by gamma[i, :]
        # Loss = Σ_{i,j} gamma[i,j] * KL(s_attn[i] || t_attn[j])
        
        # Reshape for batch computation
        # s_attn: [B, H_s, T, T] -> [B, H_s, T*T]
        # t_attn: [B, H_t, T, T] -> [B, H_t, T*T]
        s_flat = s_attn.flatten(-2)  # [B, H_s, T*T]
        t_flat = t_attn.flatten(-2)  # [B, H_t, T*T]
        
        # Compute pairwise KL divergence: [B, H_s, H_t]
        # KL(s_i || t_j) = Σ s_i * log(s_i / t_j)
        s_log = torch.log(s_flat + 1e-12)  # [B, H_s, T*T]
        t_log = torch.log(t_flat + 1e-12)  # [B, H_t, T*T]
        
        # KL(s_i || t_j) = Σ s_i * (log_s_i - log_t_j)
        # = Σ s_i * log_s_i - Σ s_i * log_t_j
        s_entropy = (s_flat * s_log).sum(dim=-1)  # [B, H_s]
        cross_entropy = torch.bmm(
            s_flat,  # [B, H_s, T*T]
            t_log.transpose(1, 2),  # [B, T*T, H_t]
        )  # [B, H_s, H_t]
        kl_matrix = s_entropy.unsqueeze(-1) - cross_entropy  # [B, H_s, H_t]
        
        # Weight by OT plan and sum
        # Loss = Σ_{i,j} gamma[i,j] * KL[i,j]
        gamma_expanded = gamma.unsqueeze(0)  # [1, H_s, H_t]
        loss = (gamma_expanded * kl_matrix).sum(dim=(1, 2)).mean()  # scalar
        
        return loss

    def forward(
        self,
        student_attentions: Sequence[torch.Tensor],
        teacher_attentions: Sequence[torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        student_attentions:
            List of attention maps, each ``[B, num_heads, T, T]``.
        teacher_attentions:
            List of attention maps, each ``[B, num_heads, T, T]``.
        valid_mask:
            ``[B, T]`` boolean mask (True = valid token).

        Returns
        -------
        attention distillation loss.
        """
        losses = []
        
        for s_attn, t_attn in zip(student_attentions, teacher_attentions):
            # s_attn: [B, H, T, T]
            # t_attn: [B, H, T, T]
            
            t_attn = t_attn.detach()
            
            if self.strategy == "ot":
                # Use OT-based alignment (handles head mismatch automatically)
                loss = self._compute_ot_attention_loss(s_attn, t_attn, valid_mask)
            else:
                # Traditional strategies: truncate to match sizes
                # Handle head mismatch by averaging extra heads
                if s_attn.shape[1] != t_attn.shape[1]:
                    min_heads = min(s_attn.shape[1], t_attn.shape[1])
                    s_attn = s_attn[:, :min_heads]
                    t_attn = t_attn[:, :min_heads]
                
                # Handle sequence length mismatch
                min_len = min(s_attn.shape[-1], t_attn.shape[-1])
                s_attn = s_attn[..., :min_len, :min_len]
                t_attn = t_attn[..., :min_len, :min_len]
                
                if self.normalize:
                    s_attn = F.softmax(s_attn, dim=-1)
                    t_attn = F.softmax(t_attn, dim=-1)
                
                if self.strategy == "kl":
                    loss = F.kl_div(
                        torch.log(s_attn + 1e-12),
                        t_attn,
                        reduction="none",
                    ).sum(dim=(-1, -2)).mean()
                elif self.strategy == "cosine":
                    s_flat = s_attn.flatten(-2)
                    t_flat = t_attn.flatten(-2)
                    loss = (1.0 - F.cosine_similarity(s_flat, t_flat, dim=-1)).mean()
                elif self.strategy == "mse":
                    loss = F.mse_loss(s_attn, t_attn)
            
            losses.append(loss)
        
        return torch.stack(losses).mean()


# ============================================================================
# MOT-FD: Optimal Transport Alignment Loss (Enhanced)
# ============================================================================


class OptimalTransportAlignLoss(nn.Module):
    """Optimal Transport alignment between student layers and teacher groups.

    Enhanced features:
    1. Adaptive temperature: dynamically adjusts based on alignment difficulty
    2. Bidirectional alignment: student→teacher and teacher→student
    3. Improved handling of dimension mismatches

    For each batch:
    1. Compute layer-wise mean representations from student hidden states.
    2. Build cost matrix C_{l,k} = ||h_l^S - g_k^T||^2.
    3. Solve transport plan γ via Sinkhorn (with optional adaptive temperature).
    4. Compute OT loss: L_OT = Σ_{l,k} γ_{l,k} * C_{l,k}.
    5. Optional backward OT: L_OT_backward = Σ_{k,l} γ_{k,l} * C_{k,l}.
    6. Compute expected functional position μ_l = Σ_k k * π_{l,k}.
    7. Compute monotonic regularization: L_mono = Σ max(0, μ_l - μ_{l+1}).

    Returns (L_OT, L_OT_backward, L_mono, transport_plan, expected_positions).
    """

    def __init__(
        self,
        ot_temperature: float = 0.1,
        sinkhorn_iters: int = 50,
        soft_assign_temperature: float = 1.0,
        adaptive_temperature: bool = False,
        adaptive_temp_min: float = 0.05,
        adaptive_temp_max: float = 0.5,
        adaptive_temp_scale: float = 1.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        if ot_temperature <= 0:
            raise ValueError("ot_temperature must be > 0")
        self.base_temperature = ot_temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.soft_assign_temperature = soft_assign_temperature
        self.adaptive_temperature = adaptive_temperature
        self.adaptive_temp_min = adaptive_temp_min
        self.adaptive_temp_max = adaptive_temp_max
        self.adaptive_temp_scale = adaptive_temp_scale
        self.bidirectional = bidirectional

    def _compute_adaptive_temperature(self, cost_matrix: torch.Tensor) -> float:
        """Compute adaptive temperature based on alignment difficulty.
        
        Higher cost (harder alignment) → higher temperature (softer assignment)
        Lower cost (easier alignment) → lower temperature (harder assignment)
        """
        mean_cost = cost_matrix.mean().item()
        adaptive_temp = self.base_temperature * (1 + self.adaptive_temp_scale * mean_cost)
        return float(torch.clamp(torch.tensor(adaptive_temp), 
                                min=self.adaptive_temp_min, 
                                max=self.adaptive_temp_max))

    def forward(
        self,
        student_hidden_states: Sequence[torch.Tensor],
        teacher_group_reps: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        ot_loss, ot_backward_loss, mono_loss, transport_plan, expected_positions:
            - ot_loss: scalar, forward OT loss (student→teacher).
            - ot_backward_loss: scalar, backward OT loss (teacher→student).
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

        # Handle dimension mismatch (teacher vs student hidden sizes)
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

        # 3. Determine OT temperature
        if self.adaptive_temperature:
            ot_temp = self._compute_adaptive_temperature(C)
        else:
            ot_temp = self.base_temperature

        # 4. Forward transport plan (student→teacher) via Sinkhorn
        gamma_st = sinkhorn(
            C,
            eps=ot_temp,
            num_iters=self.sinkhorn_iters,
        )  # [L, G]

        # 5. Forward OT loss
        ot_loss = (gamma_st * C).sum()

        # 6. Backward OT loss (teacher→student)
        ot_backward_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if self.bidirectional:
            # Cost matrix for backward alignment: C_{k,l} = ||g_k - s_l||^2 / D
            # This is just C^T
            C_ts = C.T  # [G, L]
            gamma_ts = sinkhorn(
                C_ts,
                eps=ot_temp,
                num_iters=self.sinkhorn_iters,
            )  # [G, L]
            ot_backward_loss = (gamma_ts * C_ts).sum()

        # 7. Soft assignment and expected functional position
        # π_{l,k} = softmax(-C_{l,k} / τ)
        pi = F.softmax(-C / self.soft_assign_temperature, dim=-1)  # [L, G]
        group_indices = torch.arange(num_groups, device=device, dtype=dtype)  # [G]
        expected_pos = (pi * group_indices.unsqueeze(0)).sum(dim=-1)  # [L]

        # 8. Monotonic regularization
        mono_penalty = F.relu(expected_pos[:-1] - expected_pos[1:])
        mono_loss = mono_penalty.sum()

        return ot_loss, ot_backward_loss, mono_loss, gamma_st.detach(), expected_pos.detach()


# ============================================================================
# Main Distillation Loss (MOT-FD Enhanced)
# ============================================================================


class DistillationLoss(nn.Module):
    """Composite distillation loss supporting MOT-FD and legacy QAT modes.

    Enhanced MOT-FD mode (teacher_group_reps is provided):
        L = α·CE + β·KD + λ_ot·L_OT + λ_backward·L_OT_backward + λ_mono·L_mono + δ·L_attn

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
        lambda_ot_backward: float = 0.5,
        # Temperature / hyperparams
        kd_temperature: float = 2.0,
        ot_temperature: float = 0.1,
        sinkhorn_iters: int = 50,
        # Adaptive OT temperature
        adaptive_ot_temp: bool = False,
        adaptive_temp_min: float = 0.05,
        adaptive_temp_max: float = 0.5,
        adaptive_temp_scale: float = 1.0,
        # Attention distillation
        attn_distill_strategy: str = "kl",
        attn_ot_temperature: float = 0.1,
        attn_sinkhorn_iters: int = 50,
    ):
        super().__init__()
        self.alpha_ce = alpha_ce
        self.beta_kd = beta_kd
        self.gamma_hidden = gamma_hidden
        self.delta_attn = delta_attn
        self.lambda_ot = lambda_ot
        self.lambda_mono = lambda_mono
        self.lambda_ot_backward = lambda_ot_backward

        # KD loss (always used)
        self.kd = KDLoss(temperature=kd_temperature)

        # Hidden cosine loss (used in legacy QAT mode)
        self.hidden_loss = HiddenCosineLoss(
            teacher_dim=teacher_hidden_size,
            student_dim=student_hidden_size,
            whiten_features=True,
        )

        # Attention distillation loss
        self.attn_loss_fn = AttentionDistillationLoss(
            strategy=attn_distill_strategy,
            normalize=True,
            ot_temperature=attn_ot_temperature,
            sinkhorn_iters=attn_sinkhorn_iters,
        )

        # OT alignment (used in MOT-FD mode when teacher_group_reps is provided)
        self.ot_loss_fn: Optional[OptimalTransportAlignLoss] = None
        if teacher_group_reps is not None:
            self.register_buffer("teacher_group_reps", teacher_group_reps)
            self.ot_loss_fn = OptimalTransportAlignLoss(
                ot_temperature=ot_temperature,
                sinkhorn_iters=sinkhorn_iters,
                adaptive_temperature=adaptive_ot_temp,
                adaptive_temp_min=adaptive_temp_min,
                adaptive_temp_max=adaptive_temp_max,
                adaptive_temp_scale=adaptive_temp_scale,
                bidirectional=lambda_ot_backward > 0,
            )

    def update_teacher_group_reps(self, new_group_reps: torch.Tensor, momentum: float = 0.99):
        """Update teacher group representations (for dynamic groups feature).
        
        Parameters
        ----------
        new_group_reps:
            New group representations ``[num_groups, hidden_dim]``.
        momentum:
            Momentum factor for smooth update (0 = replace, 1 = keep).
        """
        if hasattr(self, "teacher_group_reps"):
            with torch.no_grad():
                self.teacher_group_reps.data = (
                    momentum * self.teacher_group_reps.data
                    + (1 - momentum) * new_group_reps.detach().to(self.teacher_group_reps.device)
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
            Optional attention maps for attention distillation.
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
            ot_loss, ot_backward_loss, mono_loss, _, __ = self.ot_loss_fn(
                student_hidden_states=student_hidden_states,
                teacher_group_reps=self.teacher_group_reps,  # type: ignore[arg-type]
                valid_mask=valid_mask,
            )
            breakdown["ot"] = ot_loss
            breakdown["mono"] = mono_loss
            total = total + self.lambda_ot * ot_loss + self.lambda_mono * mono_loss
            
            if self.lambda_ot_backward > 0:
                breakdown["ot_backward"] = ot_backward_loss
                total = total + self.lambda_ot_backward * ot_backward_loss
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
        # 4. Attention Distillation Loss
        # ================================================================
        if self.delta_attn > 0 and student_attentions is not None and teacher_attentions is not None:
            attn_loss = self.attn_loss_fn(
                student_attentions=student_attentions,
                teacher_attentions=teacher_attentions,
                valid_mask=valid_mask,
            )
        else:
            attn_loss = student_logits.new_zeros(())
        breakdown["attn"] = attn_loss
        total = total + self.delta_attn * attn_loss

        breakdown["total"] = total
        return LossOutput(total=total, breakdown=breakdown)
