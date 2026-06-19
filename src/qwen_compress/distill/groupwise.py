# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Monotonic Optimal Transport Functional Distillation (MOT-FD).

Teacher functional decomposition (48 layers → 12 groups):

1. Extract layer representations z_l^T = E_{x~D}[h_l^T(x)]
2. Compute representation dynamics energy E(l)
3. Detect change points via peak detection
4. Build functional groups from breakpoints
5. Compute group representations g_k^T = mean(G_k^T)

During training, Optimal Transport aligns student layers to these group
representations with a monotonicity constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from qwen_compress.utils.logging import get_logger

_logger = get_logger(__name__)


@dataclass
class GroupAssignment:
    """Result of teacher functional decomposition.

    Attributes
    ----------
    groups:
        List of 12 functional groups, each containing teacher layer indices.
    group_representations:
        Pre-computed group centroids ``g_k^T`` of shape ``[num_groups, hidden_dim]``.
        These are fixed during training and used in the OT cost matrix.
    energy_signal:
        The computed representation dynamics energy E(l) for layers 1..L-1.
    breakpoints:
        Indices of the 11 detected change-points (b_1, ..., b_11).
    """

    groups: List[List[int]]
    group_representations: torch.Tensor  # [num_groups, hidden_dim]
    energy_signal: torch.Tensor  # [num_layers - 1]
    breakpoints: List[int]

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def teacher_anchor_layers(self) -> List[int]:
        """Last layer index of each group (for backwards compat with old code)."""
        return [g[-1] for g in self.groups]

    @property
    def student_target_layers(self) -> List[int]:
        """Not used in MOT-FD; provided for backwards compat."""
        return list(range(len(self.groups)))

    def pairs(self) -> List[tuple[int, int]]:
        """Not used in MOT-FD; provided for backwards compat."""
        return list(zip(self.teacher_anchor_layers, self.student_target_layers))


def _minmax_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize tensor to [0, 1] via min-max scaling."""
    return (x - x.min()) / (x.max() - x.min() + eps)


def compute_energy_signal(
    layer_reps: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
) -> torch.Tensor:
    """Compute representation dynamics energy.

    E(l) = α·norm(Δ¹_l) + β·norm(Δ²_l) + γ·norm(cosine_dist_l)

    Each component is min-max normalized to [0, 1] before weighting so that
    α, β, γ are pure contribution weights regardless of the raw scale of each
    term (L2 norms vs cosine distances have different magnitudes).

    Parameters
    ----------
    layer_reps:
        Teacher layer representations, shape ``[num_layers, hidden_dim]``.
    alpha, beta, gamma:
        Weights for the three energy components.

    Returns
    -------
    Energy signal of shape ``[num_layers - 1]`` (E(1) to E(L-1)).
    """
    num_layers, _ = layer_reps.shape
    if num_layers < 3:
        raise ValueError(f"Need at least 3 layers for energy computation, got {num_layers}")

    z = layer_reps  # [L, D]

    # First-order difference: ||z_{l+1} - z_l||
    dz = z[1:] - z[:-1]  # [L-1, D]
    first_order = dz.norm(dim=-1)  # [L-1]

    # Second-order difference: ||z_{l+1} - 2z_l + z_{l-1}||
    d2z = z[2:] - 2 * z[1:-1] + z[:-2]  # [L-2, D]
    second_order = torch.cat([
        torch.zeros(1, device=z.device, dtype=z.dtype),
        d2z.norm(dim=-1),
    ])  # [L-1]

    # Cosine distance: 1 - cos(z_{l+1}, z_l)
    z_l = z[:-1]  # [L-1, D]
    z_next = z[1:]  # [L-1, D]
    cos_sim = (z_l * z_next).sum(dim=-1) / (
        z_l.norm(dim=-1) * z_next.norm(dim=-1) + 1e-8
    )
    cosine_dist = 1.0 - cos_sim  # [L-1]

    # Normalize each component to [0, 1] so α/β/γ are dimensionless weights.
    energy = (
        alpha * _minmax_normalize(first_order)
        + beta * _minmax_normalize(second_order)
        + gamma * _minmax_normalize(cosine_dist)
    )
    return energy


def detect_breakpoints(
    energy: torch.Tensor,
    num_breakpoints: int = 11,
    min_distance: int = 2,
) -> List[int]:
    """Detect change points via peak detection on the energy signal.

    Finds local maxima in E(l), sorts by magnitude, and picks the top-K
    while enforcing a minimum distance constraint.

    Parameters
    ----------
    energy:
        Energy signal of shape ``[L-1]``.
    num_breakpoints:
        Number of breakpoints to detect (default: 11 → 12 groups).
    min_distance:
        Minimum distance between adjacent breakpoints.

    Returns
    -------
    Sorted list of breakpoint indices (1-indexed layer positions).
    """
    energy_np = energy.detach().cpu().float().numpy()
    L = len(energy_np)

    # Find local maxima
    candidates = []
    for i in range(1, L - 1):
        if energy_np[i] > energy_np[i - 1] and energy_np[i] > energy_np[i + 1]:
            candidates.append((i, energy_np[i]))

    # Sort by energy magnitude (descending)
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Greedy selection with min_distance
    selected: List[int] = []
    for idx, _ in candidates:
        # idx is position in energy signal (0-indexed),
        # the breakpoint is idx+1 in layer space (1-indexed)
        bp = idx + 1
        if all(abs(bp - s) >= min_distance for s in selected):
            selected.append(bp)
        if len(selected) >= num_breakpoints:
            break

    # If not enough peaks found, fill with evenly-spaced fallbacks.
    # Real peaks take priority — fallbacks only fill the remaining slots.
    if len(selected) < num_breakpoints:
        step = (L + 1) // (num_breakpoints + 1)  # num_layers // num_groups
        fallback = [step * (i + 1) for i in range(num_breakpoints)]
        for fb in fallback:
            if len(selected) >= num_breakpoints:
                break
            if fb not in selected and all(abs(fb - s) >= min_distance for s in selected):
                selected.append(fb)
    selected.sort()
    _logger.info(f"Detected {len(selected)} breakpoints: {selected}")
    return selected


def build_functional_groups(
    num_layers: int,
    breakpoints: List[int],
) -> List[List[int]]:
    """Build functional groups from breakpoints.

    G_k = [b_{k-1}, b_k) where b_0=0, b_{num_groups}=num_layers.

    Parameters
    ----------
    num_layers:
        Total number of teacher layers.
    breakpoints:
        Sorted list of breakpoints (b_1, ..., b_{G-1}).

    Returns
    -------
    List of G groups, each containing layer indices.
    """
    boundaries = [0] + sorted(breakpoints) + [num_layers]
    groups = []
    for k in range(len(boundaries) - 1):
        start = boundaries[k]
        end = boundaries[k + 1]
        groups.append(list(range(start, end)))
    return groups


def compute_group_representations(
    groups: List[List[int]],
    layer_reps: torch.Tensor,
) -> torch.Tensor:
    """Compute group representations: g_k^T = mean of z_l for l in G_k.

    Parameters
    ----------
    groups:
        List of functional groups.
    layer_reps:
        Teacher layer representations, shape ``[num_layers, hidden_dim]``.

    Returns
    -------
    Group representations, shape ``[num_groups, hidden_dim]``.
    """
    reps = []
    for g in groups:
        g_reps = layer_reps[g]  # [|G_k|, D]
        reps.append(g_reps.mean(dim=0))
    return torch.stack(reps, dim=0)  # [num_groups, D]


def build_group_assignment(
    teacher_layer_reps: torch.Tensor,
    num_groups: int = 12,
    energy_alpha: float = 1.0,
    energy_beta: float = 0.5,
    energy_gamma: float = 0.3,
    min_peak_distance: int = 2,
) -> GroupAssignment:
    """Main entry point: teacher functional decomposition via MOT-FD.

    1. Compute representation dynamics energy E(l).
    2. Detect change points (breakpoints) via peak detection.
    3. Build functional groups.
    4. Compute group representations g_k^T.

    Parameters
    ----------
    teacher_layer_reps:
        Teacher layer representations of shape ``[num_layers, hidden_dim]``.
        Obtain by running calibration data through the teacher and averaging
        hidden states per layer over batch and sequence dimensions.
    num_groups:
        Number of functional groups (default: 12).
    energy_alpha, energy_beta, energy_gamma:
        Weights for the three energy components.
    min_peak_distance:
        Minimum distance between detected breakpoints.

    Returns
    -------
    A :class:`GroupAssignment` with groups, representations, energy, and breakpoints.
    """
    num_layers = teacher_layer_reps.shape[0]
    if num_groups > num_layers:
        raise ValueError(
            f"num_groups={num_groups} cannot exceed num_layers={num_layers}"
        )

    # Step 1: Compute energy signal
    energy = compute_energy_signal(
        teacher_layer_reps,
        alpha=energy_alpha,
        beta=energy_beta,
        gamma=energy_gamma,
    )

    # Step 2: Detect breakpoints
    num_breakpoints = num_groups - 1
    breakpoints = detect_breakpoints(
        energy,
        num_breakpoints=num_breakpoints,
        min_distance=min_peak_distance,
    )

    # Step 3: Build functional groups
    groups = build_functional_groups(num_layers, breakpoints)

    # Step 4: Compute group representations
    group_reps = compute_group_representations(groups, teacher_layer_reps)

    _logger.info(
        f"Built {len(groups)} functional groups from {num_layers} teacher layers: "
        f"group sizes={[len(g) for g in groups]}, "
        f"breakpoints={breakpoints}"
    )

    return GroupAssignment(
        groups=groups,
        group_representations=group_reps,
        energy_signal=energy,
        breakpoints=breakpoints,
    )


# ---------------------------------------------------------------------------
# Legacy API — kept for backwards compatibility with QAT trainer.
# ---------------------------------------------------------------------------


def build_group_assignment_legacy(
    teacher_num_layers: int,
    student_num_layers: int,
    num_groups: int,
    strategy: str = "uniform",
) -> GroupAssignment:
    """Legacy group assignment using uniform strategy.

    This is kept for QAT trainer compatibility. In MOT-FD, the proper entry
    point is :func:`build_group_assignment` with teacher layer representations.

    Returns a GroupAssignment with empty group_representations (since no
    calibration pass was done). The QAT trainer uses its own hidden-state
    hooks for cosine matching, not OT alignment.
    """
    if num_groups > min(teacher_num_layers, student_num_layers):
        raise ValueError(
            f"num_groups={num_groups} cannot exceed min(teacher, student) layers"
        )
    base, extra = divmod(teacher_num_layers, num_groups)
    cursor = 0
    groups = []
    for g in range(num_groups):
        size = base + (1 if g < extra else 0)
        groups.append(list(range(cursor, cursor + size)))
        cursor += size

    return GroupAssignment(
        groups=groups,
        group_representations=torch.empty(0),  # placeholder
        energy_signal=torch.empty(0),
        breakpoints=[g[-1] for g in groups[:-1]],
    )
