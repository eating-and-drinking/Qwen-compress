# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Group-wise teacher-student layer mapping.

The idea: when distilling a 14B teacher (e.g. 40 decoder layers) into a 3B
student (e.g. 32 layers), naive last-layer logit matching wastes the rich
intermediate signal the teacher offers. Group-wise distillation splits the
teacher into ``G`` contiguous *groups* of blocks, picks one *anchor layer* per
group as the supervision target, and maps it to the closest-fraction-depth
student layer. The student is then trained to match those anchors' hidden
states and attention maps.

Two strategies are supported:

* ``uniform``     : groups have equal size; the anchor is the *last* block of
                    each group (most informative output of that group).
* ``depth_aware`` : groups are sized proportionally to depth so that earlier
                    layers (which encode more local features) get more anchors
                    than late layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence

from qwen_compress.utils.logging import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True)
class GroupAssignment:
    """Layer-level mapping from teacher anchors to student layers.

    Attributes
    ----------
    teacher_anchor_layers:
        Sorted list of teacher layer indices (0-based) that are used as
        distillation targets — one per group.
    student_target_layers:
        For each teacher anchor, the matched student layer index.
    num_groups:
        ``len(teacher_anchor_layers)``.
    """

    teacher_anchor_layers: List[int]
    student_target_layers: List[int]

    @property
    def num_groups(self) -> int:
        return len(self.teacher_anchor_layers)

    def pairs(self) -> List[tuple[int, int]]:
        """Convenience: ``list(zip(teacher_anchors, student_targets))``."""
        return list(zip(self.teacher_anchor_layers, self.student_target_layers))


def _uniform_anchors(num_layers: int, num_groups: int) -> List[int]:
    """Split [0, num_layers) into ``num_groups`` contiguous chunks; pick last."""
    if num_groups <= 0 or num_groups > num_layers:
        raise ValueError(
            f"num_groups={num_groups} must be in (0, num_layers={num_layers}]"
        )
    # Compute group boundaries via numpy-free integer split.
    boundaries: List[int] = []
    base, extra = divmod(num_layers, num_groups)
    cursor = 0
    for g in range(num_groups):
        size = base + (1 if g < extra else 0)
        cursor += size
        boundaries.append(cursor - 1)  # last index of this group
    return boundaries


def _depth_aware_anchors(num_layers: int, num_groups: int) -> List[int]:
    """Allocate more groups to early/middle layers (sqrt-weighted)."""
    import math

    if num_groups <= 0 or num_groups > num_layers:
        raise ValueError(
            f"num_groups={num_groups} must be in (0, num_layers={num_layers}]"
        )
    # Weight each layer by sqrt of (num_layers - i) so early layers get bigger weight.
    weights = [math.sqrt(num_layers - i) for i in range(num_layers)]
    total = sum(weights)
    cum = 0.0
    anchors: List[int] = []
    target = total / num_groups
    next_threshold = target
    for i, w in enumerate(weights):
        cum += w
        if cum >= next_threshold and len(anchors) < num_groups:
            anchors.append(i)
            next_threshold += target
    # Guarantee exactly num_groups anchors and that the last layer is included.
    while len(anchors) < num_groups:
        candidate = num_layers - 1
        while candidate in anchors and candidate > 0:
            candidate -= 1
        anchors.append(candidate)
    anchors = sorted(set(anchors))[:num_groups]
    if anchors[-1] != num_layers - 1:
        anchors[-1] = num_layers - 1
    return anchors


def build_group_assignment(
    teacher_num_layers: int,
    student_num_layers: int,
    num_groups: int,
    strategy: Literal["uniform", "depth_aware"] = "uniform",
) -> GroupAssignment:
    """Build the teacher-to-student anchor mapping.

    Parameters
    ----------
    teacher_num_layers:
        Number of decoder layers in the teacher.
    student_num_layers:
        Number of decoder layers in the student.
    num_groups:
        Number of supervision points. Must satisfy
        ``1 <= num_groups <= min(teacher_num_layers, student_num_layers)``.
    strategy:
        ``"uniform"`` or ``"depth_aware"``.

    Returns
    -------
    A :class:`GroupAssignment`.

    Notes
    -----
    For each teacher anchor at depth ``d_T``, the student target is set to
    ``round((d_T + 1) / teacher_num_layers * student_num_layers) - 1``, capped
    to ``[0, student_num_layers - 1]``. This preserves relative depth.
    """
    if num_groups > min(teacher_num_layers, student_num_layers):
        raise ValueError(
            f"num_groups={num_groups} cannot exceed min(teacher_num_layers, "
            f"student_num_layers) = {min(teacher_num_layers, student_num_layers)}"
        )

    if strategy == "uniform":
        teacher_anchors = _uniform_anchors(teacher_num_layers, num_groups)
    elif strategy == "depth_aware":
        teacher_anchors = _depth_aware_anchors(teacher_num_layers, num_groups)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    student_targets: List[int] = []
    for d_t in teacher_anchors:
        # Map (d_t + 1)/T_L  to  s_L positions, then take the 1-indexed -> 0-indexed.
        rel = (d_t + 1) / teacher_num_layers
        s_idx = int(round(rel * student_num_layers)) - 1
        s_idx = max(0, min(student_num_layers - 1, s_idx))
        student_targets.append(s_idx)

    # Ensure student targets are strictly increasing — otherwise some groups
    # share a student layer and we lose supervision diversity.
    student_targets = _enforce_monotonic(student_targets, student_num_layers - 1)

    assignment = GroupAssignment(
        teacher_anchor_layers=teacher_anchors,
        student_target_layers=student_targets,
    )
    _logger.info(
        f"Built group assignment ({strategy}): "
        f"teacher anchors={teacher_anchors}, student targets={student_targets}"
    )
    return assignment


def _enforce_monotonic(seq: Sequence[int], max_value: int) -> List[int]:
    """Adjust ``seq`` (in-place semantics) to be strictly increasing in [0, max_value]."""
    out = list(seq)
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = min(out[i - 1] + 1, max_value)
    # Backward pass to keep within bound.
    for i in range(len(out) - 2, -1, -1):
        if out[i] >= out[i + 1]:
            out[i] = max(0, out[i + 1] - 1)
    return out
