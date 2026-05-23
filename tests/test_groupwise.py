# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Unit tests for the group assignment logic."""

from __future__ import annotations

import pytest

from qwen_compress.distill.groupwise import build_group_assignment


class TestUniformStrategy:
    def test_simple_14b_to_3b(self):
        # Qwen2.5-14B has 48 layers; Qwen2.5-3B has 36.
        a = build_group_assignment(48, 36, num_groups=12, strategy="uniform")
        assert len(a.teacher_anchor_layers) == 12
        assert len(a.student_target_layers) == 12
        assert a.teacher_anchor_layers[-1] == 47
        assert all(0 <= s < 36 for s in a.student_target_layers)
        # Monotonic.
        assert a.teacher_anchor_layers == sorted(a.teacher_anchor_layers)
        assert a.student_target_layers == sorted(a.student_target_layers)

    def test_equal_depth(self):
        a = build_group_assignment(32, 32, num_groups=8, strategy="uniform")
        assert a.teacher_anchor_layers == a.student_target_layers

    def test_invalid_num_groups(self):
        with pytest.raises(ValueError):
            build_group_assignment(10, 5, num_groups=20, strategy="uniform")

    def test_single_group(self):
        a = build_group_assignment(40, 32, num_groups=1, strategy="uniform")
        assert a.num_groups == 1
        # Single group => only anchor is the last layer of the teacher.
        assert a.teacher_anchor_layers == [39]
        assert a.student_target_layers == [31]


class TestDepthAwareStrategy:
    def test_returns_correct_count(self):
        a = build_group_assignment(48, 36, num_groups=12, strategy="depth_aware")
        assert a.num_groups == 12

    def test_includes_final_layer(self):
        a = build_group_assignment(48, 36, num_groups=8, strategy="depth_aware")
        assert a.teacher_anchor_layers[-1] == 47


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        build_group_assignment(10, 10, num_groups=2, strategy="bogus")  # type: ignore[arg-type]
