# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Unit tests for MOT-FD group assignment (functional decomposition)."""

from __future__ import annotations

import pytest
import torch

from qwen_compress.distill.groupwise import (
    GroupAssignment,
    build_functional_groups,
    build_group_assignment,
    compute_energy_signal,
    compute_group_representations,
    detect_breakpoints,
)


class TestEnergySignal:
    def test_shape(self):
        z = torch.randn(10, 16)
        energy = compute_energy_signal(z, alpha=1.0, beta=0.5, gamma=0.3)
        assert energy.shape == (9,)
        assert torch.all(energy >= 0)

    def test_constant_input_zero(self):
        """Identical layer reps → no change → energy near zero."""
        z = torch.ones(5, 8) * 3.0
        energy = compute_energy_signal(z)
        assert energy.max().item() == pytest.approx(0.0, abs=1e-5)

    def test_increasing_variation(self):
        """Gradually changing reps should produce increasing energy."""
        z = torch.stack([torch.linspace(0, i, 8) for i in range(10)], dim=0)
        energy = compute_energy_signal(z)
        assert energy.sum() > 0

    def test_too_few_layers_raises(self):
        with pytest.raises(ValueError):
            compute_energy_signal(torch.randn(2, 8))


class TestBreakpointDetection:
    def test_returns_correct_count(self):
        """Energy signal with clear peaks → exact 11 breakpoints."""
        energy = torch.linspace(0, 1, 47)
        # Add clear peaks at positions 5, 10, 15, 20, ..., 55
        for i in range(5, 47, 5):
            energy[i] = 100.0
        bps = detect_breakpoints(energy, num_breakpoints=11, min_distance=2)
        assert len(bps) == 11

    def test_sorted_output(self):
        energy = torch.randn(47)
        # Create artificial peaks
        energy[10] = 10.0
        energy[20] = 10.0
        energy[30] = 10.0
        bps = detect_breakpoints(energy, num_breakpoints=3, min_distance=2)
        assert bps == sorted(bps)


class TestFunctionalGroups:
    def test_build_groups(self):
        bps = [10, 20, 35]
        groups = build_functional_groups(48, bps)
        assert len(groups) == 4
        assert groups[0] == list(range(0, 10))
        assert groups[1] == list(range(10, 20))
        assert groups[2] == list(range(20, 35))
        assert groups[3] == list(range(35, 48))

    def test_group_reps_shape(self):
        groups = [list(range(0, 4)), list(range(4, 10))]
        z = torch.randn(10, 16)
        reps = compute_group_representations(groups, z)
        assert reps.shape == (2, 16)


class TestBuildGroupAssignment:
    def test_end_to_end(self):
        """Full pipeline: layer reps → decomposition → assignment."""
        z = torch.randn(48, 2048)
        # Add artificial structure: early layers similar, late layers drift
        z[24:] = z[24:] + 0.5
        z[40:] = z[40:] + 0.5
        assignment = build_group_assignment(
            teacher_layer_reps=z,
            num_groups=12,
            energy_alpha=1.0,
            energy_beta=0.5,
            energy_gamma=0.3,
            min_peak_distance=2,
        )
        assert isinstance(assignment, GroupAssignment)
        assert assignment.num_groups == 12
        assert assignment.group_representations.shape[0] == 12
        assert assignment.group_representations.shape[1] == 2048
        assert len(assignment.breakpoints) == 11
        # All 48 layers accounted for
        all_layers = [l for g in assignment.groups for l in g]
        assert all_layers == list(range(48))

    def test_invalid_num_groups(self):
        z = torch.randn(10, 16)
        with pytest.raises(ValueError):
            build_group_assignment(z, num_groups=20)

    def test_properties_backward_compat(self):
        z = torch.randn(48, 256)
        assignment = build_group_assignment(z, num_groups=12)
        assert len(assignment.teacher_anchor_layers) == 12
        assert len(assignment.student_target_layers) == 12
        assert len(assignment.pairs()) == 12
