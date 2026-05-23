# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Schema validation tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from qwen_compress.utils.config import (
    DistillConfig,
    PruneConfig,
    QATConfig,
    load_config,
)


def _yaml_tempfile(payload: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(payload, tmp)
    tmp.close()
    return Path(tmp.name)


def test_distill_minimal_valid():
    payload = {
        "stage": "distill",
        "teacher_model_name_or_path": "Qwen/Qwen2.5-14B-Instruct",
        "student_model_name_or_path": "Qwen/Qwen2.5-3B",
        "data": {"train_path": "data.jsonl"},
        "training": {},
    }
    cfg = load_config(_yaml_tempfile(payload))
    assert isinstance(cfg, DistillConfig)
    assert cfg.num_groups == 12
    assert cfg.training.bf16 is True


def test_prune_minimal_valid():
    payload = {
        "stage": "prune",
        "model_name_or_path": "Qwen/Qwen2.5-3B",
        "calibration_path": "data.jsonl",
    }
    cfg = load_config(_yaml_tempfile(payload))
    assert isinstance(cfg, PruneConfig)
    assert cfg.sparsity == 0.5


def test_qat_minimal_valid():
    payload = {
        "stage": "qat",
        "model_name_or_path": "Qwen/Qwen2.5-3B",
        "calibration_path": "data.jsonl",
        "data": {"train_path": "data.jsonl"},
        "training": {},
    }
    cfg = load_config(_yaml_tempfile(payload))
    assert isinstance(cfg, QATConfig)


def test_rejects_unknown_field():
    payload = {
        "stage": "distill",
        "teacher_model_name_or_path": "x",
        "student_model_name_or_path": "y",
        "data": {"train_path": "data.jsonl"},
        "training": {},
        "bogus_field": True,
    }
    with pytest.raises(Exception):
        load_config(_yaml_tempfile(payload))


def test_bf16_fp16_mutually_exclusive():
    payload = {
        "stage": "distill",
        "teacher_model_name_or_path": "x",
        "student_model_name_or_path": "y",
        "data": {"train_path": "data.jsonl"},
        "training": {"bf16": True, "fp16": True},
    }
    with pytest.raises(Exception):
        load_config(_yaml_tempfile(payload))
