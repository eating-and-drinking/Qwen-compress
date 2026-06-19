# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Pydantic-backed configuration models."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["adamw", "adamw_8bit", "lion"] = "adamw"
    lr: float = Field(default=1e-4, gt=0.0)
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(default=0.0, ge=0.0)
    eps: float = Field(default=1e-8, gt=0.0)


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["constant", "linear", "cosine", "cosine_with_warmup"] = "cosine_with_warmup"
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train_path: str
    eval_path: Optional[str] = None
    max_seq_length: int = Field(default=2048, ge=1)
    batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    num_workers: int = Field(default=4, ge=0)
    shuffle_seed: int = 42
    pack_sequences: bool = False
    cot_mode: Literal["direct", "cot", "dual"] = "dual"


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_dir: str = "./checkpoints"
    num_train_epochs: float = Field(default=3.0, gt=0.0)
    max_steps: int = Field(default=-1, ge=-1)
    gradient_checkpointing: bool = True
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    bf16: bool = True
    fp16: bool = False
    save_steps: int = Field(default=500, ge=1)
    eval_steps: int = Field(default=500, ge=1)
    logging_steps: int = Field(default=10, ge=1)
    save_total_limit: int = Field(default=3, ge=0)
    seed: int = 42
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()

    @model_validator(mode="after")
    def _exclusive_precision(self) -> "TrainingConfig":
        if self.bf16 and self.fp16:
            raise ValueError("Only one of `bf16` or `fp16` may be true.")
        return self


class DistillConfig(BaseModel):
    """MOT-FD distillation configuration.

    Key parameters for Monotonic Optimal Transport Functional Distillation:
    - Teacher decomposition: energy_alpha/beta/gamma, min_peak_distance
    - OT alignment: lambda_ot, lambda_mono, ot_temperature, sinkhorn_iters
    - Composite loss: L = α·CE + β·KD + λ_ot·L_OT + λ_mono·L_mono
    
    Enhanced features:
    - Adaptive OT temperature: dynamically adjust based on alignment difficulty
    - Bidirectional alignment: student→teacher and teacher→student
    - Attention distillation: match attention patterns
    - Dynamic functional groups: update group reps during training
    """
    model_config = ConfigDict(extra="forbid")
    teacher_model_name_or_path: str
    student_model_name_or_path: str

    # ---- Functional decomposition ----
    num_groups: int = Field(default=12, ge=1)
    calibration_samples: int = Field(default=256, ge=1)
    energy_alpha: float = Field(default=1.0, ge=0.0)
    energy_beta: float = Field(default=0.5, ge=0.0)
    energy_gamma: float = Field(default=0.3, ge=0.0)
    min_peak_distance: int = Field(default=2, ge=1)

    # ---- Composite loss weights (MOT-FD) ----
    alpha_ce: float = Field(default=1.0, ge=0.0)
    beta_kd: float = Field(default=1.0, ge=0.0)
    gamma_hidden: float = Field(default=0.0, ge=0.0)   # legacy, 0 disables
    delta_attn: float = Field(default=0.0, ge=0.0)      # attention distillation

    # ---- OT-specific weights ----
    lambda_ot: float = Field(default=1.0, ge=0.0)
    lambda_mono: float = Field(default=0.1, ge=0.0)

    # ---- Temperature / hyperparams ----
    kd_temperature: float = Field(default=2.0, gt=0.0)
    ot_temperature: float = Field(default=0.1, gt=0.0)
    sinkhorn_iters: int = Field(default=50, ge=1)

    # ---- Adaptive OT temperature ----
    adaptive_ot_temp: bool = Field(default=False)
    adaptive_temp_min: float = Field(default=0.05, gt=0.0)
    adaptive_temp_max: float = Field(default=0.5, gt=0.0)
    adaptive_temp_scale: float = Field(default=1.0, ge=0.0)

    # ---- Attention distillation ----
    attn_distill_strategy: Literal["kl", "cosine", "mse", "ot"] = "kl"
    attn_ot_temperature: float = Field(default=0.1, gt=0.0)
    attn_sinkhorn_iters: int = Field(default=50, ge=1)

    # ---- Dynamic functional groups ----
    dynamic_groups: bool = Field(default=False)
    dynamic_groups_update_interval: int = Field(default=500, ge=1)
    dynamic_groups_momentum: float = Field(default=0.99, ge=0.0, le=1.0)

    # ---- Memory optimizations ----
    teacher_load_in_8bit: bool = False
    teacher_load_in_4bit: bool = False

    @model_validator(mode="after")
    def _exclusive_teacher_quant(self) -> "DistillConfig":
        if self.teacher_load_in_8bit and self.teacher_load_in_4bit:
            raise ValueError("teacher_load_in_8bit and teacher_load_in_4bit are mutually exclusive.")
        return self

    # ---- Training ----
    freeze_embedding: bool = False
    projector_lr_multiplier: float = Field(default=0.1, ge=0.0, le=1.0)
    data: DataConfig
    training: TrainingConfig


class PermutationConfig(BaseModel):
    """Channel-permutation pass that aligns unstructured sparsity to N:M."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    target: Literal["2:4", "4:8"] = "2:4"
    algorithm: Literal["greedy"] = "greedy"
    max_iters: int = Field(default=300, ge=1)
    swaps_per_iter: int = Field(default=200, ge=1)
    patience: int = Field(default=30, ge=1)
    enforce_after: bool = True
    seed: int = 0


class PruneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name_or_path: str
    output_dir: str = "./checkpoints/pruned"
    sparsity: float = Field(default=0.5, ge=0.0, lt=1.0)
    sparsity_type: Literal["unstructured", "2:4", "4:8"] = "unstructured"
    block_size: int = Field(default=128, ge=1)
    percdamp: float = Field(default=0.01, gt=0.0, le=1.0)
    nsamples: int = Field(default=128, ge=1)
    calibration_path: str
    max_seq_length: int = Field(default=2048, ge=1)
    seed: int = 42
    device: str = "cuda"
    permutation: Optional[PermutationConfig] = None
    recovery_finetune: bool = True
    recovery: Optional[TrainingConfig] = None


class QATConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name_or_path: str
    teacher_model_name_or_path: Optional[str] = None
    weight_bits: Literal[4, 8] = 8
    activation_bits: Literal[8, 16] = 8
    weight_granularity: Literal["per_tensor", "per_channel"] = "per_channel"
    activation_granularity: Literal["per_tensor", "per_token"] = "per_tensor"
    weight_symmetric: bool = True
    activation_symmetric: bool = False
    quantize_kv_cache: bool = True
    quantize_lm_head: bool = False
    skip_layers: List[str] = Field(default_factory=lambda: ["lm_head", "embed_tokens"])
    calibration_path: str
    nsamples_calib: int = Field(default=512, ge=1)
    calibration_method: Literal["minmax", "percentile", "mse", "entropy"] = "percentile"
    percentile: float = Field(default=99.99, gt=0.0, le=100.0)
    use_qad: bool = True
    alpha_ce: float = Field(default=0.5, ge=0.0)
    beta_kd: float = Field(default=1.0, ge=0.0)
    gamma_hidden: float = Field(default=0.5, ge=0.0)
    kd_temperature: float = Field(default=2.0, gt=0.0)
    export_format: Literal["safetensors", "onnx"] = "safetensors"
    data: DataConfig
    training: TrainingConfig


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "qwen-compress-pipeline"
    distill: Optional[DistillConfig] = None
    prune: Optional[PruneConfig] = None
    qat: Optional[QATConfig] = None


_STAGE_MAP: Dict[str, Any] = {
    "distill": DistillConfig,
    "prune": PruneConfig,
    "qat": QATConfig,
    "pipeline": PipelineConfig,
}


def load_config(path, stage=None):
    """Load a YAML config and validate against the schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}
    file_stage = raw.pop("stage", None)
    if stage is None:
        stage = file_stage
        if stage is None:
            raise ValueError(f"Config {path} missing `stage` and none passed.")
    if stage not in _STAGE_MAP:
        raise ValueError(f"Unknown stage {stage!r}.")
    return _STAGE_MAP[stage].model_validate(raw)
