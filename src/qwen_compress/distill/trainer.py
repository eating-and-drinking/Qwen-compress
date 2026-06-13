# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""MOT-FD: Monotonic Optimal Transport Functional Distillation Trainer.

Pipeline:
1. Load teacher (frozen) and student (trainable) models.
2. Teacher functional decomposition:
   a. Run calibration data through teacher to extract layer representations.
   b. Compute representation dynamics energy.
   c. Detect change points → build 12 functional groups.
   d. Compute group representations g_k^T.
3. Register hooks to capture ALL student layer hidden states.
4. Training loop with composite loss:
   L = L_CE + λ_KD·L_KD + λ_OT·L_OT + λ_mono·L_mono
"""

from __future__ import annotations

import itertools
import math
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from qwen_compress.data.cot_dataset import CoTDataset, DataCollatorForCoT
from qwen_compress.distill.groupwise import GroupAssignment, build_group_assignment
from qwen_compress.distill.losses import DistillationLoss
from qwen_compress.models.qwen_wrapper import (
    get_decoder_layers,
    inspect_qwen_model,
    load_qwen_model,
    load_qwen_tokenizer,
)
from qwen_compress.utils.checkpoint import rotate_checkpoints, save_compressed_model
from qwen_compress.utils.config import DistillConfig
from qwen_compress.utils.dist import is_main_process
from qwen_compress.utils.logging import get_logger
from qwen_compress.utils.seed import set_seed

_logger = get_logger(__name__)


# ============================================================================
# Hook helpers
# ============================================================================


class _LayerOutputCapture:
    """Forward hook that stashes the output hidden state of a decoder layer."""

    def __init__(self) -> None:
        self.last_output: Optional[torch.Tensor] = None
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def attach(self, layer: nn.Module) -> None:
        if self._handle is not None:
            raise RuntimeError("Hook already attached.")

        def _hook(_mod, _inp, out):  # noqa: ANN001
            self.last_output = out[0] if isinstance(out, tuple) else out

        self._handle = layer.register_forward_hook(_hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.last_output = None


class _AttentionCapture:
    """Forward hook that captures attention weights from a decoder layer."""

    def __init__(self) -> None:
        self.last_attention: Optional[torch.Tensor] = None
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def attach(self, layer: nn.Module) -> None:
        if self._handle is not None:
            raise RuntimeError("Hook already attached.")

        def _hook(_mod, _inp, out):  # noqa: ANN001
            if isinstance(out, tuple):
                if len(out) > 1 and isinstance(out[1], dict) and 'attentions' in out[1]:
                    self.last_attention = out[1]['attentions']
                elif hasattr(out[0], 'attentions'):
                    self.last_attention = out[0].attentions
            elif hasattr(out, 'attentions'):
                self.last_attention = out.attentions

        self._handle = layer.register_forward_hook(_hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.last_attention = None


# ============================================================================
# LR Scheduler
# ============================================================================


def _make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
    name: str,
) -> LambdaLR:
    warmup_steps = max(1, int(num_training_steps * warmup_ratio))

    def cosine_with_warmup(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def linear(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        return max(min_lr_ratio, 1.0 - progress)

    if name in ("cosine_with_warmup", "cosine"):
        return LambdaLR(optimizer, cosine_with_warmup)
    if name == "linear":
        return LambdaLR(optimizer, linear)
    return LambdaLR(optimizer, lambda _s: 1.0)


# ============================================================================
# MOT-FD Trainer
# ============================================================================


class GroupwiseDistillTrainer:
    """End-to-end MOT-FD distillation orchestrator.

    Parameters
    ----------
    config:
        A validated :class:`DistillConfig`.
    """

    def __init__(self, config: DistillConfig) -> None:
        self.config = config
        set_seed(config.training.seed)

        # ---- Models -----------------------------------------------------
        self.tokenizer = load_qwen_tokenizer(config.student_model_name_or_path)
        dtype = "bf16" if config.training.bf16 else ("fp16" if config.training.fp16 else "fp32")

        _logger.info("Loading teacher (frozen)...")
        self.teacher = load_qwen_model(
            config.teacher_model_name_or_path,
            dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        _logger.info("Loading student (trainable)...")
        self.student = load_qwen_model(
            config.student_model_name_or_path,
            dtype=dtype,
            device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
            attn_implementation="sdpa",
            gradient_checkpointing=config.training.gradient_checkpointing,
        )
        self.student.train()

        if config.freeze_embedding:
            self.student.get_input_embeddings().requires_grad_(False)

        t_info = inspect_qwen_model(self.teacher)
        s_info = inspect_qwen_model(self.student)

        # ---- Phase 0: Teacher Functional Decomposition ------------------
        _logger.info("Phase 0: Teacher functional decomposition (48 → 12 groups)...")
        teacher_layer_reps = self._run_teacher_decomposition()
        self.assignment: GroupAssignment = build_group_assignment(
            teacher_layer_reps=teacher_layer_reps,
            num_groups=config.num_groups,
            energy_alpha=config.energy_alpha,
            energy_beta=config.energy_beta,
            energy_gamma=config.energy_gamma,
            min_peak_distance=config.min_peak_distance,
        )

        # ---- Loss (MOT-FD mode with enhanced features) -------------------
        self.loss_fn = DistillationLoss(
            student_hidden_size=s_info.hidden_size,
            teacher_hidden_size=t_info.hidden_size,
            teacher_group_reps=self.assignment.group_representations,
            alpha_ce=config.alpha_ce,
            beta_kd=config.beta_kd,
            gamma_hidden=config.gamma_hidden,
            delta_attn=config.delta_attn,
            lambda_ot=config.lambda_ot,
            lambda_mono=config.lambda_mono,
            lambda_ot_backward=getattr(config, 'lambda_ot_backward', 0.0),
            kd_temperature=config.kd_temperature,
            ot_temperature=config.ot_temperature,
            sinkhorn_iters=config.sinkhorn_iters,
            adaptive_ot_temp=getattr(config, 'adaptive_ot_temp', False),
            adaptive_temp_min=getattr(config, 'adaptive_temp_min', 0.05),
            adaptive_temp_max=getattr(config, 'adaptive_temp_max', 0.5),
            adaptive_temp_scale=getattr(config, 'adaptive_temp_scale', 1.0),
            attn_distill_strategy=getattr(config, 'attn_distill_strategy', 'kl'),
            attn_ot_temperature=getattr(config, 'attn_ot_temperature', 0.1),
            attn_sinkhorn_iters=getattr(config, 'attn_sinkhorn_iters', 50),
        )
        self.loss_fn.to(next(self.student.parameters()).device)

        # ---- Hooks: capture student hidden states and attention ----------
        self._student_captures: List[_LayerOutputCapture] = []
        self._student_attn_captures: List[_AttentionCapture] = []
        self._attach_student_hooks()

        # ---- Data --------------------------------------------------------
        train_ds = CoTDataset(
            path=config.data.train_path,
            tokenizer=self.tokenizer,
            max_seq_length=config.data.max_seq_length,
            mode=config.data.cot_mode,
            seed=config.data.shuffle_seed,
        )
        collator = DataCollatorForCoT(self.tokenizer)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=True,
        )

        # ---- Validation Data --------------------------------------------
        self.eval_loader: Optional[DataLoader] = None
        if config.data.eval_path:
            eval_ds = CoTDataset(
                path=config.data.eval_path,
                tokenizer=self.tokenizer,
                max_seq_length=config.data.max_seq_length,
                mode=config.data.cot_mode,
                seed=config.data.shuffle_seed,
            )
            self.eval_loader = DataLoader(
                eval_ds,
                batch_size=config.data.batch_size,
                shuffle=False,
                num_workers=min(1, config.data.num_workers),
                collate_fn=collator,
                pin_memory=True,
                drop_last=False,
            )
            _logger.info(f"Loaded validation dataset with {len(eval_ds)} examples")

        # ---- Optimizer / Scheduler ---------------------------------------
        student_params = [p for p in self.student.parameters() if p.requires_grad]
        projector_params = [
            p for p in self.loss_fn.parameters() if p.requires_grad
        ]
        projector_lr = config.projector_lr_multiplier
        opt = config.training.optimizer
        param_groups = [
            {"params": student_params, "lr": opt.lr},
            {"params": projector_params, "lr": opt.lr * projector_lr},
        ]
        self.optimizer = AdamW(
            param_groups,
            lr=opt.lr,
            betas=opt.betas,
            eps=opt.eps,
            weight_decay=opt.weight_decay,
        )

        steps_per_epoch = max(1, len(self.train_loader) // config.data.gradient_accumulation_steps)
        if config.training.max_steps > 0:
            self.total_steps = config.training.max_steps
        else:
            self.total_steps = max(1, int(steps_per_epoch * config.training.num_train_epochs))
        self.scheduler = _make_lr_scheduler(
            self.optimizer,
            num_training_steps=self.total_steps,
            warmup_ratio=config.training.scheduler.warmup_ratio,
            min_lr_ratio=config.training.scheduler.min_lr_ratio,
            name=config.training.scheduler.name,
        )

        # ---- AMP ----------------------------------------------------------
        self._amp_dtype: Optional[torch.dtype] = None
        if config.training.bf16:
            self._amp_dtype = torch.bfloat16
        elif config.training.fp16:
            self._amp_dtype = torch.float16
        self._scaler = (
            torch.cuda.amp.GradScaler() if config.training.fp16 else None
        )

    # ======================================================================
    # Teacher Decomposition (Phase 0)
    # ======================================================================

    def _run_teacher_decomposition(self) -> torch.Tensor:
        """Run calibration data through teacher to extract layer representations.

        1. Register hooks on ALL 48 teacher layers.
        2. Run a small calibration set (up to 256 samples) through the teacher.
        3. For each layer, accumulate hidden states and compute mean z_l^T.
        4. Returns tensor of shape [num_layers, hidden_dim].

        Returns
        -------
        Teacher layer representations z_l^T for all layers.
        """
        cfg = self.config
        t_info = inspect_qwen_model(self.teacher)
        t_layers = get_decoder_layers(self.teacher)

        # Register hooks on all teacher layers
        captures: List[_LayerOutputCapture] = []
        for layer in t_layers:
            cap = _LayerOutputCapture()
            cap.attach(layer)
            captures.append(cap)

        # Use a small subset of training data for calibration
        train_ds = CoTDataset(
            path=cfg.data.train_path,
            tokenizer=self.tokenizer,
            max_seq_length=cfg.data.max_seq_length,
            mode=cfg.data.cot_mode,
            seed=cfg.data.shuffle_seed,
        )
        collator = DataCollatorForCoT(self.tokenizer)
        calib_loader = DataLoader(
            train_ds,
            batch_size=1,  # single sample for decomposition
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
            pin_memory=True,
            drop_last=False,
        )

        device = next(self.student.parameters()).device
        num_calib = min(cfg.calibration_samples, len(calib_loader))

        # Accumulate representations per layer
        accum = [None] * t_info.num_hidden_layers
        count = 0

        for batch in itertools.islice(calib_loader, num_calib):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

            # Valid token mask for mean computation: use attention_mask
            # (all non-padding tokens carry semantic information).
            valid = (attention_mask == 1).unsqueeze(-1).to(dtype=torch.float32)  # [B, T, 1]

            for i, cap in enumerate(captures):
                h = cap.last_output
                if h is None:
                    continue
                h = h.to(device=device)
                # Mean over valid tokens and batch: [D]
                h_valid = h * valid.to(h.dtype)
                denom = valid.sum().clamp_min(1.0)
                rep = h_valid.sum(dim=(0, 1)) / denom

                if accum[i] is None:
                    accum[i] = rep
                else:
                    accum[i] += rep

            count += 1

        # Detach hooks
        for cap in captures:
            cap.detach()

        # Average over calibration samples
        layer_reps_list = []
        for a in accum:
            if a is not None:
                layer_reps_list.append(a / max(count, 1))
            else:
                # Fallback: zero vector
                layer_reps_list.append(torch.zeros(t_info.hidden_size, device=device))

        layer_reps = torch.stack(layer_reps_list, dim=0)  # [L, D]
        _logger.info(
            f"Teacher decomposition complete: {layer_reps.shape[0]} layers, "
            f"using {count} calibration samples."
        )
        return layer_reps

    # ======================================================================
    # Student hooks
    # ======================================================================

    def _attach_student_hooks(self) -> None:
        """Register hooks on ALL student decoder layers to capture hidden states and attentions."""
        s_layers = get_decoder_layers(self.student)
        for layer in s_layers:
            cap = _LayerOutputCapture()
            cap.attach(layer)
            self._student_captures.append(cap)
            if self.config.delta_attn > 0:
                attn_cap = _AttentionCapture()
                attn_cap.attach(layer)
                self._student_attn_captures.append(attn_cap)
        _logger.info(f"Attached hooks to {len(self._student_captures)} student layers "
                   f"({len(self._student_attn_captures)} attention hooks.")

    def _detach_hooks(self) -> None:
        for cap in self._student_captures:
            cap.detach()
        self._student_captures.clear()
        for cap in getattr(self, '_student_attn_captures', []):
            cap.detach()
        self._student_attn_captures.clear()

    # ======================================================================
    # Dynamic groups helpers
    # ======================================================================

    def _extract_teacher_layer_reps(
        self, batch: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Extract teacher layer representations for a single batch.

        Used to accumulate representations for dynamic group update.
        """
        device = next(self.student.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        t_layers = get_decoder_layers(self.teacher)
        teacher_captures: List[_LayerOutputCapture] = []
        for layer in t_layers:
            cap = _LayerOutputCapture()
            cap.attach(layer)
            teacher_captures.append(cap)

        try:
            with torch.no_grad():
                self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

            reps = []
            for cap in teacher_captures:
                if cap.last_output is not None:
                    rep = cap.last_output.mean(dim=(0, 1))
                    reps.append(rep.to(device))
                else:
                    return None
        finally:
            for cap in teacher_captures:
                cap.detach()

        if reps:
            return torch.stack(reps, dim=0)
        return None

    def _recompute_group_reps(
        self, layer_reps: torch.Tensor
    ) -> torch.Tensor:
        """Recompute group representations using cached group assignment.

        Uses the same group boundaries as the initial decomposition.
        """
        groups = self.assignment.groups
        group_reps = []
        for g in groups:
            g_reps = layer_reps[g]
            group_reps.append(g_reps.mean(dim=0))
        return torch.stack(group_reps, dim=0)

    # ======================================================================
    # Auto-cast context
    # ======================================================================

    @contextmanager
    def _autocast(self) -> Iterator[None]:
        if self._amp_dtype is not None and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                yield
        else:
            yield

    # ======================================================================
    # Forward pass
    # ======================================================================

    def _forward_pair(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """Run forward pass and return (student_logits, teacher_logits, student_hidden, student_attn, teacher_attn).

        In MOT-FD mode, only student hidden states are captured for OT alignment.
        Teacher is only run for logits (KD loss).
        When delta_attn > 0, also captures attention weights for distillation.
        """
        device = next(self.student.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        output_attn = self.config.delta_attn > 0

        with torch.no_grad():
            teacher_out = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=output_attn,
            )

        student_out = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=output_attn,
        )

        student_hidden = [c.last_output for c in self._student_captures]

        if any(h is None for h in student_hidden):
            raise RuntimeError(
                "A student hook did not capture output — verify layer indices."
            )

        # Collect attentions if configured
        student_attns = None
        teacher_attns = None
        if output_attn:
            if hasattr(student_out, 'attentions') and student_out.attentions is not None:
                student_attns = list(student_out.attentions)
            if hasattr(teacher_out, 'attentions') and teacher_out.attentions is not None:
                teacher_attns = list(teacher_out.attentions)

        return (
            student_out.logits,
            teacher_out.logits.to(device),
            student_hidden,
            student_attns,
            teacher_attns,
        )

    # ======================================================================
    # Evaluation
    # ======================================================================

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the student model on the validation dataset.

        Returns
        -------
        Dictionary containing validation loss and breakdown metrics.
        """
        if self.eval_loader is None:
            _logger.warning("No evaluation dataset configured. Skipping evaluation.")
            return {}

        self.student.eval()
        self.loss_fn.eval()

        total_loss: Dict[str, float] = {}
        count = 0

        device = next(self.student.parameters()).device

        with torch.no_grad():
            for batch in self.eval_loader:
                with self._autocast():
                    s_logits, t_logits, s_hidden, s_attns, t_attns = self._forward_pair(batch)
                    labels = batch["labels"].to(device)
                    attention_mask = batch["attention_mask"].to(device)

                    loss_out = self.loss_fn(
                        student_logits=s_logits,
                        teacher_logits=t_logits,
                        labels=labels,
                        attention_mask=attention_mask,
                        student_hidden_states=s_hidden,
                        teacher_hidden_states=None,
                        student_attentions=s_attns,
                        teacher_attentions=t_attns,
                    )

                    for k, v in loss_out.to_log_dict().items():
                        total_loss[k] = total_loss.get(k, 0.0) + v
                    count += 1

        self.student.train()
        self.loss_fn.train()

        if count > 0:
            avg_loss = {k: v / count for k, v in total_loss.items()}
            _logger.info(
                "Evaluation complete | " + " ".join(f"{k}={v:.4f}" for k, v in avg_loss.items())
            )
            return avg_loss
        return {}

    # ======================================================================
    # Training loop
    # ======================================================================

    def train(self) -> Path:
        """Run the full MOT-FD distillation loop.

        Returns
        -------
        Path to the final checkpoint.
        """
        cfg = self.config
        global_step = 0
        accumulated = 0
        running_loss: Dict[str, float] = {}
        t0 = time.time()
        output_dir = Path(cfg.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        _logger.info(
            f"Starting MOT-FD distillation: total_steps={self.total_steps}, "
            f"batch_size={cfg.data.batch_size}, "
            f"grad_accum={cfg.data.gradient_accumulation_steps}, "
            f"num_groups={self.assignment.num_groups}, "
            f"student_layers_captured={len(self._student_captures)}"
        )

        # Track best model based on validation loss
        best_val_loss: Optional[float] = None
        best_step: int = 0

        dynamic_groups_enabled = getattr(cfg, 'dynamic_groups', False)
        dynamic_update_interval = getattr(cfg, 'dynamic_groups_update_interval', 500)
        dynamic_momentum = getattr(cfg, 'dynamic_groups_momentum', 0.99)

        # Track running teacher layer reps for dynamic groups
        running_teacher_layer_reps = None
        running_teacher_count = 0

        while global_step < self.total_steps:
            for batch in self.train_loader:
                if global_step >= self.total_steps:
                    break

                with self._autocast():
                    s_logits, t_logits, s_hidden, s_attns, t_attns = self._forward_pair(batch)
                    labels = batch["labels"].to(s_logits.device)
                    attention_mask = batch["attention_mask"].to(s_logits.device)
                    loss_out = self.loss_fn(
                        student_logits=s_logits,
                        teacher_logits=t_logits,
                        labels=labels,
                        attention_mask=attention_mask,
                        student_hidden_states=s_hidden,
                        teacher_hidden_states=None,  # MOT-FD uses group reps
                        student_attentions=s_attns,
                        teacher_attentions=t_attns,
                    )
                    loss = loss_out.total / cfg.data.gradient_accumulation_steps

                # Accumulate teacher layer reps for dynamic groups
                if dynamic_groups_enabled:
                    device = s_logits.device
                    batch_teacher_reps = self._extract_teacher_layer_reps(batch)
                    if batch_teacher_reps is not None:
                        batch_teacher_reps = batch_teacher_reps.to(device)
                        if running_teacher_layer_reps is None:
                            running_teacher_layer_reps = batch_teacher_reps
                        else:
                            running_teacher_layer_reps = running_teacher_layer_reps + batch_teacher_reps
                        running_teacher_count += 1

                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulated += 1
                for k, v in loss_out.to_log_dict().items():
                    running_loss[k] = running_loss.get(k, 0.0) + v

                if accumulated >= cfg.data.gradient_accumulation_steps:
                    if self._scaler is not None:
                        self._scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for pg in self.optimizer.param_groups
                         for p in pg["params"] if p.grad is not None],
                        cfg.training.max_grad_norm,
                    )
                    if self._scaler is not None:
                        self._scaler.step(self.optimizer)
                        self._scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulated = 0
                    global_step += 1

                    # Dynamic groups: update teacher group reps periodically
                    if (dynamic_groups_enabled
                        and global_step % dynamic_update_interval == 0
                        and running_teacher_layer_reps is not None
                        and running_teacher_count > 0):
                        avg_teacher_reps = running_teacher_layer_reps / running_teacher_count
                        # Re-compute group representations based on new layer reps
                        new_group_reps = self._recompute_group_reps(
                            avg_teacher_reps.to(self.assignment.group_representations.device)
                        )
                        self.loss_fn.update_teacher_group_reps(
                            new_group_reps,
                            momentum=dynamic_momentum,
                        )
                        # Reset running accumulators
                        running_teacher_layer_reps = None
                        running_teacher_count = 0
                        _logger.info(
                            f"[step={global_step}] Updated teacher group representations "
                            f"via dynamic groups (momentum={dynamic_momentum})"
                        )

                    if global_step % cfg.training.logging_steps == 0 and is_main_process():
                        avg = {
                            k: v / (cfg.training.logging_steps * cfg.data.gradient_accumulation_steps)
                            for k, v in running_loss.items()
                        }
                        lr = self.scheduler.get_last_lr()[0]
                        dt = time.time() - t0
                        _logger.info(
                            f"step={global_step}/{self.total_steps} lr={lr:.3e} "
                            f"elapsed={dt:.0f}s | "
                            + " ".join(f"{k}={v:.4f}" for k, v in avg.items())
                        )
                        running_loss.clear()

                    # Save intermediate checkpoints only if save_total_limit > 0
                    if (
                        global_step % cfg.training.save_steps == 0
                        and is_main_process()
                        and cfg.training.save_total_limit > 0
                    ):
                        self._save(output_dir, f"step-{global_step}")
                        rotate_checkpoints(
                            output_dir, keep=cfg.training.save_total_limit, prefix="step-"
                        )

                    if (
                        global_step % cfg.training.eval_steps == 0
                        and is_main_process()
                    ):
                        eval_metrics = self.evaluate()
                        # Track best model based on validation total loss
                        if eval_metrics and "total" in eval_metrics:
                            current_val_loss = eval_metrics["total"]
                            if best_val_loss is None or current_val_loss < best_val_loss:
                                best_val_loss = current_val_loss
                                best_step = global_step
                                self._save(output_dir, "best")
                                _logger.info(
                                    f"New best model at step {global_step}: val_loss={current_val_loss:.4f}"
                                )

        # Final save: use best model if available, otherwise use last model
        if best_val_loss is not None:
            _logger.info(
                f"Using best model from step {best_step} (val_loss={best_val_loss:.4f}) as final model"
            )
            # Copy best to final
            best_path = output_dir / "best"
            final_path = output_dir / "final"
            if best_path.exists():
                if final_path.exists():
                    shutil.rmtree(final_path)
                shutil.copytree(best_path, final_path)
            else:
                final_path = self._save(output_dir, "final")
        else:
            final_path = self._save(output_dir, "final")
        
        self._detach_hooks()
        _logger.info(f"MOT-FD distillation complete. Final checkpoint: {final_path}")
        return final_path

    # ======================================================================
    # Checkpoint
    # ======================================================================

    def _save(self, output_dir: Path, name: str) -> Path:
        target = output_dir / name
        save_compressed_model(
            self.student,
            self.tokenizer,
            target,
            extra_meta={
                "stage": "distill",
                "algorithm": "mot_fd",
                "teacher": self.config.teacher_model_name_or_path,
                "num_groups": self.assignment.num_groups,
                "groups": [g for g in self.assignment.groups],
                "breakpoints": self.assignment.breakpoints,
                "alpha_ce": self.config.alpha_ce,
                "beta_kd": self.config.beta_kd,
                "lambda_ot": self.config.lambda_ot,
                "lambda_mono": self.config.lambda_mono,
            },
        )
        return target
