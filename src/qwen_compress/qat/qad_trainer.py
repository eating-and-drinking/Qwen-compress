# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Quantization-aware distillation (QAD) trainer.

This is the trainer that does the "QAT with a teacher" loop discussed in our
pipeline write-up: while fake-quant nodes are active on the student, a frozen
FP teacher provides soft-label supervision via KD loss, plus optional
hidden-state matching at a few anchor layers.

If ``config.teacher_model_name_or_path`` is ``None``, this falls back to plain
QAT (CE loss only) so the same code path handles both regimes.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from qwen_compress.data.calibration_data import build_calibration_loader
from qwen_compress.data.cot_dataset import CoTDataset, DataCollatorForCoT
from qwen_compress.distill.groupwise import build_group_assignment_legacy
from qwen_compress.distill.losses import DistillationLoss
from qwen_compress.models.qwen_wrapper import (
    get_decoder_layers,
    get_linear_layers_in_block,
    inspect_qwen_model,
    load_qwen_model,
    load_qwen_tokenizer,
)
from qwen_compress.qat.calibration import calibrate_model
from qwen_compress.qat.fake_quant import QuantSpec, prepare_qat_model
from qwen_compress.utils.checkpoint import rotate_checkpoints, save_compressed_model
from qwen_compress.utils.config import QATConfig
from qwen_compress.utils.dist import is_main_process
from qwen_compress.utils.logging import get_logger
from qwen_compress.utils.seed import set_seed

_logger = get_logger(__name__)


def _capture_sparsity_masks(model: nn.Module) -> List[tuple]:
    """Snapshot zero-position masks for any Linear with non-trivial sparsity.

    Must be called BEFORE :func:`prepare_qat_model` wraps the Linears in
    ``QuantizedLinear`` (after wrapping, ``get_linear_layers_in_block`` no
    longer finds them by ``isinstance(nn.Linear)``).

    Returns a list of ``(linear_module, boolean_mask)`` pairs. Empty if the
    model has no sparsity.
    """
    pairs: List[tuple] = []
    layers = get_decoder_layers(model)
    total_zeros = 0
    total_elems = 0
    for block in layers:
        for _, linear in get_linear_layers_in_block(block).items():
            w = linear.weight.data
            zeros = int((w == 0).sum().item())
            total_zeros += zeros
            total_elems += w.numel()
            if zeros > 0:
                pairs.append((linear, (w != 0).to(torch.bool)))
    if total_zeros > 0 and total_elems > 0:
        _logger.info(
            f"Detected pre-existing sparsity: {total_zeros / total_elems * 100:.1f}% "
            f"zeros across {len(pairs)} Linear layers. Will preserve via mask "
            f"re-application after every QAT optimizer step."
        )
    return pairs


def _reapply_sparsity_masks(pairs: List[tuple]) -> None:
    """Re-zero the captured positions on each Linear's weight tensor."""
    for linear, mask in pairs:
        linear.weight.data.mul_(mask.to(linear.weight.dtype))


class _LayerCapture:
    def __init__(self) -> None:
        self.last: Optional[torch.Tensor] = None
        self._h = None

    def attach(self, layer: nn.Module) -> None:
        def _hook(_m, _i, o):  # noqa: ANN001
            self.last = o[0] if isinstance(o, tuple) else o
        self._h = layer.register_forward_hook(_hook)

    def detach(self) -> None:
        if self._h is not None:
            self._h.remove()
            self._h = None


def _scheduler(opt, total_steps: int, warmup_ratio: float, min_lr_ratio: float, name: str):
    warmup = max(1, int(total_steps * warmup_ratio))

    def cosine(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * prog)))

    def linear(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return max(min_lr_ratio, 1.0 - prog)

    if name in ("cosine_with_warmup", "cosine"):
        return LambdaLR(opt, cosine)
    if name == "linear":
        return LambdaLR(opt, linear)
    return LambdaLR(opt, lambda _s: 1.0)


class QADTrainer:
    """Quantization-Aware Distillation trainer."""

    def __init__(self, config: QATConfig) -> None:
        self.config = config
        set_seed(config.training.seed)

        dtype = "bf16" if config.training.bf16 else ("fp16" if config.training.fp16 else "fp32")

        # ---- Student (the model we quantize) -------------------------------
        _logger.info(f"Loading student model: {config.model_name_or_path}")
        self.student = load_qwen_model(
            config.model_name_or_path,
            dtype=dtype,
            device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
            attn_implementation="sdpa",
            gradient_checkpointing=config.training.gradient_checkpointing,
        )
        self.tokenizer = load_qwen_tokenizer(config.model_name_or_path)

        # ---- Capture sparsity masks BEFORE wrapping Linears with FakeQuant.
        # If the student came from a pruning stage, these masks must be
        # re-applied after every optimizer step or QAT will fill the zeros.
        self._sparsity_masks: List[tuple] = _capture_sparsity_masks(self.student)

        # ---- Teacher (optional) -------------------------------------------
        self.teacher: Optional[nn.Module] = None
        if config.use_qad and config.teacher_model_name_or_path:
            _logger.info(f"Loading teacher (frozen): {config.teacher_model_name_or_path}")
            self.teacher = load_qwen_model(
                config.teacher_model_name_or_path,
                dtype=dtype,
                device_map="auto",
                attn_implementation="sdpa",
            )
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

        # ---- Insert FakeQuant nodes ---------------------------------------
        spec = QuantSpec(
            weight_bits=config.weight_bits,
            activation_bits=config.activation_bits,
            weight_granularity=config.weight_granularity,
            activation_granularity=config.activation_granularity,
            weight_symmetric=config.weight_symmetric,
            activation_symmetric=config.activation_symmetric,
            learnable_weight_scale=False,
        )
        prepare_qat_model(
            self.student,
            spec=spec,
            skip_layers=config.skip_layers,
            quantize_kv_cache=config.quantize_kv_cache,
        )

        # ---- Activation calibration ---------------------------------------
        calib_loader = build_calibration_loader(
            path=config.calibration_path,
            tokenizer=self.tokenizer,
            nsamples=config.nsamples_calib,
            seq_length=config.data.max_seq_length,
            seed=config.training.seed,
            batch_size=1,
        )
        device = next(self.student.parameters()).device
        calibrate_model(
            self.student,
            calibration_iter=calib_loader,
            method=config.calibration_method,
            percentile=config.percentile,
            device=device,
        )

        # ---- QAD loss ------------------------------------------------------
        if self.teacher is not None:
            t_info = inspect_qwen_model(self.teacher)
            s_info = inspect_qwen_model(self.student)
            # Use one anchor per ~4 student layers, capped at 8 (legacy mode).
            num_anchors = min(8, max(1, s_info.num_hidden_layers // 4))
            self.assignment = build_group_assignment_legacy(
                t_info.num_hidden_layers,
                s_info.num_hidden_layers,
                num_groups=num_anchors,
                strategy="uniform",
            )
            self.loss_fn = DistillationLoss(
                student_hidden_size=s_info.hidden_size,
                teacher_hidden_size=t_info.hidden_size,
                teacher_group_reps=None,  # legacy QAT mode (hidden cosine only)
                alpha_ce=config.alpha_ce,
                beta_kd=config.beta_kd,
                gamma_hidden=config.gamma_hidden,
                delta_attn=0.0,
                kd_temperature=config.kd_temperature,
            )
            self.loss_fn.to(device)

            self._teacher_caps: List[_LayerCapture] = []
            self._student_caps: List[_LayerCapture] = []
            t_layers = get_decoder_layers(self.teacher)
            s_layers = get_decoder_layers(self.student)
            for t_idx, s_idx in self.assignment.pairs():
                tc = _LayerCapture()
                tc.attach(t_layers[t_idx])
                sc = _LayerCapture()
                sc.attach(s_layers[s_idx])
                self._teacher_caps.append(tc)
                self._student_caps.append(sc)
        else:
            self.loss_fn = None  # type: ignore[assignment]
            self.assignment = None  # type: ignore[assignment]

        # ---- Data ----------------------------------------------------------
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

        # ---- Optimizer -----------------------------------------------------
        trainable = [p for p in self.student.parameters() if p.requires_grad]
        if self.loss_fn is not None:
            trainable.extend(p for p in self.loss_fn.parameters() if p.requires_grad)
        opt = config.training.optimizer
        self.optimizer = AdamW(
            trainable, lr=opt.lr, betas=opt.betas, eps=opt.eps, weight_decay=opt.weight_decay
        )

        steps_per_epoch = max(1, len(self.train_loader) // config.data.gradient_accumulation_steps)
        self.total_steps = (
            config.training.max_steps
            if config.training.max_steps > 0
            else max(1, int(steps_per_epoch * config.training.num_train_epochs))
        )
        self.scheduler = _scheduler(
            self.optimizer,
            total_steps=self.total_steps,
            warmup_ratio=config.training.scheduler.warmup_ratio,
            min_lr_ratio=config.training.scheduler.min_lr_ratio,
            name=config.training.scheduler.name,
        )
        self._amp = torch.bfloat16 if config.training.bf16 else (torch.float16 if config.training.fp16 else None)
        self._scaler = torch.cuda.amp.GradScaler() if config.training.fp16 else None

    def _autocast_ctx(self):
        if self._amp is not None and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=self._amp)
        from contextlib import nullcontext
        return nullcontext()

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(self.student.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        s_out = self.student(input_ids=input_ids, attention_mask=attn, use_cache=False)

        if self.teacher is None or self.loss_fn is None:
            shift_logits = s_out.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            return {"total": ce, "ce": ce}

        with torch.no_grad():
            t_out = self.teacher(input_ids=input_ids, attention_mask=attn, use_cache=False)

        s_hidden = [c.last.to(device) for c in self._student_caps if c.last is not None]  # type: ignore[union-attr]
        t_hidden = [c.last.to(device) for c in self._teacher_caps if c.last is not None]  # type: ignore[union-attr]

        loss_out = self.loss_fn(
            student_logits=s_out.logits,
            teacher_logits=t_out.logits.to(device),
            labels=labels,
            attention_mask=attn,
            student_hidden_states=s_hidden,
            teacher_hidden_states=t_hidden,
        )
        return {"total": loss_out.total, **loss_out.breakdown}

    def train(self) -> Path:
        cfg = self.config
        global_step = 0
        accumulated = 0
        running: Dict[str, float] = {}
        t0 = time.time()
        output_dir = Path(cfg.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        _logger.info(
            f"Starting QAT/QAD: total_steps={self.total_steps}, "
            f"teacher_supervision={'on' if self.teacher is not None else 'off'}, "
            f"w{cfg.weight_bits}a{cfg.activation_bits}"
        )

        while global_step < self.total_steps:
            for batch in self.train_loader:
                if global_step >= self.total_steps:
                    break
                with self._autocast_ctx():
                    losses = self._compute_loss(batch)
                    loss = losses["total"] / cfg.data.gradient_accumulation_steps
                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                else:
                    loss.backward()
                accumulated += 1
                for k, v in losses.items():
                    running[k] = running.get(k, 0.0) + float(v.detach())

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
                    # Preserve pre-existing sparsity (no-op if model was dense).
                    _reapply_sparsity_masks(self._sparsity_masks)
                    accumulated = 0
                    global_step += 1

                    if global_step % cfg.training.logging_steps == 0 and is_main_process():
                        denom = cfg.training.logging_steps * cfg.data.gradient_accumulation_steps
                        avg = {k: v / denom for k, v in running.items()}
                        lr = self.scheduler.get_last_lr()[0]
                        dt = time.time() - t0
                        _logger.info(
                            f"[qat] step={global_step}/{self.total_steps} lr={lr:.3e} "
                            f"elapsed={dt:.0f}s | " + " ".join(f"{k}={v:.4f}" for k, v in avg.items())
                        )
                        running.clear()

                    if global_step % cfg.training.save_steps == 0 and is_main_process():
                        self._save(output_dir, f"step-{global_step}")
                        rotate_checkpoints(output_dir, keep=cfg.training.save_total_limit, prefix="step-")

        final = self._save(output_dir, "final")
        _logger.info(f"QAT complete. Final checkpoint: {final}")
        return final

    def _save(self, output_dir: Path, name: str) -> Path:
        target = output_dir / name
        save_compressed_model(
            self.student,
            self.tokenizer,
            target,
            extra_meta={
                "stage": "qat",
                "weight_bits": self.config.weight_bits,
                "activation_bits": self.config.activation_bits,
                "weight_granularity": self.config.weight_granularity,
                "activation_granularity": self.config.activation_granularity,
                "teacher": self.config.teacher_model_name_or_path,
                "calibration_method": self.config.calibration_method,
                "quantize_kv_cache": self.config.quantize_kv_cache,
            },
        )
        return target
