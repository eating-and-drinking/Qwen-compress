# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Group-wise distillation trainer.

The trainer:

1. Loads teacher (frozen, eval mode) and student (trainable) models.
2. Builds the group anchor assignment via :func:`build_group_assignment`.
3. Registers forward hooks on the chosen layers to capture hidden states.
4. Runs the standard SFT loop with a composite distillation loss.

It supports gradient accumulation, mixed precision, gradient checkpointing,
and DDP. For sharded multi-GPU teachers, pass ``device_map="auto"`` when
loading; the teacher is wrapped in ``no_grad``.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

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


class _LayerOutputCapture:
    """Forward hook that stashes the *output hidden state* of a decoder layer.

    Qwen decoder layers return a tuple ``(hidden_states, ...optional cache...)``.
    We always take element 0.
    """

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

    def constant(_step: int) -> float:
        return 1.0

    if name == "cosine_with_warmup" or name == "cosine":
        return LambdaLR(optimizer, cosine_with_warmup)
    if name == "linear":
        return LambdaLR(optimizer, linear)
    if name == "constant":
        return LambdaLR(optimizer, constant)
    raise ValueError(f"Unsupported scheduler {name!r}")


class GroupwiseDistillTrainer:
    """End-to-end orchestrator for group-wise distillation.

    Parameters
    ----------
    config:
        A validated :class:`DistillConfig`.
    """

    def __init__(self, config: DistillConfig) -> None:
        self.config = config
        set_seed(config.training.seed)

        # ---- Models --------------------------------------------------------
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

        # ---- Group assignment ---------------------------------------------
        t_info = inspect_qwen_model(self.teacher)
        s_info = inspect_qwen_model(self.student)
        self.assignment: GroupAssignment = build_group_assignment(
            teacher_num_layers=t_info.num_hidden_layers,
            student_num_layers=s_info.num_hidden_layers,
            num_groups=config.num_groups,
            strategy=config.group_strategy,
        )

        # ---- Loss ----------------------------------------------------------
        self.loss_fn = DistillationLoss(
            student_hidden_size=s_info.hidden_size,
            teacher_hidden_size=t_info.hidden_size,
            num_groups=self.assignment.num_groups,
            alpha_ce=config.alpha_ce,
            beta_kd=config.beta_kd,
            gamma_hidden=config.gamma_hidden,
            delta_attn=config.delta_attn,
            kd_temperature=config.kd_temperature,
        )
        # The hidden-state projectors live on the same device as the student.
        self.loss_fn.to(next(self.student.parameters()).device)

        # ---- Hooks ---------------------------------------------------------
        self._teacher_captures: List[_LayerOutputCapture] = []
        self._student_captures: List[_LayerOutputCapture] = []
        self._attach_hooks()

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

        # ---- Optimizer / scheduler ----------------------------------------
        trainable = [p for p in self.student.parameters() if p.requires_grad]
        trainable.extend(p for p in self.loss_fn.parameters() if p.requires_grad)
        opt = config.training.optimizer
        self.optimizer = AdamW(
            trainable,
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

        # ---- AMP -----------------------------------------------------------
        self._amp_dtype: Optional[torch.dtype] = None
        if config.training.bf16:
            self._amp_dtype = torch.bfloat16
        elif config.training.fp16:
            self._amp_dtype = torch.float16
        self._scaler = (
            torch.cuda.amp.GradScaler() if config.training.fp16 else None
        )

    # ------------------------------------------------------------------ hooks
    def _attach_hooks(self) -> None:
        teacher_layers = get_decoder_layers(self.teacher)
        student_layers = get_decoder_layers(self.student)
        for t_idx, s_idx in self.assignment.pairs():
            t_cap = _LayerOutputCapture()
            t_cap.attach(teacher_layers[t_idx])
            s_cap = _LayerOutputCapture()
            s_cap.attach(student_layers[s_idx])
            self._teacher_captures.append(t_cap)
            self._student_captures.append(s_cap)

    def _detach_hooks(self) -> None:
        for cap in self._teacher_captures + self._student_captures:
            cap.detach()
        self._teacher_captures.clear()
        self._student_captures.clear()

    # ------------------------------------------------------------------ train
    @contextmanager
    def _autocast(self) -> Iterator[None]:
        if self._amp_dtype is not None and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                yield
        else:
            yield

    def _forward_pair(
        self, batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Run teacher and student forward, returning logits + captured hidden states."""
        device = next(self.student.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            teacher_out = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        student_out = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        teacher_hidden = [c.last_output for c in self._teacher_captures]
        student_hidden = [c.last_output for c in self._student_captures]

        if any(h is None for h in teacher_hidden + student_hidden):
            raise RuntimeError(
                "A registered hook did not capture any output — verify the layer "
                "indices in the group assignment match the model."
            )

        # Teacher hidden states may be on a different device (sharded). Move
        # them to the student device for loss computation.
        teacher_hidden = [h.to(device) for h in teacher_hidden]
        return student_out.logits, teacher_out.logits.to(device), student_hidden, teacher_hidden

    def train(self) -> Path:
        """Run the full distillation loop. Returns the final checkpoint path."""
        cfg = self.config
        global_step = 0
        accumulated = 0
        running_loss: Dict[str, float] = {}
        t0 = time.time()
        output_dir = Path(cfg.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        _logger.info(
            f"Starting distillation: total_steps={self.total_steps}, "
            f"batch_size={cfg.data.batch_size}, "
            f"grad_accum={cfg.data.gradient_accumulation_steps}, "
            f"num_groups={self.assignment.num_groups}"
        )

        while global_step < self.total_steps:
            for batch in self.train_loader:
                if global_step >= self.total_steps:
                    break

                with self._autocast():
                    s_logits, t_logits, s_hidden, t_hidden = self._forward_pair(batch)
                    labels = batch["labels"].to(s_logits.device)
                    loss_out = self.loss_fn(
                        student_logits=s_logits,
                        teacher_logits=t_logits,
                        labels=labels,
                        student_hidden_states=s_hidden,
                        teacher_hidden_states=t_hidden,
                    )
                    loss = loss_out.total / cfg.data.gradient_accumulation_steps

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
                        [p for p in self.student.parameters() if p.requires_grad],
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

                    if (
                        global_step % cfg.training.save_steps == 0
                        and is_main_process()
                    ):
                        self._save(output_dir, f"step-{global_step}")
                        rotate_checkpoints(
                            output_dir, keep=cfg.training.save_total_limit, prefix="step-"
                        )

        # Final save.
        final_path = self._save(output_dir, "final")
        self._detach_hooks()
        _logger.info(f"Distillation complete. Final checkpoint: {final_path}")
        return final_path

    # ------------------------------------------------------------------ save
    def _save(self, output_dir: Path, name: str) -> Path:
        target = output_dir / name
        save_compressed_model(
            self.student,
            self.tokenizer,
            target,
            extra_meta={
                "stage": "distill",
                "teacher": self.config.teacher_model_name_or_path,
                "num_groups": self.assignment.num_groups,
                "teacher_anchors": self.assignment.teacher_anchor_layers,
                "student_targets": self.assignment.student_target_layers,
                "alpha_ce": self.config.alpha_ce,
                "beta_kd": self.config.beta_kd,
                "gamma_hidden": self.config.gamma_hidden,
                "kd_temperature": self.config.kd_temperature,
            },
        )
        return target
