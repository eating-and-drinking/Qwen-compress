# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Recovery fine-tuning after SparseGPT pruning.

A short SFT pass that respects the pruning mask: gradients are computed
normally, but after each optimizer step we re-apply the binary mask so the
sparsity pattern is preserved. This is the cheapest way to recover most of
the perplexity lost to one-shot pruning.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from qwen_compress.data.cot_dataset import CoTDataset, DataCollatorForCoT
from qwen_compress.models.qwen_wrapper import (
    get_decoder_layers,
    get_linear_layers_in_block,
    load_qwen_tokenizer,
)
from qwen_compress.utils.checkpoint import save_compressed_model
from qwen_compress.utils.config import TrainingConfig, DataConfig
from qwen_compress.utils.logging import get_logger
from qwen_compress.utils.seed import set_seed

_logger = get_logger(__name__)


def _capture_masks(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Snapshot ``{name: boolean mask}`` for every Linear in every decoder block."""
    masks: Dict[str, torch.Tensor] = {}
    for layer_idx, block in enumerate(get_decoder_layers(model)):
        for name, linear in get_linear_layers_in_block(block).items():
            masks[f"layers.{layer_idx}.{name}"] = (linear.weight.data != 0).to(torch.bool)
    return masks


def _apply_masks(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """Re-zero the pruned positions."""
    for layer_idx, block in enumerate(get_decoder_layers(model)):
        for name, linear in get_linear_layers_in_block(block).items():
            key = f"layers.{layer_idx}.{name}"
            m = masks.get(key)
            if m is not None:
                linear.weight.data.mul_(m.to(linear.weight.dtype))


def _scheduler(opt, total_steps: int, warmup_ratio: float, min_lr_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))

    def fn(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(opt, fn)


def run_recovery_finetune(
    model: nn.Module,
    tokenizer_path_or_obj,
    data_cfg: DataConfig,
    training_cfg: TrainingConfig,
    output_dir: Path,
    extra_meta: Optional[Dict[str, object]] = None,
) -> Path:
    """Mask-preserving recovery fine-tune.

    Returns the final checkpoint directory.
    """
    set_seed(training_cfg.seed)
    device = next(model.parameters()).device

    tokenizer = (
        load_qwen_tokenizer(tokenizer_path_or_obj)
        if isinstance(tokenizer_path_or_obj, (str, Path))
        else tokenizer_path_or_obj
    )

    train_ds = CoTDataset(
        path=data_cfg.train_path,
        tokenizer=tokenizer,
        max_seq_length=data_cfg.max_seq_length,
        mode=data_cfg.cot_mode,
        seed=data_cfg.shuffle_seed,
    )
    collator = DataCollatorForCoT(tokenizer)
    loader = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    masks = _capture_masks(model)
    _logger.info(f"Captured {len(masks)} pruning masks for preservation.")

    model.train()
    if training_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    opt_cfg = training_cfg.optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=opt_cfg.lr,
        betas=opt_cfg.betas,
        eps=opt_cfg.eps,
        weight_decay=opt_cfg.weight_decay,
    )
    steps_per_epoch = max(1, len(loader) // data_cfg.gradient_accumulation_steps)
    total_steps = (
        training_cfg.max_steps
        if training_cfg.max_steps > 0
        else max(1, int(steps_per_epoch * training_cfg.num_train_epochs))
    )
    scheduler = _scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=training_cfg.scheduler.warmup_ratio,
        min_lr_ratio=training_cfg.scheduler.min_lr_ratio,
    )
    amp_dtype = torch.bfloat16 if training_cfg.bf16 else (torch.float16 if training_cfg.fp16 else None)
    scaler = torch.cuda.amp.GradScaler() if training_cfg.fp16 else None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    accumulated = 0
    running_loss = 0.0
    t0 = time.time()
    _logger.info(f"Starting recovery FT: total_steps={total_steps}")

    while global_step < total_steps:
        for batch in loader:
            if global_step >= total_steps:
                break
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_dtype is not None and torch.cuda.is_available()
                else _NullCtx()
            )
            with ctx:
                out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
                logits = out.logits[..., :-1, :].contiguous()
                tgt = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-100
                ) / data_cfg.gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accumulated += 1
            running_loss += float(loss.detach()) * data_cfg.gradient_accumulation_steps

            if accumulated >= data_cfg.gradient_accumulation_steps:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    training_cfg.max_grad_norm,
                )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                # >>> Mask re-application: this is what makes recovery sparsity-preserving.
                _apply_masks(model, masks)
                accumulated = 0
                global_step += 1

                if global_step % training_cfg.logging_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    dt = time.time() - t0
                    _logger.info(
                        f"[recovery] step={global_step}/{total_steps} "
                        f"lr={lr:.3e} loss={running_loss / training_cfg.logging_steps:.4f} "
                        f"elapsed={dt:.0f}s"
                    )
                    running_loss = 0.0

                # Periodic checkpoint saves.
                if global_step % training_cfg.save_steps == 0:
                    ckpt_dir = output_dir / f"step-{global_step}"
                    save_compressed_model(model, tokenizer, ckpt_dir, extra_meta=extra_meta)
                    _logger.info(f"[recovery] Saved checkpoint: {ckpt_dir}")

    final = output_dir / "final"
    save_compressed_model(model, tokenizer, final, extra_meta=extra_meta)
    _logger.info(f"Recovery FT done. Saved to {final}")
    return final


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
