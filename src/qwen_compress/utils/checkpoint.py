# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Checkpoint I/O for compressed models.

Uses ``safetensors`` for weight serialisation (faster + safer than ``torch.save``)
and JSON sidecar files for non-tensor state (optimizer step, quant params, etc.).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from safetensors.torch import load_file as _st_load
from safetensors.torch import save_file as _st_save

from qwen_compress.utils.logging import get_logger

_logger = get_logger(__name__)


def _atomic_replace(src: Path, dst: Path) -> None:
    """Move ``src`` to ``dst`` atomically, removing an existing ``dst`` first."""
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    src.replace(dst)


def save_checkpoint(
    state: Dict[str, Any],
    output_dir: Union[str, Path],
    name: str = "checkpoint",
    weights_key: str = "model",
) -> Path:
    """Save a checkpoint atomically.

    Tensors under ``state[weights_key]`` go into ``model.safetensors``; everything
    else is JSON-encoded as ``meta.json``.

    Returns the final checkpoint directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / name
    tmp_dir = output_dir / f"{name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    weights = state.pop(weights_key, None)
    if weights is not None:
        # safetensors requires contiguous tensors on CPU.
        weights = {k: v.detach().cpu().contiguous() for k, v in weights.items()}
        _st_save(weights, str(tmp_dir / "model.safetensors"))

    if state:
        with (tmp_dir / "meta.json").open("w", encoding="utf-8") as fp:
            json.dump(state, fp, indent=2, default=str)

    _atomic_replace(tmp_dir, final_dir)
    _logger.info(f"Saved checkpoint to {final_dir}")
    return final_dir


def load_checkpoint(
    checkpoint_dir: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = "cpu",
) -> Dict[str, Any]:
    """Inverse of :func:`save_checkpoint`."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(checkpoint_dir)

    state: Dict[str, Any] = {}
    weights_path = checkpoint_dir / "model.safetensors"
    if weights_path.exists():
        state["model"] = _st_load(str(weights_path), device=str(map_location or "cpu"))

    meta_path = checkpoint_dir / "meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fp:
            state.update(json.load(fp))

    return state


def rotate_checkpoints(output_dir: Union[str, Path], keep: int, prefix: str = "step-") -> None:
    """Keep at most ``keep`` checkpoints matching ``prefix`` under ``output_dir``."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return
    matches = sorted(
        (p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)),
        key=lambda p: p.stat().st_mtime,
    )
    excess = len(matches) - keep
    for old in matches[: max(0, excess)]:
        _logger.info(f"Rotating out old checkpoint: {old}")
        shutil.rmtree(old, ignore_errors=True)


def save_compressed_model(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: Union[str, Path],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a HuggingFace-compatible compressed model directory.

    Writes ``model.safetensors``, ``config.json``, tokenizer files, and an
    optional ``compression_meta.json`` describing the applied stages.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # HF API handles config + weights together.
    model.save_pretrained(str(output_dir), safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_dir))

    if extra_meta is not None:
        with (output_dir / "compression_meta.json").open("w", encoding="utf-8") as fp:
            json.dump(extra_meta, fp, indent=2, default=str)

    _logger.info(f"Saved compressed model to {output_dir}")
    return output_dir
