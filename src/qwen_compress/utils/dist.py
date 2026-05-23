# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Thin wrappers around ``torch.distributed`` so the rest of the codebase doesn't
need to special-case single-process training.

All functions are safe to call before ``torch.distributed`` is initialised; they
fall back to single-process semantics.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.distributed as dist


def is_initialized() -> bool:
    """Return ``True`` iff a process group has been initialised."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the current process rank (``0`` if non-distributed)."""
    return dist.get_rank() if is_initialized() else 0


def get_world_size() -> int:
    """Return the world size (``1`` if non-distributed)."""
    return dist.get_world_size() if is_initialized() else 1


def is_main_process() -> bool:
    """Return ``True`` iff this is rank 0."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronise across ranks. No-op outside of distributed training."""
    if is_initialized():
        dist.barrier()


def setup_distributed(backend: str = "nccl") -> None:
    """Initialise ``torch.distributed`` from standard env vars.

    Expects ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, ``MASTER_ADDR``,
    ``MASTER_PORT`` to be set (as ``torchrun`` does). Idempotent.
    """
    if is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        # Single-process execution — nothing to do.
        return

    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


@contextmanager
def main_process_first() -> Iterator[None]:
    """Context manager that lets rank 0 finish a block before others enter.

    Useful for one-shot dataset preprocessing or model downloads.
    """
    if not is_initialized():
        yield
        return

    if not is_main_process():
        dist.barrier()
    try:
        yield
    finally:
        if is_main_process():
            dist.barrier()
