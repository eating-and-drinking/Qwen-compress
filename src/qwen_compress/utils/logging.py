# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Structured logging with optional rank-aware silencing for distributed training.

Uses ``loguru`` as the backend so users get colorised, structured logs out of the box
while remaining a drop-in replacement for :mod:`logging` via :func:`get_logger`.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger as _loguru_logger


class _InterceptHandler(logging.Handler):
    """Route stdlib ``logging`` records through loguru."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin wrapper
        try:
            level = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    rank: int = 0,
    rank_zero_only: bool = True,
    json_logs: bool = False,
) -> None:
    """Configure global logging.

    Parameters
    ----------
    level:
        Minimum log level for stderr output (``DEBUG`` / ``INFO`` / ``WARNING`` / ...).
    log_file:
        Optional path for a rotating log file (10 MB rotation, 7 backups).
    rank:
        Current process rank in distributed training.
    rank_zero_only:
        If ``True`` (default), non-zero ranks log only at ``WARNING`` level to avoid
        terminal spam from N replicas.
    json_logs:
        Emit machine-parsable JSON lines instead of human-readable text.
    """
    _loguru_logger.remove()

    effective_level = level if (rank == 0 or not rank_zero_only) else "WARNING"

    if json_logs:
        fmt = (
            '{{"time":"{time:YYYY-MM-DDTHH:mm:ss.SSSZ}",'
            '"level":"{level}","name":"{name}","rank":' + str(rank) + ","
            '"message":{message!r}}}'
        )
        _loguru_logger.add(sys.stderr, level=effective_level, format=fmt, serialize=False)
    else:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            f"<cyan>rank={rank}</cyan> | "
            "<level>{level: <8}</level> | "
            "<yellow>{name}</yellow>:<yellow>{function}</yellow>:<yellow>{line}</yellow> | "
            "<level>{message}</level>"
        )
        _loguru_logger.add(sys.stderr, level=effective_level, format=fmt, colorize=True)

    if log_file is not None and rank == 0:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        _loguru_logger.add(
            str(log_file),
            level="DEBUG",
            rotation="10 MB",
            retention=7,
            compression="gz",
            enqueue=True,
        )

    # Route HuggingFace / PyTorch stdlib logs through loguru too.
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for noisy in ("transformers.tokenization_utils_base", "datasets", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str):  # noqa: ANN201 - loguru.Logger is private
    """Return a child logger bound with ``name`` for module-level use."""
    return _loguru_logger.bind(name=name)


def auto_configure_from_env() -> None:
    """Auto-configure logging from environment variables.

    Reads ``QC_LOG_LEVEL``, ``QC_LOG_FILE``, ``QC_LOG_JSON``, ``RANK``.
    Safe to call from library initialisation.
    """
    level = os.environ.get("QC_LOG_LEVEL", "INFO")
    log_file = os.environ.get("QC_LOG_FILE")
    json_logs = os.environ.get("QC_LOG_JSON", "0") == "1"
    rank = int(os.environ.get("RANK", "0"))
    configure_logging(level=level, log_file=log_file, rank=rank, json_logs=json_logs)
