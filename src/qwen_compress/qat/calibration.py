# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Activation calibration: prime every ``FakeQuantize`` with realistic statistics.

Workflow:
1. Switch all relevant ``FakeQuantize`` modules into calibration mode.
2. Forward a batch of calibration sequences through the model.
3. Aggregate observations and compute scale/zero-point per module.

Weight quantizers are initialised lazily in ``QuantizedLinear.forward``, so
this routine only needs to handle activation quantizers.
"""

from __future__ import annotations

from typing import Iterable, Literal

import torch
from torch import nn
from tqdm import tqdm

from qwen_compress.qat.fake_quant import FakeQuantize
from qwen_compress.utils.logging import get_logger

_logger = get_logger(__name__)


def _all_fake_quants(model: nn.Module) -> list[FakeQuantize]:
    return [m for m in model.modules() if isinstance(m, FakeQuantize)]


@torch.no_grad()
def calibrate_model(
    model: nn.Module,
    calibration_iter: Iterable,
    method: Literal["minmax", "percentile", "mse", "entropy"] = "percentile",
    percentile: float = 99.99,
    device: str | torch.device = "cuda",
) -> None:
    """Calibrate every ``FakeQuantize`` in ``model`` in place.

    Parameters
    ----------
    model:
        Model already prepared with :func:`prepare_qat_model`.
    calibration_iter:
        An iterable yielding either ``torch.Tensor`` of input ids or dict-like
        batches with an ``input_ids`` key.
    method:
        Calibration objective. ``"entropy"`` is not yet implemented and falls
        back to ``"percentile"``.
    """
    if method == "entropy":
        _logger.warning("entropy calibration not implemented, using percentile")
        method = "percentile"

    quants = _all_fake_quants(model)
    if not quants:
        _logger.warning("No FakeQuantize modules found — did you call prepare_qat_model?")
        return

    _logger.info(f"Calibrating {len(quants)} fake-quant modules (method={method})")
    for q in quants:
        # Skip weight quantizers (already initialised) — they have initialized==True.
        if bool(q.initialized.item()):
            continue
        q.start_calibration(method=method, percentile=percentile)

    model.eval()
    for batch in tqdm(calibration_iter, desc="Calibration"):
        if isinstance(batch, torch.Tensor):
            input_ids = batch.to(device)
        else:
            input_ids = batch["input_ids"].to(device)
        model(input_ids=input_ids, use_cache=False)

    for q in quants:
        if q._calibrating:  # noqa: SLF001 - intentional internal flag check
            q.finish_calibration()

    _logger.info("Calibration finished.")
