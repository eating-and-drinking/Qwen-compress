# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Calibration dataloaders for SparseGPT pruning and QAT activation calibration.

Calibration data should resemble the deployment distribution. For LLM
compression we feed a few hundred raw token-id sequences sampled from the
target-domain corpus (e.g. the CoT SFT data, with chat templates applied).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from qwen_compress.utils.logging import get_logger

_logger = get_logger(__name__)


class _CalibrationDataset(Dataset):
    def __init__(self, samples: Sequence[torch.Tensor]) -> None:
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._samples[idx]


def _stream_texts(path: Union[str, Path]) -> Iterator[str]:
    """Yield raw text strings from JSONL (``text`` or ``instruction``+``answer``)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".json"):
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, str):
                    yield obj
                elif "text" in obj:
                    yield obj["text"]
                else:
                    # Fallback: stitch instruction + answer.
                    parts = []
                    if "instruction" in obj:
                        parts.append(obj["instruction"])
                    if "input" in obj and obj["input"]:
                        parts.append(obj["input"])
                    if "chain_of_thought" in obj and obj["chain_of_thought"]:
                        parts.append(obj["chain_of_thought"])
                    if "answer" in obj:
                        parts.append(obj["answer"])
                    if parts:
                        yield "\n".join(parts)
    elif suffix == ".txt":
        with path.open("r", encoding="utf-8") as fp:
            buf: List[str] = []
            for line in fp:
                line = line.rstrip()
                if line:
                    buf.append(line)
                else:
                    if buf:
                        yield "\n".join(buf)
                        buf = []
            if buf:
                yield "\n".join(buf)
    else:
        raise ValueError(f"Unsupported calibration file format: {suffix}")


def build_calibration_loader(
    path: Union[str, Path],
    tokenizer: PreTrainedTokenizerBase,
    nsamples: int = 128,
    seq_length: int = 2048,
    seed: int = 42,
    batch_size: int = 1,
    device: Optional[Union[str, torch.device]] = None,
) -> DataLoader:
    """Build a ``DataLoader`` yielding ``[B, seq_length]`` token-id tensors.

    Texts shorter than ``seq_length`` are skipped; longer texts contribute one
    random window. Sampling is deterministic given ``seed``.
    """
    rng = random.Random(seed)
    pool: List[torch.Tensor] = []

    for text in _stream_texts(path):
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if ids.size(0) < seq_length:
            continue
        start = rng.randint(0, ids.size(0) - seq_length)
        window = ids[start : start + seq_length]
        if device is not None:
            window = window.to(device)
        pool.append(window)
        if len(pool) >= nsamples:
            break

    if len(pool) < nsamples:
        _logger.warning(
            f"Only {len(pool)} calibration samples extracted (requested {nsamples}); "
            "consider lowering `seq_length` or providing more data."
        )

    dataset = _CalibrationDataset(pool)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: torch.stack(batch, dim=0),
    )
