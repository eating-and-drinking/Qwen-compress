# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Chain-of-Thought (CoT) supervised fine-tuning dataset.

Expected JSONL schema (one example per line)::

    {
      "instruction": "...",
      "input": "...",            # optional
      "chain_of_thought": "...", # the reasoning trace
      "answer": "..."            # the final answer
    }

Three training modes are supported:

* ``direct``  : target = answer (no CoT shown to student)
* ``cot``     : target = CoT + answer (student must produce reasoning)
* ``dual``    : alternating sampling of ``direct`` and ``cot`` per epoch,
                preserving both fast inference and reasoning ability — this is
                the mode used in our distillation pipeline.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from qwen_compress.utils.logging import get_logger

_logger = get_logger(__name__)

# Tokens used to format direct vs. CoT outputs.
COT_OPEN = "<think>"
COT_CLOSE = "</think>"


def _read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a (possibly large) JSONL file into memory."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def _format_prompt(example: Dict[str, Any]) -> str:
    """Render the input side using Qwen chat-style markers."""
    instruction = example["instruction"]
    user_input = example.get("input", "").strip()
    if user_input:
        user_msg = f"{instruction}\n\n{user_input}"
    else:
        user_msg = instruction
    return (
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _format_target(
    example: Dict[str, Any], mode: Literal["direct", "cot"]
) -> str:
    """Render the target side."""
    answer = example["answer"]
    if mode == "direct" or not example.get("chain_of_thought"):
        return f"{answer}<|im_end|>"
    cot = example["chain_of_thought"]
    return f"{COT_OPEN}\n{cot}\n{COT_CLOSE}\n{answer}<|im_end|>"


class CoTDataset(Dataset):
    """In-memory CoT-SFT dataset producing token IDs + labels with prompt masking.

    Parameters
    ----------
    path:
        JSONL file path (see module docstring for schema).
    tokenizer:
        Tokenizer used to encode prompt/target.
    max_seq_length:
        Hard truncation length. Examples exceeding this are right-truncated and
        their loss-mask is recomputed accordingly.
    mode:
        ``"direct"``, ``"cot"``, or ``"dual"`` (alternates between the two).
    seed:
        Seed governing ``dual`` mode sampling.
    """

    def __init__(
        self,
        path: Union[str, Path],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 2048,
        mode: Literal["direct", "cot", "dual"] = "dual",
        seed: int = 42,
    ) -> None:
        self.examples = _read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode
        self._rng = random.Random(seed)
        _logger.info(f"Loaded {len(self.examples)} examples from {path} (mode={mode})")

    def __len__(self) -> int:
        return len(self.examples)

    def _pick_mode(self, idx: int) -> Literal["direct", "cot"]:
        if self.mode != "dual":
            return self.mode  # type: ignore[return-value]
        # Deterministic per-index choice so re-iterating yields the same labels.
        return "cot" if (idx + self._rng.randint(0, 1)) % 2 == 0 else "direct"

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        mode = self._pick_mode(idx)
        prompt = _format_prompt(ex)
        target = _format_target(ex, mode)

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        target_ids = self.tokenizer(target, add_special_tokens=False).input_ids

        # Truncate: prompt stays whole, target is right-truncated.
        total_len = len(prompt_ids) + len(target_ids)
        if total_len > self.max_seq_length:
            keep_target = max(1, self.max_seq_length - len(prompt_ids))
            target_ids = target_ids[:keep_target]
        input_ids = prompt_ids + target_ids
        # Loss mask: -100 on prompt positions, real ids on target positions.
        labels = ([-100] * len(prompt_ids)) + list(target_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }


@dataclass
class DataCollatorForCoT:
    """Right-pads a batch of ``CoTDataset`` items.

    Padding tokens get ``-100`` in ``labels`` so they do not contribute to loss.
    """

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].size(0) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define a pad_token_id.")

        input_ids = torch.full((len(features), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)

        for i, f in enumerate(features):
            L = f["input_ids"].size(0)
            input_ids[i, :L] = f["input_ids"]
            attention_mask[i, :L] = f["attention_mask"]
            labels[i, :L] = f["labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
