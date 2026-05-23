# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Dataset tests using a small stub tokenizer (no model downloads)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from qwen_compress.data.cot_dataset import CoTDataset, DataCollatorForCoT


class _StubTokenizer:
    """Minimal tokenizer interface compatible with CoTDataset."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"

    def __call__(self, text: str, add_special_tokens: bool = False, **_):
        # Map every character to its ord() value (toy tokenizer).
        ids = [ord(c) % 1000 for c in text]

        class _Out:
            def __init__(self_inner, input_ids):
                self_inner.input_ids = input_ids

        return _Out(ids)


def _write_jsonl(records):
    fp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for r in records:
        fp.write(json.dumps(r) + "\n")
    fp.close()
    return Path(fp.name)


def test_loads_examples():
    path = _write_jsonl([
        {"instruction": "a", "input": "", "chain_of_thought": "b", "answer": "c"},
        {"instruction": "d", "input": "e", "chain_of_thought": "f", "answer": "g"},
    ])
    ds = CoTDataset(path, _StubTokenizer(), max_seq_length=64, mode="direct")
    assert len(ds) == 2
    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item
    assert item["input_ids"].dtype == torch.long


def test_prompt_masking():
    path = _write_jsonl([{"instruction": "hello", "answer": "world", "chain_of_thought": ""}])
    ds = CoTDataset(path, _StubTokenizer(), max_seq_length=64, mode="direct")
    item = ds[0]
    # All -100 positions should be the prompt prefix (some are guaranteed to be -100).
    assert (item["labels"] == -100).any()
    # Some target positions should carry real ids.
    assert (item["labels"] != -100).any()


def test_collator_pads_and_masks():
    path = _write_jsonl([
        {"instruction": "a", "answer": "b", "chain_of_thought": ""},
        {"instruction": "abcdef", "answer": "hello world", "chain_of_thought": ""},
    ])
    ds = CoTDataset(path, _StubTokenizer(), max_seq_length=64, mode="direct")
    collator = DataCollatorForCoT(_StubTokenizer(), pad_to_multiple_of=8)
    batch = collator([ds[0], ds[1]])
    assert batch["input_ids"].size(0) == 2
    assert batch["input_ids"].size(1) % 8 == 0
    # Pad positions in labels are -100.
    assert (batch["labels"] == -100).any()
