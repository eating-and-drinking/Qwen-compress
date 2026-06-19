# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Qwen model loading and introspection.

Supports Qwen, Qwen1.5, Qwen2, and Qwen2.5 series (all the ``Qwen*ForCausalLM``
variants exposed by ``transformers``). The compression stages depend on three
facts about Qwen's architecture:

* Decoder layers live at ``model.model.layers`` (a ``nn.ModuleList``).
* Each decoder block contains four linear projections in self-attention
  (``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``) and three in the MLP
  (``gate_proj``, ``up_proj``, ``down_proj``).
* The final ``lm_head`` and ``embed_tokens`` are tied in some variants.

We never hardcode these names elsewhere — go through :class:`QwenModelInfo`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from qwen_compress.utils.logging import get_logger

_logger = get_logger(__name__)

# Linear submodules inside a Qwen decoder block, in canonical order.
QWEN_ATTN_LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
QWEN_MLP_LINEARS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
QWEN_ALL_LINEARS = QWEN_ATTN_LINEARS + QWEN_MLP_LINEARS


@dataclass
class QwenModelInfo:
    """Architecture metadata for a loaded Qwen model."""

    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    tie_word_embeddings: bool
    model_type: str
    linear_block_names: List[str] = field(default_factory=lambda: list(QWEN_ALL_LINEARS))


def _resolve_dtype(dtype: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype string {dtype!r}")
    return mapping[dtype]


def load_qwen_model(
    name_or_path: Union[str, Path],
    dtype: Union[str, torch.dtype, None] = "bf16",
    device_map: Optional[Union[str, Dict[str, int]]] = None,
    trust_remote_code: bool = True,
    attn_implementation: Optional[str] = None,
    gradient_checkpointing: bool = False,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> nn.Module:
    """Load a Qwen causal-LM model.

    Parameters
    ----------
    name_or_path:
        HuggingFace repo id or local path.
    dtype:
        Computation dtype (``"bf16"``, ``"fp16"``, ``"fp32"`` or a ``torch.dtype``).
        Ignored when ``load_in_8bit`` or ``load_in_4bit`` is True.
    device_map:
        Passed straight to ``transformers``. Use ``"auto"`` for multi-GPU shards.
    trust_remote_code:
        Required for some Qwen variants that ship custom modeling files.
    attn_implementation:
        ``"sdpa"``, ``"flash_attention_2"``, or ``"eager"``. ``None`` lets HF choose.
    gradient_checkpointing:
        Enable activation checkpointing on the loaded model.
    load_in_8bit:
        Load weights in INT8 via bitsandbytes (LLM.int8()). Recommended for a
        frozen teacher on a 48 GB GPU: reduces teacher footprint from ~29 GB
        (BF16) to ~15 GB with negligible effect on logits / hidden states.
        Requires ``bitsandbytes`` to be installed.
    load_in_4bit:
        Load weights in NF4 via bitsandbytes (QLoRA-style). Reduces teacher
        footprint to ~7 GB. Higher quantization error than INT8; use only when
        VRAM is very constrained (< 40 GB for 14B→3B).
        Requires ``bitsandbytes`` to be installed.
    """
    if load_in_8bit and load_in_4bit:
        raise ValueError("load_in_8bit and load_in_4bit are mutually exclusive.")

    torch_dtype = _resolve_dtype(dtype)
    kwargs: Dict[str, object] = {"trust_remote_code": trust_remote_code}

    if load_in_8bit or load_in_4bit:
        # bitsandbytes quantization — torch_dtype is set inside the BnB config
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "bitsandbytes is required for INT8/INT4 loading. "
                "Install it with: pip install bitsandbytes"
            ) from e

        if load_in_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            _logger.info(f"Loading Qwen model from {name_or_path} (INT8 via bitsandbytes)")
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            _logger.info(f"Loading Qwen model from {name_or_path} (NF4 via bitsandbytes)")
        # device_map must be set for bitsandbytes multi-device loading
        kwargs["device_map"] = device_map if device_map is not None else "auto"
    else:
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            kwargs["device_map"] = device_map
        _logger.info(f"Loading Qwen model from {name_or_path} (dtype={dtype})")

    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(str(name_or_path), **kwargs)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    return model


def load_qwen_tokenizer(
    name_or_path: Union[str, Path],
    trust_remote_code: bool = True,
    padding_side: str = "right",
) -> PreTrainedTokenizerBase:
    """Load the matching tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        str(name_or_path),
        trust_remote_code=trust_remote_code,
        padding_side=padding_side,
    )
    if tokenizer.pad_token is None:
        # Qwen tokenizers ship an EOS/EOD but no PAD by default.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def inspect_qwen_model(model: nn.Module) -> QwenModelInfo:
    """Extract architecture metadata from a loaded HF model."""
    cfg = model.config

    return QwenModelInfo(
        num_hidden_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        vocab_size=cfg.vocab_size,
        tie_word_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)),
        model_type=str(getattr(cfg, "model_type", "qwen")),
    )


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the ``ModuleList`` of decoder blocks."""
    # ``model.model.layers`` is the conventional path for Qwen / Qwen2 / Qwen2.5.
    base = getattr(model, "model", None)
    if base is None or not hasattr(base, "layers"):
        raise AttributeError(
            "Could not locate decoder layers. Expected `model.model.layers` "
            "(Qwen2 convention)."
        )
    return base.layers


def get_linear_layers_in_block(block: nn.Module) -> Dict[str, nn.Linear]:
    """Return the named ``nn.Linear`` submodules of a single decoder block.

    The keys are dotted names relative to the block (e.g. ``"self_attn.q_proj"``).
    """
    result: Dict[str, nn.Linear] = {}
    for dotted in QWEN_ALL_LINEARS:
        mod: nn.Module = block
        for part in dotted.split("."):
            if not hasattr(mod, part):
                mod = None  # type: ignore[assignment]
                break
            mod = getattr(mod, part)
        if isinstance(mod, nn.Linear):
            result[dotted] = mod
    return result
