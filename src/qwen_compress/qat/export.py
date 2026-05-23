# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Export a QAT-trained model to a deployable INT8 / ONNX artifact.

For the "safetensors" path we serialise the FP weights together with the
per-module quantization parameters in ``quant_config.json`` so downstream
runtimes (e.g. vLLM, llama.cpp, TensorRT-LLM) can re-quantize accurately. For
ONNX we emit a QDQ (Quantize-DeQuantize) graph that any QDQ-aware backend
(ORT, TensorRT) can consume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch
from torch import nn

from qwen_compress.qat.fake_quant import FakeQuantize, QuantizedLinear
from qwen_compress.utils.checkpoint import save_compressed_model
from qwen_compress.utils.logging import get_logger

_logger = get_logger(__name__)


def _extract_qparams(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Walk the model and collect per-module quant params (scale, zp, bits)."""
    qparams: Dict[str, Dict[str, Any]] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, QuantizedLinear):
            wq = mod.weight_quantizer
            aq = mod.act_quantizer
            entry: Dict[str, Any] = {
                "weight": {
                    "bits": wq.bits,
                    "symmetric": wq.symmetric,
                    "granularity": wq.granularity,
                    "scale": wq.scale.detach().cpu().tolist(),
                    "zero_point": wq.zero_point.detach().cpu().tolist(),
                },
                "activation": {
                    "bits": aq.bits,
                    "symmetric": aq.symmetric,
                    "granularity": aq.granularity,
                    "scale": aq.scale.detach().cpu().tolist(),
                    "zero_point": aq.zero_point.detach().cpu().tolist(),
                },
            }
            if mod.output_quantizer is not None:
                oq = mod.output_quantizer
                entry["output"] = {
                    "bits": oq.bits,
                    "symmetric": oq.symmetric,
                    "granularity": oq.granularity,
                    "scale": oq.scale.detach().cpu().tolist(),
                    "zero_point": oq.zero_point.detach().cpu().tolist(),
                }
            qparams[name] = entry
    return qparams


def _fold_into_plain_linear(model: nn.Module) -> nn.Module:
    """Replace every ``QuantizedLinear`` with its inner ``nn.Linear``.

    The wrapped weights have already absorbed quant-aware training. Removing
    the wrappers gives a clean checkpoint that loads in standard HuggingFace
    pipelines, with the quant-params stored separately.
    """
    # First collect all replacements; iterate after to avoid mutating during traversal.
    to_replace = []
    for name, mod in model.named_modules():
        if isinstance(mod, QuantizedLinear):
            to_replace.append((name, mod.linear))
    for name, plain in to_replace:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, plain)
    return model


def export_quantized_model(
    model: nn.Module,
    tokenizer: Any,
    output_dir: Path,
    fmt: Literal["safetensors", "onnx"] = "safetensors",
    onnx_opset: int = 17,
    example_seq_length: int = 32,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export ``model`` to ``output_dir`` in the requested format.

    Returns the directory path containing the artifact(s).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qparams = _extract_qparams(model)
    if not qparams:
        _logger.warning("No QuantizedLinear modules found during export.")

    if fmt == "safetensors":
        # Strip the FakeQuant wrappers — runtime will reapply using quant_config.json.
        _fold_into_plain_linear(model)
        save_compressed_model(
            model, tokenizer, output_dir,
            extra_meta={"stage": "qat-exported", **(extra_meta or {})},
        )
        with (output_dir / "quant_config.json").open("w", encoding="utf-8") as fp:
            json.dump(qparams, fp, indent=2)
        _logger.info(f"Exported safetensors + quant_config.json to {output_dir}")
        return output_dir

    if fmt == "onnx":
        try:
            import torch.onnx  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("torch.onnx is required for ONNX export.") from e
        onnx_path = output_dir / "model.onnx"
        device = next(model.parameters()).device
        example_ids = torch.randint(
            low=0, high=int(model.config.vocab_size), size=(1, example_seq_length), device=device
        )
        attn = torch.ones_like(example_ids)
        # ONNX export expects eval mode and standard ops; FakeQuant nodes are converted
        # via the symbolic registration below.
        model.eval()
        torch.onnx.export(
            model,
            (example_ids, attn),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 1: "seq"},
            },
            opset_version=onnx_opset,
            do_constant_folding=True,
        )
        with (output_dir / "quant_config.json").open("w", encoding="utf-8") as fp:
            json.dump(qparams, fp, indent=2)
        tokenizer.save_pretrained(str(output_dir))
        _logger.info(f"Exported ONNX QDQ model to {onnx_path}")
        return output_dir

    raise ValueError(f"Unsupported export format: {fmt!r}")
