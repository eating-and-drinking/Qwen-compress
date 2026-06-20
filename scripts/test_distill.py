# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Quick smoke test for a distilled checkpoint.

Loads the student model, runs a few chat prompts, and reports generation +
basic perplexity so you can sanity-check that distillation produced a coherent
model.

Usage::

    python scripts/test_distill.py --model checkpoints/distill/best
    python scripts/test_distill.py --model checkpoints/distill/best --ppl
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "用一句话解释什么是机器学习。",
    "Write a Python function that returns the nth Fibonacci number.",
    "中国的首都是哪里？",
    "9.11 和 9.9 哪个更大？请简要说明。",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/distill/best")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--ppl", action="store_true", help="also report a quick perplexity")
    ap.add_argument(
        "--mode",
        choices=["think", "no_think"],
        default="think",
        help="dual-mode control token appended to each prompt (matches training)",
    )
    args = ap.parse_args()
    hint = "/think" if args.mode == "think" else "/no_think"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e9:.2f}B  |  layers: {model.config.num_hidden_layers}\n")

    # stop on both <|endoftext|> and <|im_end|>
    eos_ids = [tok.eos_token_id]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tok.unk_token_id:
        eos_ids.append(im_end)

    for prompt in PROMPTS:
        messages = [{"role": "user", "content": f"{prompt} {hint}"}]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(text, return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        dt = time.time() - t0
        toks = gen.shape[0]
        reply = tok.decode(gen, skip_special_tokens=True)
        print("=" * 70)
        print(f"USER: {prompt}")
        print(f"ASSISTANT: {reply}")
        print(f"[{toks} tok, {dt:.2f}s, {toks/dt:.1f} tok/s]\n")

    if args.ppl:
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "Machine learning is a subfield of artificial intelligence. "
            "人工智能正在改变世界。"
        )
        enc = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            loss = model(**enc, labels=enc["input_ids"]).loss
        print(f"Quick perplexity on sample text: {torch.exp(loss).item():.2f}")


if __name__ == "__main__":
    main()
