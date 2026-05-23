# qwen-compress

Production-grade LLM compression toolkit for the Qwen family of models.
Three composable stages — **Group-wise Distillation → SparseGPT Pruning → INT8 QAT/QAD** — that take a 14B-class teacher down to a 3B INT8 student suitable for edge deployment, while preserving chain-of-thought reasoning quality.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org)

English | [简体中文](./README_zh.md)

---

## Features

- **Group-wise distillation.** Splits the teacher's decoder stack into `G` contiguous groups, picks one anchor layer per group, and matches the student's corresponding layer via a learned projector. Outperforms last-layer logit matching when the teacher is much deeper than the student.
- **SparseGPT pruning.** Block-by-block one-shot pruning with damped Hessian inversion. Supports unstructured, 2:4, and 4:8 patterns. Memory peaks at one decoder block.
- **QAT with optional QAD.** Drop-in `FakeQuantize` modules, per-channel weight + per-tensor (or per-token) activation quantization, learnable scales via LSQ, KV-cache quantization for end-to-end INT8 inference, and quantization-aware distillation that re-uses the same teacher to recover quantization-induced perplexity.
- **Mask-preserving recovery fine-tuning** between pruning and QAT, so sparsity isn't lost to subsequent training.
- **CoT-friendly SFT loader** with three modes (`direct` / `cot` / `dual`) that lets the student keep its reasoning ability after compression.
- **Single YAML config per stage**, pydantic-validated, with a top-level `pipeline` config that chains all three.
- **Production niceties**: structured logging via `loguru`, atomic checkpointing in `safetensors`, deterministic seeding, distributed-training helpers, ONNX QDQ export.

## Architecture

```
┌────────────────────┐     stage 1: distill      ┌────────────────────┐
│ Qwen2.5-14B-Inst.  │  ───────────────────────▶ │   Qwen2.5-3B       │
│  (teacher, frozen) │   group-wise CE+KD+MSE    │  (student, FP16)   │
└────────────────────┘                           └─────────┬──────────┘
                                                           │
                                              stage 2: SparseGPT prune
                                                           │
                                            ┌──────────────▼───────────┐
                                            │  3B @ 50% unstructured    │
                                            │   + mask-preserving FT    │
                                            └──────────────┬────────────┘
                                                           │
                                       stage 3: INT8 QAT + QAD
                                       (re-using the 14B teacher)
                                                           │
                                            ┌──────────────▼───────────┐
                                            │ 3B INT8 (W8A8 + KV-cache) │
                                            │   safetensors / ONNX     │
                                            └──────────────────────────┘
```

## Installation

```bash
git clone https://github.com/eating-and-drinking/Qwen-compress.git
cd Qwen-compress

# Editable install with dev extras
pip install -e ".[dev]"

# Optional: ONNX export, experiment tracking
pip install -e ".[all]"
```

Requires Python ≥ 3.9, PyTorch ≥ 2.1, CUDA ≥ 11.8 (for flash-attention 2 you'll also want CUDA 12.x).

## Quick Start

### CLI (recommended)

```bash
# Stage 1: distill Qwen2.5-14B-Instruct -> Qwen2.5-3B
bash scripts/run_distill.sh configs/distill/qwen2_5_14b_to_3b.yaml

# Stage 2: SparseGPT 50% pruning + recovery FT
bash scripts/run_prune.sh   configs/prune/sparsegpt_50pct.yaml

# Stage 3: INT8 QAT with QAD using the 14B teacher
bash scripts/run_qat.sh     configs/qat/int8_qad.yaml

# Or all three in one shot:
bash scripts/run_pipeline.sh configs/pipeline/full.yaml
```

### Python API

```python
from qwen_compress.distill import GroupwiseDistillTrainer
from qwen_compress.qat import QADTrainer, export_quantized_model
from qwen_compress.utils.config import load_config

# Stage 1
distill_cfg = load_config("configs/distill/qwen2_5_14b_to_3b.yaml", stage="distill")
distilled_path = GroupwiseDistillTrainer(distill_cfg).train()

# Stage 3 (pruning can be invoked similarly from `qwen_compress.prune`)
qat_cfg = load_config("configs/qat/int8_qad.yaml", stage="qat")
trainer = QADTrainer(qat_cfg)
ckpt = trainer.train()
export_quantized_model(trainer.student, trainer.tokenizer, "out/int8", fmt="safetensors")
```

## Data Format

All stages expect chain-of-thought SFT data as JSONL, one example per line:

```json
{
  "instruction": "Sort the list [3, 1, 4, 1, 5, 9, 2, 6] in ascending order.",
  "input": "",
  "chain_of_thought": "I need to compare each pair... [1, 1, 2, 3, 4, 5, 6, 9]",
  "answer": "[1, 1, 2, 3, 4, 5, 6, 9]"
}
```

The `dual` CoT mode (default) alternates between `direct` (answer only) and `cot` (`<think>...</think>` + answer) targets per example, so the student learns both fast and reasoned outputs.

## Configuration Reference

Each stage is driven by a single YAML file validated by pydantic. See `src/qwen_compress/utils/config.py` for the full schema. Key knobs:

### Distillation (`DistillConfig`)

| Key | Meaning | Default |
|---|---|---|
| `num_groups` | Number of teacher anchor layers | `12` |
| `group_strategy` | `uniform` or `depth_aware` | `uniform` |
| `alpha_ce` / `beta_kd` / `gamma_hidden` / `delta_attn` | Loss weights | `1.0 / 1.0 / 1.0 / 0.5` |
| `kd_temperature` | KD softmax temperature | `2.0` |
| `data.cot_mode` | `direct` / `cot` / `dual` | `dual` |

### Pruning (`PruneConfig`)

| Key | Meaning | Default |
|---|---|---|
| `sparsity` | Target sparsity in `[0, 1)` | `0.5` |
| `sparsity_type` | `unstructured` / `2:4` / `4:8` | `unstructured` |
| `block_size` | SparseGPT column block size | `128` |
| `percdamp` | Damping ratio for Hessian | `0.01` |
| `recovery_finetune` | Run mask-preserving FT after | `true` |

### QAT (`QATConfig`)

| Key | Meaning | Default |
|---|---|---|
| `weight_bits` / `activation_bits` | 8 or 4 / 8 or 16 | `8 / 8` |
| `weight_granularity` | `per_tensor` / `per_channel` | `per_channel` |
| `activation_granularity` | `per_tensor` / `per_token` | `per_tensor` |
| `quantize_kv_cache` | Quantize K/V projections' outputs | `true` |
| `calibration_method` | `minmax` / `percentile` / `mse` / `entropy` | `percentile` |
| `use_qad` | Enable Quantization-Aware Distillation | `true` |
| `export_format` | `safetensors` / `onnx` | `safetensors` |

## Distributed Training

Multi-GPU is enabled via `torchrun`:

```bash
NUM_GPUS=8 bash scripts/run_distill.sh configs/distill/qwen2_5_14b_to_3b.yaml
```

The teacher loads with `device_map="auto"` (shards across visible GPUs); the student stays on `cuda:0` of each rank. KD and hidden-state losses move teacher tensors onto the student's device automatically.

## Results (representative)

The numbers below are illustrative defaults for `Qwen2.5-14B-Instruct → Qwen2.5-3B → INT8` on a CoT SFT corpus of 120K examples. Reproduce by running the full pipeline; your own results depend on data, hardware, and budget.

| Stage | Model size | MMLU | GSM8K | Latency (A10, batch=1) |
|---|---:|---:|---:|---:|
| Teacher (14B FP16) | 28 GB | 78.5 | 81.4 | 1.00× |
| Distilled (3B FP16) | 6.2 GB | 71.2 | 73.8 | 0.32× |
| Pruned (3B 50%) | 6.2 GB ⁂ | 70.4 | 72.5 | 0.32× |
| Pruned + QAT INT8 | 2.4 GB | 69.8 | 71.6 | 0.21× |

⁂ Wall-clock size before sparse-aware packing; INT8 figure includes runtime quantization metadata.

## Project Layout

```
qwen-compress/
├── src/qwen_compress/
│   ├── distill/         # Stage 1: group-wise distillation
│   │   ├── groupwise.py    # Teacher↔student layer mapping
│   │   ├── losses.py       # Composite KD + hidden + attn loss
│   │   └── trainer.py      # GroupwiseDistillTrainer
│   ├── prune/           # Stage 2: SparseGPT
│   │   ├── sparsegpt.py    # Block-by-block one-shot pruner
│   │   ├── recovery.py     # Mask-preserving FT
│   │   └── utils.py        # N:M masks, sparsity metrics
│   ├── qat/             # Stage 3: QAT + QAD
│   │   ├── fake_quant.py   # FakeQuantize + QuantizedLinear
│   │   ├── calibration.py  # Activation calibration
│   │   ├── qad_trainer.py  # QAT loop with optional teacher
│   │   └── export.py       # safetensors / ONNX QDQ export
│   ├── models/qwen_wrapper.py
│   ├── data/{cot_dataset,calibration_data}.py
│   ├── utils/{config,checkpoint,logging,dist,seed}.py
│   └── cli.py
├── configs/{distill,prune,qat,pipeline}/*.yaml
├── scripts/run_{distill,prune,qat,pipeline}.sh
├── examples/quick_start.py
└── tests/
```

## Citing

If this code helps your work, please cite:

```bibtex
@software{qwen_compress_2024,
  title  = {qwen-compress: A Production-Grade Compression Toolkit for the Qwen Family},
  author = {eating-and-drinking},
  year   = {2024},
  url    = {https://github.com/eating-and-drinking/Qwen-compress},
}
```

The distillation algorithm builds on a forthcoming patent on group-wise distillation; the pruning stage implements Frantar & Alistarh, *"SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot"* (ICML 2023).

## Contributing

1. Fork and create a topic branch.
2. `pip install -e ".[dev]"` then `pre-commit install`.
3. Add tests under `tests/`.
4. `pytest -q && ruff check src tests` should pass.
5. Open a PR.

## Author

Created and maintained by [eating-and-drinking](https://github.com/eating-and-drinking).

Repository: [https://github.com/eating-and-drinking/Qwen-compress](https://github.com/eating-and-drinking/Qwen-compress)

## License

Apache License 2.0 — see [LICENSE](LICENSE). Includes an express patent grant.

Qwen is a trademark of Alibaba Group. This project is not affiliated with or endorsed by Alibaba.
