# qwen-compress

Production-grade LLM compression toolkit for the Qwen family of models.
Three composable stages — **MOT-FD Distillation → SparseGPT Pruning → INT8 QAT/QAD** — that take a 14B-class teacher down to a 3B INT8 student suitable for edge deployment, while preserving chain-of-thought reasoning quality.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org)

English

---

## Featuresv

- **MOT-FD distillation.** Monotonic Optimal Transport Functional Distillation: decomposes the 48-layer teacher into 12 functional groups via change-point detection on representation dynamics, then aligns the 36-layer student using Sinkhorn optimal transport with a monotonic semantic-progression constraint. Outperforms naive layer-matching strategies when the teacher is much deeper than the student.
- **SparseGPT pruning.** Block-by-block one-shot pruning with damped Hessian inversion. Supports unstructured, 2:4, and 4:8 patterns. Memory peaks at one decoder block.
- **Channel permutation for 2:4** (`prune.permutation.enabled: true`). Prune unstructured for max accuracy, then search a column permutation that snaps the existing zeros onto a 2:4 grid, then hard-enforce the residual. Yields ~unstructured accuracy with 2:4 hardware speedup — currently the Pareto frontier on Ampere/Hopper Tensor Cores and our CPU 2:4 kernel.
- **QAT with optional QAD.** Drop-in `FakeQuantize` modules, per-channel weight + per-tensor (or per-token) activation quantization, learnable scales via LSQ, KV-cache quantization for end-to-end INT8 inference, and quantization-aware distillation that re-uses the same teacher to recover quantization-induced perplexity.
- **Mask-preserving recovery fine-tuning** between pruning and QAT, so sparsity isn't lost to subsequent training.
- **CoT-friendly SFT loader** with three modes (`direct` / `cot` / `dual`) that lets the student keep its reasoning ability after compression.
- **Single YAML config per stage**, pydantic-validated, with a top-level `pipeline` config that chains all three.
- **Production niceties**: structured logging via `loguru`, atomic checkpointing in `safetensors`, deterministic seeding, distributed-training helpers, ONNX QDQ export.

## Architecture

```
┌────────────────────┐     stage 1: MOT-FD distill   ┌────────────────────┐
│ Qwen2.5-14B-Inst.  │  ──────────────────────────▶  │   Qwen2.5-3B       │
│  (teacher, frozen) │   48→12 groups, OT align,     │  (student, FP16)   │
│                    │   monotonic constraint         │                    │
└────────────────────┘                               └─────────┬──────────┘
                                                               │
                                              stage 2: SparseGPT prune
                                                               │
                                            ┌──────────────────▼───────────┐
                                            │  3B @ 50% unstructured        │
                                            │   + mask-preserving FT        │
                                            └──────────────────┬────────────┘
                                                               │
                                       stage 3: INT8 QAT + QAD
                                       (re-using the 14B teacher)
                                                               │
                                            ┌──────────────────▼───────────┐
                                            │ 3B INT8 (W8A8 + KV-cache)    │
                                            │   safetensors / ONNX         │
                                            └──────────────────────────────┘
```

## Algorithm: MOT-FD

### 1. Teacher Functional Decomposition (48 → 12 Groups)

For each teacher layer l, extract the mean representation z_l^T = E_{x~D}[h_l^T(x)]. Then compute a representation dynamics energy signal:

```
E(l) = α·||z_{l+1} - z_l|| + β·||z_{l+1} - 2z_l + z_{l-1}|| + γ·(1 - cos(z_{l+1}, z_l))
```

Detect 11 change points (breakpoints) via peak detection on E(l), and construct 12 non-uniform functional groups G_k = [b_{k-1}, b_k). Each group is summarized as g_k^T = mean(G_k^T).

**Interpretation:**
- Early groups → syntax / token formation
- Middle groups → semantic integration
- Late groups → reasoning + alignment

### 2. Student Representation Manifold

The student produces 36 layer hidden states H^S = {h_1^S, ..., h_36^S}. No explicit grouping is imposed — the student remains a continuous representation manifold.

### 3. Optimal Transport Alignment

A cost matrix C_{l,k} = ||h_l^S - g_k^T||^2 is built between all student layers and teacher group representations. The Sinkhorn algorithm solves for an entropy-regularized transport plan γ* that minimizes Σ γ_{l,k} C_{l,k} subject to marginal constraints.

A **monotonic constraint** ensures semantic order is preserved: l_1 < l_2 ⇒ E_γ[k|l_1] ≤ E_γ[k|l_2]. This prevents "semantic backtracking" in student depth.

### 4. Composite Loss

```
L = L_CE + λ_KD·L_KD + λ_OT·L_OT + λ_mono·L_mono
```

- **L_CE**: Standard next-token prediction cross-entropy.
- **L_KD**: KL-divergence between student and teacher logits.
- **L_OT**: Σ γ_{l,k} C_{l,k} — the optimal transport alignment cost.
- **L_mono**: Σ_l max(0, μ_l - μ_{l+1}) — monotonic regularization where μ_l is the expected functional position of student layer l.

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
# Stage 1: MOT-FD distill Qwen2.5-14B-Instruct -> Qwen2.5-3B
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

# Stage 1: MOT-FD distillation
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

## Sparsity Strategies: Unstructured → Permuted 2:4

The pruning stage exposes three sparsity patterns, each at a different point on the accuracy/speed Pareto frontier.

| Pattern | Δ PPL (3B, MMLU) | Hardware acceleration | When to use |
|---|---:|---|---|
| Unstructured 50%      | ~0.4 | None (storage only)                          | Memory-constrained, no sparse hardware |
| Naïve 2:4             | 1.0 – 1.5 | cuSPARSELt (~1.5×) Ampere/Hopper, custom CPU | 2:4 hardware available, accuracy hit OK |
| **Permuted 2:4** ⭐    | ~0.6 | Same as naïve 2:4                            | **Best of both worlds — current Pareto frontier** |

### Why naïve 2:4 hurts

A `2:4` constraint says: *every 4 consecutive columns must contain exactly 2 zeros*. This is a strong **positional** constraint — it doesn't care which 2 values are largest, only that they land in the right slots. When forced onto a weight matrix whose natural sparsity pattern is misaligned, it ends up zeroing values that should have been kept (and keeping values that should have been zeroed). Empirically, this costs ~1.0 PPL on Qwen2.5-3B vs. a 50% unstructured mask of the same row.

### The geometric insight: column permutation is a free variable

A linear layer is permutation-equivariant on its **input channel axis**:

```
y = W x                       (W shape [out, in], x shape [in])
y = W P P⁻¹ x  =  W' x'       (P is any permutation matrix on `in`)

where  W' = W P    and    x' = P⁻¹ x
```

You can freely reorder W's columns **as long as you reorder x the same way**. So if we permute one layer's input channels, we must permute the **previous layer's outputs** to match. For the MLP block in a Qwen2.5 decoder:

```
  hidden ──▶ gate_proj ──┐
                          ├──▶ SwiGLU ──▶ down_proj ──▶ hidden
  hidden ──▶ up_proj   ──┘
```

Permuting the `intermediate` dimension is network-equivariant if we permute, by the same `π`:

- the **rows** of `gate_proj.weight` and `up_proj.weight`, AND
- the **columns** of `down_proj.weight`

The block computes the *same function* — only the internal channel ordering changes. This is implemented in `apply_intermediate_permutation()` in `src/qwen_compress/prune/permutation.py`.

For attention blocks, the per-head structure makes the equivalent rewrite less clean (heads are independent groups of channels), so we currently restrict the permutation pass to MLP blocks. SparseGPT is run on attention layers normally, but the 2:4 alignment cost there is small because attention matrices are typically smaller than MLPs.

### The algorithm

1. **Step 1 — Unstructured SparseGPT.** Run pruning with `sparsity_type: unstructured` to get the max-quality 50% mask. This sets the accuracy ceiling.

2. **Step 2 — Greedy column-swap search.** For each decoder block, find the permutation `π` of the `intermediate` axis that pushes as many existing zeros as possible into 2:4-aligned positions. The search is implemented in `PermutationSearcher`:
   - **Cost function:** count of (row × group) pairs that don't yet have exactly 2 zeros out of 4 (the residual misalignment).
   - **Sampling:** swap candidates are biased toward groups with high misalignment (avoid wasting steps on already-aligned groups).
   - **Acceptance:** greedy — only swaps that strictly decrease cost are kept.
   - **Cost:** typically 0.1 – 2 seconds per layer on a single CPU thread.

3. **Step 3 — Apply the permutation.** Permute `gate_proj` / `up_proj` rows and `down_proj` columns by the same `π`. Verify network equivalence: output of a random forward pass before/after should match to within FP16 ULP.

4. **Step 4 — Hard-enforce 2:4.** Whatever misalignment is left after the search gets cleaned up by `hard_enforce_n_m()`: in each group of 4 columns per row, zero the 2 smallest-`|W|` entries. Typical cleanup touches < 5% of weights (vs. ~50% if you'd skipped step 2).

5. **Step 5 — Recovery fine-tune.** Short mask-frozen fine-tune on the calibration data to recover the small accuracy loss introduced by step 4. (Stage 3 QAT will also run with mask preservation, so the zeros stay zero through training.)

### When this is worth running

- ✅ Target hardware has 2:4 acceleration: NVIDIA Ampere/Hopper Tensor Cores via `cuSPARSELt`, or the [llama.cpp sparse 2:4 CPU kernel](https://github.com/eating-and-drinking/llama.cpp) shipped as a sister repository.
- ✅ You want the accuracy of unstructured pruning *and* the speedup of 2:4.
- ❌ Skip if you don't have 2:4 hardware — plain unstructured is simpler and at least as accurate.
- ❌ Skip for tiny calibration sets (< 64 samples); SparseGPT itself may not converge well, so permutation buys you nothing.

### Usage

YAML configuration (full example: `configs/prune/sparsegpt_perm_2_4.yaml`):

```yaml
stage: prune
sparsity: 0.5
sparsity_type: unstructured       # step 1: prune unstructured for max quality
block_size: 128
percdamp: 0.01

permutation:                      # steps 2-4: permute + hard-enforce
  enabled: true
  target: "2:4"
  enforce_after: true             # set false to skip step 4 and keep approximate 2:4
  max_iters: 300                  # search budget per block
  swaps_per_iter: 200             # candidates evaluated per iteration
  seed: 0

recovery_finetune: true           # step 5
recovery_steps: 500
recovery_lr: 5.0e-5
```

CLI:

```bash
qwen-compress prune --config configs/prune/sparsegpt_perm_2_4.yaml
```

Python API:

```python
from qwen_compress.prune.permutation import permute_model_for_2_4

stats = permute_model_for_2_4(
    model, n=2, m=4, enforce=True,
    max_iters=300, swaps_per_iter=200, seed=0,
)
print(f"alignment: {stats['avg_initial_alignment']:.1%} "
      f"→ {stats['avg_post_perm_alignment']:.1%} "
      f"→ 100% after enforce")
```

Per-layer log output during the search:

```
Channel permutation for 2:4 alignment across 36 blocks (enforce=True)
  layer  0: alignment 49.8% → 96.4% (misalign 102341 → 7912 → 0 after enforce)
  layer  1: alignment 49.7% → 96.1% (misalign 103014 → 8154 → 0 after enforce)
  layer  2: alignment 49.9% → 96.7% (misalign 101803 → 7521 → 0 after enforce)
  ...
```

## Configuration Reference

Each stage is driven by a single YAML file validated by pydantic. See `src/qwen_compress/utils/config.py` for the full schema. Key knobs:

### MOT-FD Distillation (`DistillConfig`)

| Key | Meaning | Default |
|---|---|---|
| `num_groups` | Number of functional groups (teacher) | `12` |
| `calibration_samples` | Samples for teacher decomposition | `256` |
| `energy_alpha` / `energy_beta` / `energy_gamma` | Energy signal weights | `1.0 / 0.5 / 0.3` |
| `min_peak_distance` | Minimum gap between breakpoints | `2` |
| `alpha_ce` | CE loss weight | `1.0` |
| `beta_kd` | KD loss weight | `1.0` |
| `lambda_ot` | OT alignment loss weight | `1.0` |
| `lambda_mono` | Monotonic regularization weight | `0.1` |
| `kd_temperature` | KD softmax temperature | `2.0` |
| `ot_temperature` | Sinkhorn entropy regularization ε | `0.1` |
| `sinkhorn_iters` | Sinkhorn iterations per step | `50` |
| `projector_lr_multiplier` | Projector LR = base LR × this value | `0.1` |
| `data.cot_mode` | `direct` / `cot` / `dual` | `dual` |

### Pruning (`PruneConfig`)

| Key | Meaning | Default |
|---|---|---|
| `sparsity` | Target sparsity in `[0, 1)` | `0.5` |
| `sparsity_type` | `unstructured` / `2:4` / `4:8` | `unstructured` |
| `block_size` | SparseGPT column block size | `128` |
| `percdamp` | Damping ratio for Hessian | `0.01` |
| `recovery_finetune` | Run mask-preserving FT after | `true` |
| `permutation.enabled` | Run channel-permutation pass after pruning | `false` |
| `permutation.target` | N:M target for permutation (`"2:4"` / `"4:8"`) | `"2:4"` |
| `permutation.enforce_after` | Hard-enforce N:M after applying permutation | `true` |
| `permutation.max_iters` | Greedy search budget per block | `300` |
| `permutation.swaps_per_iter` | Swap candidates evaluated per iteration | `200` |

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

The teacher loads with `device_map="auto"` (shards across visible GPUs); the student stays on `cuda:0` of each rank. OT alignment uses pre-computed teacher group representations (on the student's device), so no cross-device transfers are needed during the training loop.

## Results (representative)

The numbers below are illustrative defaults for `Qwen2.5-14B-Instruct → Qwen2.5-3B → INT8` on a CoT SFT corpus of 120K examples. Reproduce by running the full pipeline; your own results depend on data, hardware, and budget.

| Stage | Model size | MMLU | GSM8K | Latency (A10, batch=1) |
|---|---:|---:|---:|---:|
| Teacher (14B FP16) | 28 GB | 78.5 | 81.4 | 1.00× |
| MOT-FD distilled (3B FP16) | 6.2 GB | 72.0 | 74.5 | 0.32× |
| Pruned (3B 50%) | 6.2 GB ⁂ | 71.2 | 73.2 | 0.32× |
| Pruned + QAT INT8 | 2.4 GB | 70.5 | 72.0 | 0.21× |

⁂ Wall-clock size before sparse-aware packing; INT8 figure includes runtime quantization metadata.

## Project Layout

```
qwen-compress/
├── src/qwen_compress/
│   ├── distill/              # Stage 1: MOT-FD distillation
│   │   ├── groupwise.py         # Teacher functional decomposition (48→12 groups)
│   │   ├── losses.py            # CE + KD + OT alignment + monotonic loss
│   │   └── trainer.py           # GroupwiseDistillTrainer
│   ├── prune/                # Stage 2: SparseGPT
│   │   ├── sparsegpt.py         # Block-by-block one-shot pruner
│   │   ├── permutation.py       # Channel permutation for 2:4
│   │   ├── recovery.py          # Mask-preserving FT
│   │   └── utils.py             # N:M masks, sparsity metrics
│   ├── qat/                  # Stage 3: QAT + QAD
│   │   ├── fake_quant.py        # FakeQuantize + QuantizedLinear
│   │   ├── calibration.py       # Activation calibration
│   │   ├── qad_trainer.py       # QAT loop with optional teacher
│   │   └── export.py            # safetensors / ONNX QDQ export
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
