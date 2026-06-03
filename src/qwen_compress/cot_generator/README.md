﻿# CoT Generator

Batch generate high-quality Chain-of-Thought (CoT) training data for large model supervised fine-tuning.

---

## Features

- **Batch Generation**: Efficient batch inference based on vLLM, supporting multi-GPU parallel
- **Self-Instruct Expansion**: Automatically expand question pool to reach target data size
- **Quality Filtering**: Automatic deduplication, step completeness checking
- **Cross Validation**: Optional dual-model consistency verification (14B generation + 7B verification)
- **Dual-Mode Output**: Generate both CoT and Direct mode training data simultaneously
- **Standardized Format**: Output standard SFT training format, seamlessly integrated with mainstream training frameworks

---

## Quick Start

### 1. Install Dependencies

```bash
# Install basic dependencies
pip install -r requirements.txt

# Optional: Install HuggingFace datasets
pip install datasets
```

### 2. Run Generation

```bash
# Generate 120K examples with default config
python run.py generate

# Generate specified number
python run.py generate --target 50000

# Specify output directory
python run.py generate --output_dir /data/my_cot_dataset

# Disable cross validation (faster, suitable for single GPU)
python run.py generate --no_cross_val
```

### 3. Merge CoT/Direct Training Set

```bash
python run.py merge --output_dir outputs/cot_dataset --cot_ratio 0.7
```

### 4. View Statistics

```bash
python run.py stats --output_dir outputs/cot_dataset
```

---

## Configuration

Config file: `configs/pipeline.json`

| Field | Default | Description |
|-------|---------|-------------|
| `target_total` | 120000 | Target number of generated examples |
| `self_instruct_count` | 35000 | Number of questions to expand via Self-Instruct |
| `cot_mix_ratio` | 0.7 | Ratio of CoT examples in final mixed dataset |
| `enable_cross_validation` | false | Enable dual-model cross validation |
| `flush_every` | 500 | Flush to disk every N examples |

### Generator Configuration

```json
{
  "generator": {
    "model_name": "Qwen/Qwen2.5-14B-Instruct",
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.90,
    "max_model_len": 4096,
    "temperature": 0.7,
    "top_p": 0.9,
    "batch_size": 64,
    "min_thinking_length": 100,
    "required_steps": ["Step 1", "Step 2", "Step 3"]
  }
}
```

### Data Sources Configuration

Support HuggingFace datasets and local JSONL files:

```json
{
  "data_sources": {
    "hf_sources": {
      "math": 7500,
      "logiqa": 5000,
      "ceval": 15000,
      "mbpp": 8000
    },
    "local_sources": [
      {
        "path": "data/seeds/custom_math.jsonl",
        "domain": "math"
      }
    ]
  }
}
```

---

## Supported Data Sources

| Source | Type | Domain | Description |
|--------|------|--------|-------------|
| math | HuggingFace | Mathematics | Competition math problems |
| logiqa | HuggingFace | Logic | Logical reasoning questions |
| ceval | HuggingFace | General | Chinese exam dataset |
| mbpp | HuggingFace | Code | Python programming problems |
| local | JSONL | Custom | Local seed files |

---

## Output Files

```
outputs/cot_dataset/
 train_cot.jsonl      # CoT mode training set (with <thinking>...</thinking>)
 train_direct.jsonl   # Direct mode training set (answer only)
 train_mixed.jsonl    # Mixed training set (default 7:3 ratio)
 val.jsonl            # Validation set (5% split from training)
 test.jsonl           # Test set (5% split from training)
 metadata.json        # Statistics and metadata
```

### Data Format

Single example format (`train_mixed.jsonl`):

```json
{
  "id": "abc12345",
  "domain": "math",
  "question": "Solve the equation: 2x + 5 = 15",
  "thinking": "Step 1: Understand the problem... Step 2: Isolate x...",
  "answer": "x = 5",
  "source": "math",
  "sft_cot": {
    "messages": [
      {"role": "system", "content": "You are a logical reasoning expert..."},
      {"role": "user", "content": "Solve the equation: 2x + 5 = 15"},
      {"role": "assistant", "content": "<thinking>...steps...</thinking>\n\n<answer>x = 5</answer>"}
    ]
  },
  "sft_direct": {
    "messages": [
      {"role": "user", "content": "Solve the equation: 2x + 5 = 15"},
      {"role": "assistant", "content": "x = 5"}
    ]
  }
}
```

---

## Add Custom Seed Questions

Place JSONL files in `data/seeds/` directory, with each line formatted as:

```json
{"question": "Your question", "domain": "math"}
```

Supported domains: `math` / `logic` / `code` / `common_sense` / `science` / `language` / `general`

Then add the path in `data_sources.local_sources` in `configs/pipeline.json`.

---

## Hardware Reference

| Configuration | Throughput Estimate | Time for 120K Examples |
|--------------|---------------------|------------------------|
| 1x A100 80G | ~800 tok/s | ~25 hours |
| 2x A100 80G | ~1500 tok/s | ~13 hours |
| 4x A100 80G | ~2800 tok/s | ~7 hours |

> **Recommendation**: Set initial generation to 1.3x target (filter rate ~20-30%).

---

## Directory Structure

```
cot_generator/
 configs/
    pipeline.json          # Default configuration file
 data/seeds/
    example_seeds.jsonl    # Example seed questions
 __init__.py                # Module exports
 data_loader.py             # Multi-source data loader
 expander.py                # Self-Instruct expander
 formatter.py               # SFT format output
 generator.py               # Core generator (vLLM)
 pipeline.py                # End-to-end main pipeline
 prompts.py                 # Prompt template library
 validator.py               # Cross validation module
 run.py                     # CLI entry
 requirements.txt           # Dependencies list
 README.md                  # This file
```

---

## Import as Module

```python
from qwen_compress.cot_generator import run_pipeline

# Configuration
cfg = {
    "output_dir": "outputs/my_dataset",
    "target_total": 50000,
    "cot_mix_ratio": 0.7,
    "data_sources": {
        "hf_sources": {"math": 5000},
        "local_sources": [{"path": "data/seeds/custom.jsonl"}]
    },
    "generator": {
        "tensor_parallel_size": 1,
        "batch_size": 32
    }
}

# Run
run_pipeline(cfg)
```

---

## License

Apache License 2.0