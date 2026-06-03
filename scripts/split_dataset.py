#!/usr/bin/env python3
"""Split a JSONL dataset into train/validation splits.

Usage:
    python scripts/split_dataset.py \
        --input data/cot_sft_120k.jsonl \
        --train-output data/cot_sft_train.jsonl \
        --val-output data/cot_eval.jsonl \
        --val-ratio 0.05 \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split JSONL dataset into train/validation")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--train-output", required=True, help="Output train JSONL file")
    parser.add_argument("--val-output", required=True, help="Output validation JSONL file")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    input_path = Path(args.input)
    train_output_path = Path(args.train_output)
    val_output_path = Path(args.val_output)

    print(f"Loading dataset from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    
    print(f"Total examples: {len(examples)}")
    
    random.seed(args.seed)
    random.shuffle(examples)
    
    val_size = int(len(examples) * args.val_ratio)
    train_size = len(examples) - val_size
    
    train_examples = examples[:train_size]
    val_examples = examples[train_size:]
    
    print(f"Train examples: {len(train_examples)} ({(train_size/len(examples))*100:.1f}%)")
    print(f"Validation examples: {len(val_examples)} ({(val_size/len(examples))*100:.1f}%)")
    
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_output_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    val_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_output_path, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"Train set saved to {train_output_path}")
    print(f"Validation set saved to {val_output_path}")


if __name__ == "__main__":
    main()