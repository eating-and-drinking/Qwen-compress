"""Convert GSM8K parquet files to the CoTDataset JSONL format.

Input (GSM8K):
    question: "Natalia sold..."
    answer:   "Natalia sold 48/2 = <<48/2=24>>24 ...\n#### 72"

Output (CoTDataset):
    {
      "instruction": "Natalia sold...",
      "input": "",
      "chain_of_thought": "Natalia sold 48/2 = 24 clips in May.\n...",
      "answer": "72"
    }

Usage:
    python data/convert_gsm8k.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


RAW_DIR = Path(__file__).parent / "raw" / "gsm8k" / "main"
OUT_DIR = Path(__file__).parent
SPLITS = {
    "train-00000-of-00001.parquet": "cot_sft_train.jsonl",
    "test-00000-of-00001.parquet":  "cot_eval.jsonl",
}


def clean_cot(text: str) -> str:
    """Strip GSM8K calculator annotations <<48/2=24>>, keeping surrounding text.

    Original: "48/2 = <<48/2=24>>24 clips"
    Cleaned:  "48/2 = 24 clips"
    """
    return re.sub(r"<<[^>]*>>", "", text).strip()


def convert(row: dict) -> dict:
    raw_answer: str = row["answer"]
    # Split on the separator line "#### <number>"
    parts = raw_answer.split("####")
    cot_raw = parts[0].strip()
    final = parts[1].strip() if len(parts) > 1 else ""
    return {
        "instruction": row["question"].strip(),
        "input": "",
        "chain_of_thought": clean_cot(cot_raw),
        "answer": final,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for parquet_name, jsonl_name in SPLITS.items():
        src = RAW_DIR / parquet_name
        dst = OUT_DIR / jsonl_name
        df = pd.read_parquet(src)
        records = [convert(row) for row in df.to_dict("records")]
        with dst.open("w", encoding="utf-8") as fp:
            for r in records:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{parquet_name} → {dst.name}  ({len(records)} examples)")


if __name__ == "__main__":
    main()
