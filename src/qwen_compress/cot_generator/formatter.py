"""
Output formatting module
Convert raw entries to standard SFT training format (CoT mode / Direct answer mode)
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Single entry formatting
# ------------------------------------------------------------------

def format_sft_cot(system_prompt: str, question: str, parsed: dict) -> dict:
    """
    CoT mode: assistant outputs complete <think>...</think><answer>...</answer>
    Used to train the model's reasoning ability
    """
    assistant_content = (
        f"<think>\n{parsed['thinking']}\n</think>\n\n"
        f"<answer>\n{parsed['answer']}\n</answer>"
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def format_sft_direct(question: str, parsed: dict) -> dict:
    """
    Direct answer mode: assistant outputs only the final answer
    Used to maintain the model's ability to give concise answers, prevent overfitting to CoT format
    """
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": parsed["answer"]},
        ]
    }


# ------------------------------------------------------------------
# Dataset Writing
# ------------------------------------------------------------------

class DatasetWriter:
    """
    Write data to four files: train_cot / train_direct / val / test
    Default ratio: train 83% / val 8.5% / test 8.5%
    CoT:Direct mix ratio: 7:3
    """

    def __init__(
        self,
        output_dir: str,
        train_ratio: float = 0.83,
        val_ratio: float = 0.085,
        cot_mix_ratio: float = 0.7,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.cot_mix_ratio = cot_mix_ratio

        self._buffers: dict[str, list] = {
            "train_cot": [],
            "train_direct": [],
            "val": [],
            "test": [],
        }
        self._total = 0

    def add(self, entry: dict):
        """Add a raw entry, automatically route to corresponding split"""
        self._total += 1
        split = self._assign_split()

        if split == "train":
            self._buffers["train_cot"].append(entry["sft_cot"])
            self._buffers["train_direct"].append(entry["sft_direct"])
        elif split == "val":
            self._buffers["val"].append(entry["sft_cot"])
        else:
            self._buffers["test"].append(entry["sft_cot"])

    def flush(self):
        """Flush all buffers to disk"""
        for name, records in self._buffers.items():
            if not records:
                continue
            path = self.output_dir / f"{name}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            self._buffers[name] = []

    def finalize(self) -> dict:
        """Final flush and return statistics"""
        self.flush()
        stats = {}
        for name in ["train_cot", "train_direct", "val", "test"]:
            path = self.output_dir / f"{name}.jsonl"
            count = 0
            if path.exists():
                with open(path) as f:
                    count = sum(1 for _ in f)
            stats[name] = count

        # Write metadata
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"splits": stats, "total_raw": self._total}, f, ensure_ascii=False, indent=2)

        logger.info(f"Dataset writing complete: {stats}")
        return stats

    def _assign_split(self) -> str:
        import random
        r = random.random()
        if r < self.train_ratio:
            return "train"
        elif r < self.train_ratio + self.val_ratio:
            return "val"
        else:
            return "test"


def merge_cot_and_direct(
    output_dir: str,
    cot_ratio: float = 0.7,
    output_file: str = "train_mixed.jsonl",
):
    """
    Merge train_cot and train_direct by cot_ratio,
    Generate final training file for dual-mode fine-tuning
    """
    import random
    base = Path(output_dir)
    cot_path = base / "train_cot.jsonl"
    direct_path = base / "train_direct.jsonl"
    out_path = base / output_file

    cot_lines, direct_lines = [], []
    if cot_path.exists():
        with open(cot_path) as f:
            cot_lines = [json.loads(l) for l in f if l.strip()]
    if direct_path.exists():
        with open(direct_path) as f:
            direct_lines = [json.loads(l) for l in f if l.strip()]

    # Sample by ratio
    n_cot = int(len(cot_lines) * cot_ratio)
    n_direct = int(len(direct_lines) * (1 - cot_ratio))
    mixed = random.sample(cot_lines, min(n_cot, len(cot_lines)))
    mixed += random.sample(direct_lines, min(n_direct, len(direct_lines)))
    random.shuffle(mixed)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in mixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Mixed dataset written to: {out_path}, total {len(mixed)} entries (CoT:{n_cot} + Direct:{n_direct})")
    return str(out_path)
