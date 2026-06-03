"""
输出格式化模块
将原始条目转换为标准 SFT 训练格式（CoT模式 / 直接答案模式）
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 单条格式化
# ------------------------------------------------------------------

def format_sft_cot(system_prompt: str, question: str, parsed: dict) -> dict:
    """
    CoT 模式：assistant 输出完整 <thinking>...</thinking><answer>...</answer>
    用于训练模型的推理能力
    """
    assistant_content = (
        f"<thinking>\n{parsed['thinking']}\n</thinking>\n\n"
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
    直接回答模式：assistant 只输出最终答案
    用于保持模型的简洁回答能力，防止过拟合到 CoT 格式
    """
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": parsed["answer"]},
        ]
    }


# ------------------------------------------------------------------
# 数据集写入
# ------------------------------------------------------------------

class DatasetWriter:
    """
    按照预设比例将数据写入 train_cot / train_direct / val / test 四个文件
    默认比例：train 83% / val 8.5% / test 8.5%
    CoT:Direct 混合比例：7:3
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
        """添加一条原始条目，自动路由到对应 split"""
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
        """将所有缓冲区写入磁盘"""
        for name, records in self._buffers.items():
            if not records:
                continue
            path = self.output_dir / f"{name}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            self._buffers[name] = []

    def finalize(self) -> dict:
        """最终刷盘并返回统计信息"""
        self.flush()
        stats = {}
        for name in ["train_cot", "train_direct", "val", "test"]:
            path = self.output_dir / f"{name}.jsonl"
            count = 0
            if path.exists():
                with open(path) as f:
                    count = sum(1 for _ in f)
            stats[name] = count

        # 写元数据
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"splits": stats, "total_raw": self._total}, f, ensure_ascii=False, indent=2)

        logger.info(f"数据集写入完成：{stats}")
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
    按 cot_ratio 混合 train_cot 和 train_direct，
    生成用于双模式微调的最终训练文件
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

    # 按比例抽取
    n_cot = int(len(cot_lines) * cot_ratio)
    n_direct = int(len(direct_lines) * (1 - cot_ratio))
    mixed = random.sample(cot_lines, min(n_cot, len(cot_lines)))
    mixed += random.sample(direct_lines, min(n_direct, len(direct_lines)))
    random.shuffle(mixed)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in mixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"混合数据集已写入: {out_path}，共 {len(mixed)} 条（CoT:{n_cot} + Direct:{n_direct}）")
    return str(out_path)
