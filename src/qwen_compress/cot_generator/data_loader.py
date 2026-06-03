"""
数据源加载器
支持 HuggingFace datasets + 本地 JSONL + 自定义种子文件
"""

import json
import logging
import random
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# HuggingFace 数据集配置
# key: (hf_dataset_name, subset_or_None, split, question_field, answer_field, domain)
HF_SOURCE_CONFIGS: dict[str, tuple] = {
    "math": (
        "hendrycks/competition_math", None, "train",
        "problem", "solution", "math"
    ),
    "logiqa": (
        "lucasmccabe/logiqa", None, "train",
        "query", "correct_option", "logic"
    ),
    "ceval": (
        "ceval/ceval-exam", "mixed", "val",
        "question", "answer", "general"
    ),
    "mbpp": (
        "google-research-datasets/mbpp", "full", "train",
        "text", "code", "code"
    ),
}


def load_hf_source(
    source_key: str,
    max_samples: int = 10000,
    seed: int = 42,
) -> list[dict]:
    """从 HuggingFace 加载数据集"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("请先安装: pip install datasets")

    cfg = HF_SOURCE_CONFIGS.get(source_key)
    if cfg is None:
        raise ValueError(f"未知数据源: {source_key}。可用: {list(HF_SOURCE_CONFIGS)}")

    name, subset, split, q_field, a_field, domain = cfg
    logger.info(f"Loading HF dataset: {name} / subset={subset} / split={split}")

    ds = load_dataset(name, subset, split=split) if subset else load_dataset(name, split=split)

    # 随机采样
    indices = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(indices)
    indices = indices[:max_samples]

    results = []
    for i, idx in enumerate(indices):
        row = ds[idx]
        question = str(row.get(q_field, "")).strip()
        answer = str(row.get(a_field, "")).strip()
        if not question:
            continue
        results.append({
            "id": f"{source_key}_{i}",
            "question": question,
            "reference_answer": answer,
            "domain": domain,
            "source": source_key,
        })

    logger.info(f"  Loaded {len(results)} samples from {source_key}")
    return results


def load_jsonl_source(filepath: str, domain: str = "general") -> list[dict]:
    """从本地 JSONL 文件加载（每行需含 question 字段）"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    results = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = row.get("question", row.get("instruction", row.get("input", ""))).strip()
            if not question:
                continue
            results.append({
                "id": row.get("id", f"local_{i}"),
                "question": question,
                "reference_answer": row.get("answer", row.get("output", "")),
                "domain": row.get("domain", domain),
                "source": path.stem,
            })

    logger.info(f"Loaded {len(results)} samples from {filepath}")
    return results


def load_seed_file(filepath: str) -> list[dict]:
    """加载种子问题文件（用于 Self-Instruct 扩充）"""
    return load_jsonl_source(filepath)


def load_all_sources(config: dict, seed: int = 42) -> list[dict]:
    """
    根据 config 加载并混合所有数据源

    config 格式示例：
    {
        "hf_sources": {
            "math": 7500,
            "logiqa": 5000
        },
        "local_sources": [
            {"path": "data/seeds/custom.jsonl", "domain": "math"}
        ]
    }
    """
    all_data = []

    for source_key, max_samples in config.get("hf_sources", {}).items():
        try:
            data = load_hf_source(source_key, max_samples=max_samples, seed=seed)
            all_data.extend(data)
        except Exception as e:
            logger.warning(f"加载 {source_key} 失败: {e}")

    for local_cfg in config.get("local_sources", []):
        try:
            data = load_jsonl_source(local_cfg["path"], domain=local_cfg.get("domain", "general"))
            all_data.extend(data)
        except Exception as e:
            logger.warning(f"加载本地文件失败: {e}")

    random.seed(seed)
    random.shuffle(all_data)
    logger.info(f"总计加载数据: {len(all_data)} 条")
    return all_data


def iter_batches(data: list[dict], batch_size: int) -> Iterator[list[dict]]:
    """按 batch_size 切分，生成 batch 迭代器"""
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]
