"""
Data source loader
Supports HuggingFace datasets + local JSONL + custom seed files
"""

import json
import logging
import random
from pathlib import Path
from typing import Iterator, Callable

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Dataset-specific parsing functions
# ------------------------------------------------------------------

def parse_math(row: dict) -> tuple[str, str]:
    """MATH dataset: problem -> solution"""
    question = str(row.get("problem", "")).strip()
    answer = str(row.get("solution", "")).strip()
    return question, answer


def parse_gsm8k(row: dict) -> tuple[str, str]:
    """GSM8K dataset: question -> answer"""
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    return question, answer


def parse_arc(row: dict) -> tuple[str, str]:
    """ARC dataset: question + choices(dict) -> answerKey"""
    question = str(row.get("question", "")).strip()
    answer_key = str(row.get("answerKey", "")).strip()
    
    # choices is a dict containing 'text' and 'label' lists
    choices = row.get("choices", {})
    if isinstance(choices, dict):
        texts = choices.get("text", [])
        labels = choices.get("label", [])
        if texts and labels:
            options_text = "\n".join(
                f"{label}. {str(text).strip()}"
                for label, text in zip(labels, texts)
            )
            question = f"{question}\n\nOptions:\n{options_text}"
            # Convert answer letter to option text
            for label, text in zip(labels, texts):
                if label == answer_key:
                    answer = str(text).strip()
                    break
            else:
                answer = answer_key
    else:
        answer = answer_key
    
    return question, answer


def parse_openbookqa(row: dict) -> tuple[str, str]:
    """OpenBookQA dataset: question_stem + choices -> answerKey"""
    question = str(row.get("question_stem", "")).strip()
    answer_key = str(row.get("answerKey", "")).strip()
    
    # choices may be dict or list
    choices = row.get("choices", {})
    if isinstance(choices, dict):
        texts = choices.get("text", [])
        labels = choices.get("label", [])
        if texts and labels:
            options_text = "\n".join(
                f"{label}. {str(text).strip()}"
                for label, text in zip(labels, texts)
            )
            question = f"{question}\n\nOptions:\n{options_text}"
            for label, text in zip(labels, texts):
                if label == answer_key:
                    answer = str(text).strip()
                    break
            else:
                answer = answer_key
    elif isinstance(choices, list):
        option_labels = ["A", "B", "C", "D", "E", "F"]
        options_text = "\n".join(
            f"{option_labels[j]}. {str(opt).strip()}"
            for j, opt in enumerate(choices)
        )
        question = f"{question}\n\nOptions:\n{options_text}"
        # answer_key may be index or letter
        try:
            idx = int(answer_key)
            answer = str(choices[idx]).strip() if idx < len(choices) else answer_key
        except ValueError:
            for j, label in enumerate(option_labels):
                if label == answer_key and j < len(choices):
                    answer = str(choices[j]).strip()
                    break
            else:
                answer = answer_key
    else:
        answer = answer_key
    
    return question, answer


def parse_mbpp(row: dict) -> tuple[str, str]:
    """MBPP dataset: text -> code"""
    question = str(row.get("text", "")).strip()
    answer = str(row.get("code", "")).strip()
    return question, answer


def parse_logiqa(row: dict) -> tuple[str, str]:
    """LogiQA dataset: query + options -> correct_option(index)"""
    # Some versions use 'query', some use 'question'
    question = str(row.get("query", row.get("question", ""))).strip()
    
    # context field may contain background information
    context = str(row.get("context", "")).strip()
    if context:
        question = f"{context}\n\n{question}"
    
    options = row.get("options", [])
    if isinstance(options, list) and options:
        option_labels = ["A", "B", "C", "D", "E", "F"]
        options_text = "\n".join(
            f"{option_labels[j]}. {str(opt).strip()}"
            for j, opt in enumerate(options)
        )
        question = f"{question}\n\nOptions:\n{options_text}"
        
        # correct_option is an index
        correct_idx = row.get("correct_option", row.get("answer", 0))
        try:
            idx = int(correct_idx)
            answer = str(options[idx]).strip() if idx < len(options) else str(correct_idx)
        except (ValueError, TypeError):
            answer = str(correct_idx)
    else:
        answer = str(row.get("correct_option", row.get("answer", ""))).strip()
    
    return question, answer


def parse_ceval(row: dict) -> tuple[str, str]:
    """CEval dataset: question + A,B,C,D fields -> answer(letter)"""
    question = str(row.get("question", "")).strip()
    
    # Options are separate A, B, C, D fields
    option_fields = ["A", "B", "C", "D"]
    options = []
    for field in option_fields:
        opt = row.get(field, "")
        if opt:
            options.append(str(opt).strip())
    
    if options:
        options_text = "\n".join(
            f"{option_fields[j]}. {opt}"
            for j, opt in enumerate(options)
        )
        question = f"{question}\n\nOptions:\n{options_text}"
        
        # answer is a letter (e.g., "C")
        answer_letter = str(row.get("answer", "")).strip().upper()
        letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
        idx = letter_to_idx.get(answer_letter, -1)
        if 0 <= idx < len(options):
            answer = options[idx]
        else:
            answer = answer_letter
    else:
        answer = str(row.get("answer", "")).strip()
    
    return question, answer


# ------------------------------------------------------------------
# HuggingFace dataset configuration
# key: (hf_dataset_name, subset_or_None, split, domain, parser_function)
# ------------------------------------------------------------------
HF_SOURCE_CONFIGS: dict[str, tuple] = {
    "math": (
        "hendrycks/competition_math", None, "train",
        "math", parse_math
    ),
    "gsm8k": (
        "gsm8k", "main", "train",
        "math", parse_gsm8k
    ),
    "arc": (
        "allenai/ai2_arc", "ARC-Challenge", "train",
        "science", parse_arc
    ),
    "openbookqa": (
        "openbookqa", "main", "train",
        "general", parse_openbookqa
    ),
    "mbpp": (
        "google-research-datasets/mbpp", "full", "train",
        "code", parse_mbpp
    ),
    "logiqa": (
        "lucasmccabe/logiqa", None, "train",
        "logic", parse_logiqa
    ),
    "ceval": (
        "ceval", None, "train",
        "general", parse_ceval
    ),
}


def load_hf_source(
    source_key: str,
    max_samples: int = 10000,
    seed: int = 42,
) -> list[dict]:
    """Load dataset from HuggingFace"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Please install: pip install datasets")

    cfg = HF_SOURCE_CONFIGS.get(source_key)
    if cfg is None:
        raise ValueError(f"Unknown source: {source_key}. Available: {list(HF_SOURCE_CONFIGS)}")

    name, subset, split, domain, parser = cfg
    logger.info(f"Loading HF dataset: {name} / subset={subset} / split={split}")

    ds = load_dataset(name, subset, split=split) if subset else load_dataset(name, split=split)

    # Random sampling
    indices = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(indices)
    indices = indices[:max_samples]

    results = []
    for i, idx in enumerate(indices):
        row = ds[idx]
        
        try:
            question, answer = parser(row)
        except Exception as e:
            logger.warning(f"Failed to parse {source_key} row {idx}: {e}")
            continue
        
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
    """Load from local JSONL file (each line must contain 'question' field)"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

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
    """Load seed question file (used for Self-Instruct expansion)"""
    return load_jsonl_source(filepath)


def load_all_sources(config: dict, seed: int = 42) -> list[dict]:
    """
    Load and mix all data sources according to config

    Config format example:
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
            logger.warning(f"Failed to load {source_key}: {e}")

    for local_cfg in config.get("local_sources", []):
        try:
            data = load_jsonl_source(local_cfg["path"], domain=local_cfg.get("domain", "general"))
            all_data.extend(data)
        except Exception as e:
            logger.warning(f"Failed to load local file: {e}")

    random.seed(seed)
    random.shuffle(all_data)
    logger.info(f"Total loaded samples: {len(all_data)}")
    return all_data


def iter_batches(data: list[dict], batch_size: int) -> Iterator[list[dict]]:
    """Split data into batches of specified size"""
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]
