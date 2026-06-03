"""
CoT Generator - 核心生成模块
使用 vLLM 批量驱动 Qwen2.5-14B-Instruct 生成思维链数据
"""

import re
import hashlib
import logging
from typing import Optional
from dataclasses import dataclass, field

from vllm import LLM, SamplingParams

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .formatter import format_sft_cot, format_sft_direct

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    enable_prefix_caching: bool = True

    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024

    batch_size: int = 64
    min_thinking_length: int = 100
    required_steps: list = field(default_factory=lambda: ["步骤1", "步骤2", "步骤3"])


class CoTGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.dedup_set: set[str] = set()
        self._stats = {"total": 0, "passed": 0, "dedup_drop": 0, "quality_drop": 0}

        logger.info(f"Loading model: {config.model_name}")
        self.llm = LLM(
            model=config.model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            enable_prefix_caching=config.enable_prefix_caching,
        )
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stop=["<|im_end|>"],
        )

        # 用于构建 chat template
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def generate_batch(self, questions: list[dict]) -> list[dict]:
        """批量生成，返回通过质量过滤的条目列表"""
        prompts = [self._build_prompt(q) for q in questions]
        outputs = self.llm.generate(prompts, self.sampling_params)

        results = []
        for q, output in zip(questions, outputs):
            raw_text = output.outputs[0].text
            self._stats["total"] += 1

            parsed = self._parse_output(raw_text)
            if parsed is None:
                self._stats["quality_drop"] += 1
                continue

            if not self._quality_check(parsed, q):
                continue

            entry = self._build_entry(q, parsed)
            results.append(entry)
            self._stats["passed"] += 1

        return results

    def stats(self) -> dict:
        total = self._stats["total"] or 1
        return {
            **self._stats,
            "pass_rate": f"{self._stats['passed'] / total:.1%}",
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_prompt(self, q: dict) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(q["question"], q.get("domain", "general"))},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _parse_output(self, text: str) -> Optional[dict]:
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

        if not thinking_match or not answer_match:
            return None

        thinking = thinking_match.group(1).strip()
        answer = answer_match.group(1).strip()

        if len(thinking) < self.config.min_thinking_length:
            return None
        if len(answer) < 2:
            return None

        return {"thinking": thinking, "answer": answer}

    def _quality_check(self, parsed: dict, q: dict) -> bool:
        # 1. 去重
        content_hash = hashlib.md5(parsed["thinking"].encode()).hexdigest()
        if content_hash in self.dedup_set:
            self._stats["dedup_drop"] += 1
            return False
        self.dedup_set.add(content_hash)

        # 2. 步骤完整性
        thinking = parsed["thinking"]
        if not all(m in thinking for m in self.config.required_steps):
            self._stats["quality_drop"] += 1
            return False

        return True

    def _build_entry(self, q: dict, parsed: dict) -> dict:
        return {
            "id": q.get("id", hashlib.md5(q["question"].encode()).hexdigest()[:8]),
            "domain": q.get("domain", "general"),
            "question": q["question"],
            "thinking": parsed["thinking"],
            "answer": parsed["answer"],
            "source": q.get("source", "unknown"),
            "sft_cot": format_sft_cot(SYSTEM_PROMPT, q["question"], parsed),
            "sft_direct": format_sft_direct(q["question"], parsed),
        }
