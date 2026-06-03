"""
交叉验证模块
用 Qwen2.5-7B 对 14B 生成的答案做一致性校验，
一致 → confidence=high，不一致 → confidence=low（可选丢弃）
"""

import logging
from vllm import LLM, SamplingParams

from .prompts import build_verify_prompt

logger = logging.getLogger(__name__)


def simple_answer_match(a: str, b: str) -> bool:
    """
    简单字符串匹配：
    - 完全相等
    - 一方包含另一方（适合数字答案）
    - 数字提取后一致
    """
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return True
    if a in b or b in a:
        return True

    # 提取数字后对比
    import re
    nums_a = re.findall(r"-?\d+\.?\d*", a)
    nums_b = re.findall(r"-?\d+\.?\d*", b)
    if nums_a and nums_b and nums_a[0] == nums_b[0]:
        return True

    return False


class CrossValidator:
    """
    使用轻量级验证模型对生成的 CoT 条目进行双重校验。
    仅适用于有标准答案或可客观验证的题型（数学、逻辑选择题等）。
    对开放性问题跳过验证，默认 confidence=medium。
    """

    VERIFIABLE_DOMAINS = {"math", "logic", "science"}

    def __init__(self, verifier_model: str = "Qwen/Qwen2.5-7B-Instruct"):
        logger.info(f"Loading verifier model: {verifier_model}")
        self.llm = LLM(
            model=verifier_model,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.60,
            max_model_len=512,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,   # 贪婪解码，确定性答案
            max_tokens=128,
            stop=["<|im_end|>", "\n\n"],
        )
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(verifier_model)

    def validate_batch(
        self,
        entries: list[dict],
        drop_low_confidence: bool = False,
    ) -> list[dict]:
        """
        验证一批条目。
        返回带 confidence 字段的条目列表。
        drop_low_confidence=True 时过滤掉不一致的条目。
        """
        # 分离可验证 / 不可验证
        verifiable = [e for e in entries if e.get("domain") in self.VERIFIABLE_DOMAINS]
        non_verifiable = [e for e in entries if e.get("domain") not in self.VERIFIABLE_DOMAINS]

        # 不可验证的直接标 medium
        for e in non_verifiable:
            e["confidence"] = "medium"

        # 批量验证
        if verifiable:
            prompts = [
                self._build_verify_prompt(e["question"])
                for e in verifiable
            ]
            outputs = self.llm.generate(prompts, self.sampling_params)

            for entry, output in zip(verifiable, outputs):
                verifier_answer = output.outputs[0].text.strip()
                match = simple_answer_match(entry["answer"], verifier_answer)
                entry["confidence"] = "high" if match else "low"
                entry["verifier_answer"] = verifier_answer

        all_entries = verifiable + non_verifiable

        if drop_low_confidence:
            before = len(all_entries)
            all_entries = [e for e in all_entries if e["confidence"] != "low"]
            dropped = before - len(all_entries)
            logger.info(f"交叉验证：丢弃低置信度条目 {dropped} 条")

        return all_entries

    def _build_verify_prompt(self, question: str) -> str:
        messages = [{"role": "user", "content": build_verify_prompt(question)}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
