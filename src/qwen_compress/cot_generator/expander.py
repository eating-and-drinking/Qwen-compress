"""
Self-Instruct 扩充器
用 Qwen2.5-14B 自身扩充问题池，补足到目标数量
"""

import re
import random
import logging
from vllm import SamplingParams

from .prompts import build_self_instruct_prompt

logger = logging.getLogger(__name__)


class SelfInstructExpander:
    """
    给定种子问题列表，让模型不断生成新问题，直到达到 target_count。
    每轮从种子池随机抽取一条作为示例，生成 5 条新问题。
    """

    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
        self.sampling_params = SamplingParams(
            temperature=0.9,
            top_p=0.95,
            max_tokens=512,
            stop=["<|im_end|>"],
        )

    def expand(
        self,
        seed_questions: list[dict],
        target_count: int,
        batch_size: int = 32,
    ) -> list[dict]:
        """
        返回新生成的问题列表（不含原始种子）
        """
        existing_questions = {q["question"] for q in seed_questions}
        new_questions: list[dict] = []
        attempt = 0
        max_attempts = target_count * 3  # 防止死循环

        logger.info(f"Self-Instruct 扩充目标: {target_count} 条")

        while len(new_questions) < target_count and attempt < max_attempts:
            # 随机抽取一批种子
            seeds = random.choices(seed_questions, k=batch_size)
            prompts = [
                self._build_prompt(s["question"], s.get("domain", "general"))
                for s in seeds
            ]

            outputs = self.llm.generate(prompts, self.sampling_params)

            for seed, output in zip(seeds, outputs):
                raw = output.outputs[0].text.strip()
                extracted = self._parse_questions(raw)

                for q_text in extracted:
                    if q_text in existing_questions:
                        continue
                    existing_questions.add(q_text)
                    new_questions.append({
                        "id": f"si_{len(new_questions)}",
                        "question": q_text,
                        "reference_answer": "",
                        "domain": seed.get("domain", "general"),
                        "source": "self_instruct",
                    })
                    if len(new_questions) >= target_count:
                        break

                if len(new_questions) >= target_count:
                    break

            attempt += batch_size
            if attempt % 500 == 0:
                logger.info(f"  Self-Instruct 进度: {len(new_questions)}/{target_count}")

        logger.info(f"Self-Instruct 扩充完成: 生成 {len(new_questions)} 条新问题")
        return new_questions

    # ------------------------------------------------------------------

    def _build_prompt(self, seed_question: str, domain: str) -> str:
        prompt_text = build_self_instruct_prompt(seed_question, domain)
        messages = [{"role": "user", "content": prompt_text}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _parse_questions(text: str) -> list[str]:
        """解析模型输出，提取编号问题列表"""
        lines = text.split("\n")
        questions = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 去掉序号前缀：1. / 1、/ （1）
            cleaned = re.sub(r"^[\d]+[\.、）\)]\s*", "", line).strip()
            if len(cleaned) >= 10:  # 过短的不要
                questions.append(cleaned)
        return questions
