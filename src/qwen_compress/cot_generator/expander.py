"""
Self-Instruct Expander
Uses Qwen2.5-14B to expand question pool to target count
"""

import re
import random
import logging
from vllm import SamplingParams

from .prompts import build_self_instruct_prompt

logger = logging.getLogger(__name__)


class SelfInstructExpander:
    """
    Given a list of seed questions, generates new questions until reaching target_count.
    Each iteration randomly selects a seed as example and generates 5 new questions.
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
        Returns list of newly generated questions (excluding original seeds)
        """
        existing_questions = {q["question"] for q in seed_questions}
        new_questions: list[dict] = []
        attempt = 0
        max_attempts = target_count * 3  # Prevent infinite loop

        logger.info(f"Self-Instruct target: {target_count} items")

        while len(new_questions) < target_count and attempt < max_attempts:
            # Randomly select a batch of seeds
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
                logger.info(f"  Self-Instruct progress: {len(new_questions)}/{target_count}")

        logger.info(f"Self-Instruct complete: Generated {len(new_questions)} new questions")
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
        """Parse model output and extract numbered question list"""
        lines = text.split("\n")
        questions = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove number prefix: 1. / 1、/ (1)
            cleaned = re.sub(r"^[\d]+[\.、）\)]\s*", "", line).strip()
            if len(cleaned) >= 10:  # Skip too short questions
                questions.append(cleaned)
        return questions
