"""
Prompt template library
Supports domain-specific guidance for math/logic/code/common sense
"""

SYSTEM_PROMPT = """You are a professional logical reasoning assistant. Please solve the problem following this format:

<think>
Reason step by step:
1. First understand the core requirements of the problem
2. Analyze the given conditions and constraints
3. Show your reasoning process
4. Verify the correctness of the answer
</think>

<answer>
[Your final answer]
</answer>

Ensure your thinking process is clear and logically rigorous so readers can understand your reasoning path."""


DOMAIN_HINTS: dict[str, str] = {
    "math":         "Math reasoning: Write detailed calculation steps including formulas and intermediate results.",
    "science":      "Science reasoning: Cite relevant scientific principles and follow the scientific method.",
    "code":         "Code analysis: Analyze code logic line by line, tracking variable changes and execution flow.",
    "common_sense": "Common sense reasoning: Connect with real-life experience, analyze cause-effect relationships.",
    "language":     "Language understanding: Analyze semantics, context, and implied meaning.",
    "general":      "",
    "logic":        "Logical reasoning: Clearly state premises, show reasoning chain and conclusion.",
}


SELF_INSTRUCT_PROMPT = """Based on the following example question, generate 5 new questions of the same type but with different content.

Requirements:
1. Similar difficulty level as the example
2. Cover different scenarios, do not duplicate the example
3. Each question on a separate line, starting with number (1. 2. 3. 4. 5.)
4. Output only the questions, no answers or explanations

Example question (Domain: {domain}):
{seed_question}

Generated questions:"""


VERIFY_PROMPT = """Please answer the following question directly, only provide the final answer without explanation.

Question: {question}

Answer:"""


def build_user_prompt(question: str, domain: str = "general") -> str:
    hint = DOMAIN_HINTS.get(domain, "")
    if hint:
        return f"{hint}\n\nQuestion: {question}"
    return f"Question: {question}"


def build_self_instruct_prompt(seed_question: str, domain: str) -> str:
    return SELF_INSTRUCT_PROMPT.format(
        domain=domain,
        seed_question=seed_question,
    )


def build_verify_prompt(question: str) -> str:
    return VERIFY_PROMPT.format(question=question)
