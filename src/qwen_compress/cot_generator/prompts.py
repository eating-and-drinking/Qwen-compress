"""
Prompt 模板库
支持多域（数学/逻辑/代码/常识）差异化引导
"""

SYSTEM_PROMPT = """你是一个严格的逻辑推理专家。
对于每个问题，你必须按以下格式输出：

<thinking>
[步骤1] 理解问题：...
[步骤2] 分析条件：...
[步骤3] 推理过程：...
[步骤4] 验证结论：...
</thinking>

<answer>
[最终答案]
</answer>

不允许跳过任何步骤。每个步骤必须有实质内容，不得留空或敷衍。"""


DOMAIN_HINTS: dict[str, str] = {
    "math":         "这是一道数学推理题，注意列出每个计算步骤，包括中间变量和单位。",
    "science":      "这是科学推理题，引用相关原理，严格按照科学方法推导。",
    "code":         "这是代码分析题，逐行追踪变量状态，标注关键转折点。",
    "common_sense": "这是常识推理题，联系生活实际，逐步分析因果关系。",
    "language":     "这是语言理解题，分析语义、语境和隐含假设。",
    "general":      "",
    "logic":        "这是逻辑推理题，明确标出前提、推论关系和排除过程。",
}

# Self-Instruct 扩充问题用的模板
SELF_INSTRUCT_PROMPT = """基于以下示例问题，生成5个同类型但不同内容的新问题。

要求：
1. 难度与示例相近
2. 覆盖不同具体场景，不要与示例重复
3. 每个问题独占一行，以数字序号开头（1. 2. 3. 4. 5.）
4. 只输出问题本身，不要答案，不要解释

示例问题（领域：{domain}）：
{seed_question}

生成的新问题："""

# 交叉验证用的简洁 prompt
VERIFY_PROMPT = """请直接回答以下问题，只给出最终答案，不需要解释过程。

问题：{question}

答案："""


def build_user_prompt(question: str, domain: str = "general") -> str:
    hint = DOMAIN_HINTS.get(domain, "")
    if hint:
        return f"{hint}\n\n问题：{question}"
    return f"问题：{question}"


def build_self_instruct_prompt(seed_question: str, domain: str) -> str:
    return SELF_INSTRUCT_PROMPT.format(
        domain=domain,
        seed_question=seed_question,
    )


def build_verify_prompt(question: str) -> str:
    return VERIFY_PROMPT.format(question=question)
