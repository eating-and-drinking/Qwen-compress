"""
CoT Generator Module
批量生成 Chain-of-Thought 训练数据
"""

from .generator import CoTGenerator, GeneratorConfig
from .pipeline import run_pipeline

__all__ = [
    "CoTGenerator",
    "GeneratorConfig",
    "run_pipeline",
]
