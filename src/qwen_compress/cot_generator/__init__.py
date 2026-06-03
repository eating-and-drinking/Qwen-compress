"""
CoT Generator Module
Batch generation of Chain-of-Thought training data
"""

from .generator import CoTGenerator, GeneratorConfig
from .pipeline import run_pipeline

__all__ = [
    "CoTGenerator",
    "GeneratorConfig",
    "run_pipeline",
]
