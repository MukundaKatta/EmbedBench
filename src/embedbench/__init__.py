"""EmbedBench — Embedding model comparison toolkit."""

from embedbench.core import (
    BaseEmbedder,
    BowEmbedder,
    EmbedBench,
    NgramEmbedder,
    TfIdfEmbedder,
    evaluate_retrieval,
)
from embedbench.config import BenchmarkConfig, BowConfig, NgramConfig, TfIdfConfig

__all__ = [
    "BaseEmbedder",
    "BowEmbedder",
    "EmbedBench",
    "NgramEmbedder",
    "TfIdfEmbedder",
    "evaluate_retrieval",
    "BenchmarkConfig",
    "BowConfig",
    "NgramConfig",
    "TfIdfConfig",
]

__version__ = "0.1.0"
