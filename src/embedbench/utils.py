"""Utility functions for EmbedBench."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Split text into word tokens using a simple regex tokeniser.

    Args:
        text: Input string.
        lowercase: Whether to lower-case the text first.

    Returns:
        List of token strings.
    """
    if lowercase:
        text = text.lower()
    return re.findall(r"\b\w+\b", text)


def char_ngrams(text: str, n_min: int = 2, n_max: int = 4, lowercase: bool = True) -> list[str]:
    """Extract character n-grams from *text*.

    Args:
        text: Input string.
        n_min: Minimum n-gram length.
        n_max: Maximum n-gram length.
        lowercase: Whether to lower-case the text first.

    Returns:
        List of character n-gram strings.
    """
    if lowercase:
        text = text.lower()
    grams: list[str] = []
    for n in range(n_min, n_max + 1):
        for i in range(len(text) - n + 1):
            grams.append(text[i : i + n])
    return grams


def cosine_similarity(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 when either vector has zero magnitude.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def l2_normalize(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the L2-normalised version of *v*. Returns zeros for zero vectors."""
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return v
    return v / norm


def precision_at_k(retrieved: Sequence[int], relevant: set[int], k: int) -> float:
    """Compute precision@k.

    Args:
        retrieved: Ordered list of retrieved document indices.
        relevant: Set of relevant document indices.
        k: Cut-off rank.

    Returns:
        Precision value between 0 and 1.
    """
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for doc in top_k if doc in relevant) / len(top_k)


def recall_at_k(retrieved: Sequence[int], relevant: set[int], k: int) -> float:
    """Compute recall@k."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for doc in top_k if doc in relevant) / len(relevant)


def mean_reciprocal_rank(retrieved: Sequence[int], relevant: set[int]) -> float:
    """Compute the reciprocal rank of the first relevant result."""
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0
