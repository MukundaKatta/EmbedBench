"""Core embedding and benchmarking classes for EmbedBench."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from embedbench.config import BenchmarkConfig, BowConfig, NgramConfig, TfIdfConfig
from embedbench.utils import (
    char_ngrams,
    cosine_similarity,
    l2_normalize,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    tokenize,
)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseEmbedder(ABC):
    """Abstract base for all embedders."""

    name: str = "base"

    @abstractmethod
    def fit(self, documents: Sequence[str]) -> None:
        """Learn vocabulary / statistics from a corpus."""

    @abstractmethod
    def embed(self, text: str) -> NDArray[np.float64]:
        """Return a dense vector for *text*."""

    def compare(self, text1: str, text2: str) -> float:
        """Return cosine similarity between two texts."""
        return cosine_similarity(self.embed(text1), self.embed(text2))


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------


class TfIdfEmbedder(BaseEmbedder):
    """Term-frequency / inverse-document-frequency embedder."""

    name = "tfidf"

    def __init__(self, config: TfIdfConfig | None = None) -> None:
        self.config = config or TfIdfConfig()
        self._vocab: dict[str, int] = {}
        self._idf: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._fitted = False

    def fit(self, documents: Sequence[str]) -> None:
        n_docs = len(documents)
        if n_docs == 0:
            raise ValueError("Cannot fit on an empty corpus.")

        # Build vocabulary and document frequencies
        df: dict[str, int] = {}
        for doc in documents:
            seen: set[str] = set()
            for token in tokenize(doc, lowercase=self.config.lowercase):
                if token not in seen:
                    df[token] = df.get(token, 0) + 1
                    seen.add(token)

        # Filter by min_df / max_df_ratio
        filtered_terms = sorted(
            term
            for term, count in df.items()
            if count >= self.config.min_df and (count / n_docs) <= self.config.max_df_ratio
        )

        self._vocab = {term: idx for idx, term in enumerate(filtered_terms)}
        idf_values = np.zeros(len(self._vocab), dtype=np.float64)
        for term, idx in self._vocab.items():
            idf_values[idx] = math.log((1 + n_docs) / (1 + df[term])) + 1.0
        self._idf = idf_values
        self._fitted = True

    def embed(self, text: str) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("Call fit() before embed().")
        vec = np.zeros(len(self._vocab), dtype=np.float64)
        tokens = tokenize(text, lowercase=self.config.lowercase)
        for token in tokens:
            idx = self._vocab.get(token)
            if idx is not None:
                vec[idx] += 1.0

        if self.config.sublinear_tf:
            mask = vec > 0
            vec[mask] = 1.0 + np.log(vec[mask])

        vec *= self._idf
        return l2_normalize(vec)


# ---------------------------------------------------------------------------
# Bag-of-Words
# ---------------------------------------------------------------------------


class BowEmbedder(BaseEmbedder):
    """Bag-of-Words embedder."""

    name = "bow"

    def __init__(self, config: BowConfig | None = None) -> None:
        self.config = config or BowConfig()
        self._vocab: dict[str, int] = {}
        self._fitted = False

    def fit(self, documents: Sequence[str]) -> None:
        vocab_set: set[str] = set()
        for doc in documents:
            vocab_set.update(tokenize(doc, lowercase=self.config.lowercase))
        self._vocab = {term: idx for idx, term in enumerate(sorted(vocab_set))}
        self._fitted = True

    def embed(self, text: str) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("Call fit() before embed().")
        vec = np.zeros(len(self._vocab), dtype=np.float64)
        tokens = tokenize(text, lowercase=self.config.lowercase)
        for token in tokens:
            idx = self._vocab.get(token)
            if idx is not None:
                if self.config.binary:
                    vec[idx] = 1.0
                else:
                    vec[idx] += 1.0
        if self.config.normalize:
            vec = l2_normalize(vec)
        return vec


# ---------------------------------------------------------------------------
# Character N-gram (hashing trick)
# ---------------------------------------------------------------------------


class NgramEmbedder(BaseEmbedder):
    """Character n-gram embedder using the hashing trick."""

    name = "ngram"

    def __init__(self, config: NgramConfig | None = None) -> None:
        self.config = config or NgramConfig()
        self._fitted = False

    def fit(self, documents: Sequence[str]) -> None:  # noqa: ARG002
        """N-gram embedder does not need fitting but conforms to the interface."""
        self._fitted = True

    def embed(self, text: str) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("Call fit() before embed().")
        dim = self.config.hash_dim
        vec = np.zeros(dim, dtype=np.float64)
        grams = char_ngrams(
            text,
            n_min=self.config.n_min,
            n_max=self.config.n_max,
            lowercase=self.config.lowercase,
        )
        for gram in grams:
            idx = hash(gram) % dim
            vec[idx] += 1.0
        return l2_normalize(vec)


# ---------------------------------------------------------------------------
# Retrieval evaluator
# ---------------------------------------------------------------------------


def evaluate_retrieval(
    embedder: BaseEmbedder,
    corpus: Sequence[str],
    queries: Sequence[str],
    relevance: dict[int, list[int]],
    top_k: int = 5,
) -> dict[str, float]:
    """Evaluate an embedder's retrieval quality on a corpus.

    Args:
        embedder: A fitted embedder instance.
        corpus: List of documents.
        queries: List of query strings.
        relevance: Mapping from query index to list of relevant doc indices.
        top_k: Number of top results for precision/recall.

    Returns:
        Dict with mean_precision, mean_recall, and mean_mrr.
    """
    corpus_vecs = np.array([embedder.embed(doc) for doc in corpus])

    precisions: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []

    for q_idx, query in enumerate(queries):
        q_vec = embedder.embed(query)
        # Compute similarities
        sims = np.array([cosine_similarity(q_vec, c_vec) for c_vec in corpus_vecs])
        ranked = list(np.argsort(-sims))

        rel_set = set(relevance.get(q_idx, []))
        precisions.append(precision_at_k(ranked, rel_set, top_k))
        recalls.append(recall_at_k(ranked, rel_set, top_k))
        mrrs.append(mean_reciprocal_rank(ranked, rel_set))

    n = max(len(queries), 1)
    return {
        "mean_precision": sum(precisions) / n,
        "mean_recall": sum(recalls) / n,
        "mean_mrr": sum(mrrs) / n,
    }


# ---------------------------------------------------------------------------
# EmbedBench orchestrator
# ---------------------------------------------------------------------------


class EmbedBench:
    """High-level benchmark runner that compares multiple embedders."""

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        self.config = config or BenchmarkConfig()
        self._results: dict[str, dict[str, float]] = {}

    def _build_embedders(self) -> list[BaseEmbedder]:
        mapping: dict[str, BaseEmbedder] = {
            "tfidf": TfIdfEmbedder(self.config.tfidf),
            "bow": BowEmbedder(self.config.bow),
            "ngram": NgramEmbedder(self.config.ngram),
        }
        return [mapping[name] for name in self.config.embedders if name in mapping]

    def benchmark(
        self,
        corpus: Sequence[str],
        queries: Sequence[str],
        relevance: dict[int, list[int]],
    ) -> dict[str, dict[str, float]]:
        """Run retrieval evaluation for every configured embedder.

        Returns:
            Nested dict ``{embedder_name: {metric: value}}``.
        """
        embedders = self._build_embedders()
        results: dict[str, dict[str, float]] = {}
        for emb in embedders:
            emb.fit(list(corpus))
            metrics = evaluate_retrieval(emb, corpus, queries, relevance, top_k=self.config.top_k)
            results[emb.name] = metrics
        self._results = results
        return results

    def get_report(self) -> str:
        """Return a human-readable report of the last benchmark run."""
        if not self._results:
            return "No benchmark results yet. Call benchmark() first."

        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("EmbedBench Retrieval Report")
        lines.append("=" * 60)
        header = f"{'Embedder':<12} {'Precision':>10} {'Recall':>10} {'MRR':>10}"
        lines.append(header)
        lines.append("-" * 60)
        for name, metrics in self._results.items():
            row = (
                f"{name:<12} "
                f"{metrics['mean_precision']:>10.4f} "
                f"{metrics['mean_recall']:>10.4f} "
                f"{metrics['mean_mrr']:>10.4f}"
            )
            lines.append(row)
        lines.append("=" * 60)
        return "\n".join(lines)
