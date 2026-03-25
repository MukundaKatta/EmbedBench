"""Tests for EmbedBench core functionality."""

from __future__ import annotations

import numpy as np
import pytest

from embedbench import (
    BowEmbedder,
    EmbedBench,
    NgramEmbedder,
    TfIdfEmbedder,
    evaluate_retrieval,
)
from embedbench.utils import cosine_similarity, precision_at_k, recall_at_k, mean_reciprocal_rank


CORPUS = [
    "the cat sat on the mat",
    "the dog chased the cat",
    "birds fly in the sky",
    "fish swim in the sea",
]


class TestTfIdfEmbedder:
    def test_embed_returns_vector(self) -> None:
        emb = TfIdfEmbedder()
        emb.fit(CORPUS)
        vec = emb.embed("cat sat")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert len(vec) > 0

    def test_similar_texts_higher_similarity(self) -> None:
        emb = TfIdfEmbedder()
        emb.fit(CORPUS)
        sim_high = emb.compare("the cat sat", "the cat chased")
        sim_low = emb.compare("the cat sat", "fish swim in the sea")
        assert sim_high > sim_low

    def test_embed_without_fit_raises(self) -> None:
        emb = TfIdfEmbedder()
        with pytest.raises(RuntimeError, match="fit"):
            emb.embed("hello")


class TestBowEmbedder:
    def test_embed_returns_normalized_vector(self) -> None:
        emb = BowEmbedder()
        emb.fit(CORPUS)
        vec = emb.embed("the cat sat on the mat")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-6

    def test_compare_identical_texts(self) -> None:
        emb = BowEmbedder()
        emb.fit(CORPUS)
        sim = emb.compare("cat dog", "cat dog")
        assert abs(sim - 1.0) < 1e-6


class TestNgramEmbedder:
    def test_embed_dimension(self) -> None:
        emb = NgramEmbedder()
        emb.fit(CORPUS)
        vec = emb.embed("hello world")
        assert len(vec) == 5000

    def test_similar_substrings_similar_vectors(self) -> None:
        emb = NgramEmbedder()
        emb.fit(CORPUS)
        sim = emb.compare("running", "runner")
        # Both share "run", "unn", "nn" etc.
        assert sim > 0.3


class TestEvaluateRetrieval:
    def test_perfect_retrieval(self) -> None:
        emb = TfIdfEmbedder()
        emb.fit(CORPUS)
        # Query "cat sat mat" should rank doc 0 first
        metrics = evaluate_retrieval(
            emb,
            corpus=CORPUS,
            queries=["cat sat mat"],
            relevance={0: [0]},
            top_k=1,
        )
        assert metrics["mean_precision"] == 1.0
        assert metrics["mean_recall"] == 1.0
        assert metrics["mean_mrr"] == 1.0


class TestEmbedBench:
    def test_benchmark_returns_all_embedders(self) -> None:
        bench = EmbedBench()
        results = bench.benchmark(
            corpus=CORPUS,
            queries=["the cat"],
            relevance={0: [0, 1]},
        )
        assert "tfidf" in results
        assert "bow" in results
        assert "ngram" in results

    def test_get_report_before_benchmark(self) -> None:
        bench = EmbedBench()
        report = bench.get_report()
        assert "No benchmark" in report

    def test_get_report_after_benchmark(self) -> None:
        bench = EmbedBench()
        bench.benchmark(
            corpus=CORPUS,
            queries=["the cat"],
            relevance={0: [0, 1]},
        )
        report = bench.get_report()
        assert "tfidf" in report
        assert "Precision" in report


class TestUtils:
    def test_cosine_similarity_orthogonal(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == 0.0

    def test_precision_at_k(self) -> None:
        assert precision_at_k([0, 1, 2], {0, 2}, 3) == pytest.approx(2 / 3)

    def test_recall_at_k(self) -> None:
        assert recall_at_k([0, 1, 2], {0, 2, 5}, 3) == pytest.approx(2 / 3)

    def test_mrr(self) -> None:
        assert mean_reciprocal_rank([3, 1, 0], {0}) == pytest.approx(1 / 3)
