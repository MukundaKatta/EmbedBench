# Architecture

## Overview

EmbedBench is a lightweight Python toolkit for benchmarking text embedding strategies. It compares TF-IDF, bag-of-words, and character n-gram embeddings on retrieval tasks, producing precision, recall, and MRR metrics.

## Module Structure

```
src/embedbench/
├── __init__.py      # Public API exports
├── config.py        # Pydantic configuration models
├── core.py          # Embedder implementations and benchmark runner
└── utils.py         # Tokenisation, similarity, and metric helpers
```

## Key Components

### Embedders

All embedders inherit from `BaseEmbedder` and implement two methods:

- **`fit(documents)`** — learn vocabulary or statistics from a corpus
- **`embed(text)`** — return a numpy vector for a given text

| Embedder | Strategy | Vocabulary | Dimensionality |
|---|---|---|---|
| `TfIdfEmbedder` | TF-IDF weighting | Learned from corpus | Number of unique terms |
| `BowEmbedder` | Raw / binary word counts | Learned from corpus | Number of unique terms |
| `NgramEmbedder` | Character n-grams + hashing trick | None (hash-based) | Configurable (default 5000) |

### Configuration

Each embedder has a Pydantic model (`TfIdfConfig`, `BowConfig`, `NgramConfig`) that validates parameters at construction time. The top-level `BenchmarkConfig` composes all three plus benchmark-level settings like `top_k`.

### Evaluation Pipeline

```
corpus + queries + relevance labels
           │
           ▼
   ┌───────────────┐
   │  EmbedBench    │
   │  .benchmark()  │
   └───────┬───────┘
           │  for each embedder:
           ▼
   ┌───────────────┐
   │  fit(corpus)   │
   └───────┬───────┘
           │
           ▼
   ┌───────────────────┐
   │ evaluate_retrieval │──▶ embed corpus + queries
   └───────┬───────────┘     rank by cosine similarity
           │                 compute precision@k, recall@k, MRR
           ▼
   ┌───────────────┐
   │   get_report() │──▶ formatted comparison table
   └───────────────┘
```

### Vector Math

All vector operations use numpy:

- **Cosine similarity** — `dot(a, b) / (||a|| * ||b||)`
- **L2 normalisation** — `v / ||v||`
- **Ranking** — `np.argsort(-similarities)`

## Design Decisions

1. **No external API calls** — all embeddings are computed locally, making benchmarks reproducible and fast.
2. **Hashing trick for n-grams** — avoids building an explicit vocabulary, keeping memory usage bounded.
3. **Pydantic configs** — catches invalid parameters early with clear error messages.
4. **Fit/embed pattern** — mirrors scikit-learn conventions for familiarity.
