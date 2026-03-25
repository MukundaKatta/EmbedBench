"""Configuration models for EmbedBench using Pydantic."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TfIdfConfig(BaseModel):
    """Configuration for the TF-IDF embedder."""

    lowercase: bool = Field(default=True, description="Convert text to lowercase before tokenisation.")
    min_df: int = Field(default=1, ge=1, description="Minimum document frequency for a term to be included.")
    max_df_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum document-frequency ratio (0-1). Terms appearing in more than this fraction of documents are dropped.",
    )
    sublinear_tf: bool = Field(default=False, description="Apply sublinear TF scaling (1 + log(tf)).")


class BowConfig(BaseModel):
    """Configuration for the Bag-of-Words embedder."""

    lowercase: bool = Field(default=True, description="Convert text to lowercase before tokenisation.")
    binary: bool = Field(default=False, description="If True use binary counts (0/1) instead of raw frequencies.")
    normalize: bool = Field(default=True, description="L2-normalise the output vectors.")


class NgramConfig(BaseModel):
    """Configuration for the character n-gram embedder."""

    n_min: int = Field(default=2, ge=1, description="Minimum character n-gram length.")
    n_max: int = Field(default=4, ge=1, description="Maximum character n-gram length.")
    hash_dim: int = Field(default=5000, ge=64, description="Dimensionality of the hashed output vector.")
    lowercase: bool = Field(default=True, description="Convert text to lowercase before extraction.")


class BenchmarkConfig(BaseModel):
    """Configuration for the benchmarking pipeline."""

    top_k: int = Field(default=5, ge=1, description="Number of top results to consider for retrieval metrics.")
    embedders: list[str] = Field(
        default=["tfidf", "bow", "ngram"],
        description="Which embedders to include in the benchmark run.",
    )
    tfidf: TfIdfConfig = Field(default_factory=TfIdfConfig)
    bow: BowConfig = Field(default_factory=BowConfig)
    ngram: NgramConfig = Field(default_factory=NgramConfig)
