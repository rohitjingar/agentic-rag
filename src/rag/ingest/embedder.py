"""Embedding backends.

- SentenceTransformerEmbedder: bge-small-en-v1.5, normalized vectors (cosine),
  MPS/CPU. bge expects a query instruction prefix on QUERIES ONLY; passages
  are encoded bare.
- FakeEmbedder: deterministic hash-seeded vectors for fast tests and CI paths
  that must not download models.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Protocol

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dim: int

    def encode_passages(self, texts: list[str]) -> list[list[float]]: ...

    def encode_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dim: int = 384):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = dim

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    def encode_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            _BGE_QUERY_PREFIX + text, normalize_embeddings=True, show_progress_bar=False
        )
        return vector.tolist()


class FakeEmbedder:
    """Deterministic pseudo-embeddings: same text -> same unit vector."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _encode(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        rng = random.Random(seed)
        raw = [rng.gauss(0, 1) for _ in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._encode(text)


def build_embedder(backend: str, model_name: str, dim: int) -> Embedder:
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name, dim)
    if backend == "fake":
        return FakeEmbedder(dim)
    raise ValueError(f"unknown embedding backend: {backend}")
