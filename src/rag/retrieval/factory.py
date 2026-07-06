"""Retriever abstraction + registry, shared by the API and the eval runner.

A Retriever maps (query, k) -> ranked chunks. Modes compose: hybrid wraps
dense+sparse, and later phases wrap a base retriever with reranking (Phase 5)
and the agentic loop (Phase 6). Keeping this in rag/ (not eval/) lets the
serving path and the measurement path run the exact same retrieval code.
"""

from __future__ import annotations

from typing import Protocol

import anyio.to_thread
from psycopg_pool import AsyncConnectionPool

from rag.ingest.embedder import Embedder
from rag.models import RetrievedChunk
from rag.retrieval.dense import dense_search
from rag.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from rag.retrieval.rerank import Reranker, RerankRetriever, build_reranker
from rag.retrieval.sparse import sparse_search

# candidates each retriever contributes to fusion before taking top-k
FANOUT = 50
# candidates the cross-encoder reranks down to top-k
RERANK_POOL = 50


class Retriever(Protocol):
    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]: ...


class DenseRetriever:
    def __init__(self, pool: AsyncConnectionPool, embedder: Embedder, chunk_config_hash: str):
        self._pool = pool
        self._embedder = embedder
        self._cfg = chunk_config_hash

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        vector = await anyio.to_thread.run_sync(self._embedder.encode_query, query)
        return await dense_search(self._pool, vector, self._cfg, k)


class SparseRetriever:
    def __init__(self, pool: AsyncConnectionPool, chunk_config_hash: str):
        self._pool = pool
        self._cfg = chunk_config_hash

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return await sparse_search(self._pool, query, self._cfg, k)


class HybridRetriever:
    """Dense + sparse candidates fused by RRF — the interview-story retriever."""

    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedder: Embedder,
        chunk_config_hash: str,
        rrf_k: int = DEFAULT_RRF_K,
        fanout: int = FANOUT,
    ):
        self._dense = DenseRetriever(pool, embedder, chunk_config_hash)
        self._sparse = SparseRetriever(pool, chunk_config_hash)
        self._rrf_k = rrf_k
        self._fanout = fanout

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        dense = await self._dense.retrieve(query, self._fanout)
        sparse = await self._sparse.retrieve(query, self._fanout)
        return reciprocal_rank_fusion([dense, sparse], k=self._rrf_k, top_k=k)


def build_retriever(
    mode: str,
    pool: AsyncConnectionPool,
    embedder: Embedder,
    chunk_config_hash: str,
    reranker: Reranker | None = None,
    reranker_backend: str = "cross-encoder",
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> Retriever:
    if mode == "dense":
        return DenseRetriever(pool, embedder, chunk_config_hash)
    if mode == "sparse":
        return SparseRetriever(pool, chunk_config_hash)
    if mode == "hybrid":
        return HybridRetriever(pool, embedder, chunk_config_hash)
    # rerank a base retriever's wide pool. rerank == hybrid+rerank (headline
    # pipeline); rerank-dense isolates whether hybrid's recall gain survives
    # reranking or dense+rerank would have done as well.
    if mode in ("rerank", "rerank-dense"):
        base_mode = "dense" if mode == "rerank-dense" else "hybrid"
        base = build_retriever(base_mode, pool, embedder, chunk_config_hash)
        reranker = reranker or build_reranker(reranker_backend, reranker_model)
        return RerankRetriever(base, reranker, pool_size=RERANK_POOL)
    raise ValueError(f"unknown retrieval mode: {mode!r}")
