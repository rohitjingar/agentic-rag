"""Retriever registry keyed by mode name.

Phase 2 ships `dense`. Phases 4-6 register `sparse`, `hybrid`, `rerank`,
`agentic` here — the runner and metrics never change, only the mode does.
That is the whole eval-first bet: one measurement harness, many retrievers,
every mode compared on identical golden labels.
"""

from __future__ import annotations

from typing import Protocol

from psycopg_pool import AsyncConnectionPool

from rag.config import Settings
from rag.ingest.embedder import Embedder
from rag.models import RetrievedChunk
from rag.retrieval.dense import dense_search


class Retriever(Protocol):
    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]: ...


class DenseRetriever:
    def __init__(self, pool: AsyncConnectionPool, embedder: Embedder, chunk_config_hash: str):
        self._pool = pool
        self._embedder = embedder
        self._cfg = chunk_config_hash

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        import anyio.to_thread

        vector = await anyio.to_thread.run_sync(self._embedder.encode_query, query)
        return await dense_search(self._pool, vector, self._cfg, k)


def build_retriever(
    mode: str,
    pool: AsyncConnectionPool,
    embedder: Embedder,
    settings: Settings,
    chunk_config_hash: str,
) -> Retriever:
    if mode == "dense":
        return DenseRetriever(pool, embedder, chunk_config_hash)
    raise ValueError(f"unknown retrieval mode: {mode!r}")
