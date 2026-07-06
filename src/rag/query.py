"""Query orchestration: embed -> retrieve -> generate, with per-stage timings."""

from __future__ import annotations

from time import perf_counter

import anyio.to_thread
from psycopg_pool import AsyncConnectionPool

from rag.generation.client import OllamaClient
from rag.generation.prompts import REFUSAL, SYSTEM_ANSWER, build_user_prompt
from rag.ingest.chunker import ChunkConfig
from rag.ingest.embedder import Embedder
from rag.models import QueryResult, StageTimings
from rag.retrieval.dense import dense_search


class RAGService:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedder: Embedder,
        llm: OllamaClient,
        chunk_config: ChunkConfig,
        top_k: int,
    ):
        self.pool = pool
        self.embedder = embedder
        self.llm = llm
        self.chunk_config = chunk_config
        self.top_k = top_k

    async def answer(self, question: str, top_k: int | None = None) -> QueryResult:
        k = top_k or self.top_k
        timings = StageTimings()

        t0 = perf_counter()
        query_vector = await anyio.to_thread.run_sync(self.embedder.encode_query, question)
        timings.embed_ms = (perf_counter() - t0) * 1000

        t0 = perf_counter()
        chunks = await dense_search(self.pool, query_vector, self.chunk_config.config_hash, k)
        timings.retrieve_ms = (perf_counter() - t0) * 1000

        if not chunks:
            return QueryResult(
                answer=REFUSAL, sources=[], tokens_in=0, tokens_out=0, timings=timings
            )

        t0 = perf_counter()
        response = await self.llm.chat(SYSTEM_ANSWER, build_user_prompt(question, chunks))
        timings.generate_ms = (perf_counter() - t0) * 1000

        return QueryResult(
            answer=response.text,
            sources=chunks,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            timings=timings,
        )
