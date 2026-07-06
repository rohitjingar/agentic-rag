"""Agentic retrieval loop: classify -> (decompose) -> retrieve -> rerank ->
self-critique -> (HyDE re-query), bounded by an iteration cap and a token budget.

Candidates are gathered with possibly-transformed queries, but reranking always
uses the ORIGINAL question — the transforms are a retrieval aid, not a change to
what the user asked. The loop only spends LLM budget when the cheap confidence
signal says the base retrieval looks weak; a confident first pass costs exactly
one classification call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anyio.to_thread
from psycopg_pool import AsyncConnectionPool

from rag.agent.classifier import classify_query
from rag.agent.critique import is_confident, top_score
from rag.agent.transform import decompose, hyde_passage
from rag.generation.client import OllamaClient
from rag.ingest.embedder import Embedder
from rag.models import RetrievedChunk
from rag.retrieval.dense import dense_search
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.rerank import Reranker
from rag.retrieval.sparse import sparse_search

ITER_CAP = 2  # max retrieval rounds (initial + up to one HyDE re-query)
TOKEN_BUDGET = 6000  # per-query LLM budget; transforms stop when exhausted
FANOUT = 50


@dataclass
class AgenticRun:
    classification: str = ""
    iterations: int = 0
    tokens: int = 0
    steps: list[str] = field(default_factory=list)
    final_top_score: float = 0.0
    out_of_scope: bool = False

    def spend(self, tokens: int, step: str) -> None:
        self.tokens += tokens
        self.steps.append(step)

    @property
    def budget_left(self) -> bool:
        return self.tokens < TOKEN_BUDGET


class AgenticRetriever:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedder: Embedder,
        chunk_config_hash: str,
        reranker: Reranker,
        llm: OllamaClient,
        iter_cap: int = ITER_CAP,
    ):
        self._pool = pool
        self._embedder = embedder
        self._cfg = chunk_config_hash
        self._reranker = reranker
        self._llm = llm
        self._iter_cap = iter_cap
        self.last_run: AgenticRun | None = None

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        run = AgenticRun()

        cls, tok = await classify_query(self._llm, query)
        run.spend(tok, "classify")
        run.classification = cls.query_type
        run.out_of_scope = cls.is_out_of_scope

        queries = [query]
        if cls.is_multi_hop and run.budget_left:
            subs, tok = await decompose(self._llm, query)
            run.spend(tok, "decompose")
            queries = [query, *subs]

        candidates = await self._gather(queries)
        reranked = await self._rerank(query, candidates)
        run.iterations = 1

        # self-critique -> HyDE re-query while the cheap signal says "weak"
        while not is_confident(reranked) and run.iterations < self._iter_cap and run.budget_left:
            passage, tok = await hyde_passage(self._llm, query)
            run.spend(tok, "hyde")
            hyde_hits = await self._dense(passage, FANOUT)
            candidates = reciprocal_rank_fusion([candidates, hyde_hits], top_k=FANOUT)
            reranked = await self._rerank(query, candidates)
            run.iterations += 1

        run.final_top_score = top_score(reranked)
        self.last_run = run
        return reranked[:k]

    async def _gather(self, queries: list[str]) -> list[RetrievedChunk]:
        rankings: list[list[RetrievedChunk]] = []
        for q in queries:
            rankings.append(await self._dense(q, FANOUT))
            rankings.append(await sparse_search(self._pool, q, self._cfg, FANOUT))
        return reciprocal_rank_fusion(rankings, top_k=FANOUT)

    async def _dense(self, text: str, k: int) -> list[RetrievedChunk]:
        vector = await anyio.to_thread.run_sync(self._embedder.encode_query, text)
        return await dense_search(self._pool, vector, self._cfg, k)

    async def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return await anyio.to_thread.run_sync(self._reranker.rerank, query, candidates)
