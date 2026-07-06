"""Query orchestration: retrieve -> generate, with per-stage timings.

Retrieval is a pluggable Retriever (dense / sparse / hybrid / rerank / agentic),
so serving and eval share one code path and the API's retrieval mode is a config
flag, not a rewrite.
"""

from __future__ import annotations

from time import perf_counter

from rag.generation.client import OllamaClient
from rag.generation.prompts import REFUSAL, SYSTEM_ANSWER, build_user_prompt
from rag.models import QueryResult, StageTimings
from rag.retrieval.factory import Retriever


class RAGService:
    def __init__(self, retriever: Retriever, llm: OllamaClient, top_k: int):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    async def answer(self, question: str, top_k: int | None = None) -> QueryResult:
        k = top_k or self.top_k
        timings = StageTimings()

        t0 = perf_counter()
        chunks = await self.retriever.retrieve(question, k)
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
