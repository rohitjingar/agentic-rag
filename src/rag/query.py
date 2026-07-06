"""Query orchestration: (cache) -> retrieve -> generate, with per-stage timings,
OTel spans, semantic caching, and per-query shadow-cost.

Retrieval is a pluggable Retriever (dense / sparse / hybrid / rerank / agentic),
so serving and eval share one code path and the API's retrieval mode is a config
flag, not a rewrite. A semantic-cache hit short-circuits both expensive stages.
"""

from __future__ import annotations

from time import perf_counter

from rag.cache.semantic import SemanticCache
from rag.generation.client import OllamaClient
from rag.generation.prompts import REFUSAL, SYSTEM_ANSWER, build_user_prompt
from rag.models import QueryResult, StageTimings
from rag.obs.cost import shadow_usd
from rag.obs.otel import set_attrs, span
from rag.retrieval.factory import Retriever


class RAGService:
    def __init__(
        self,
        retriever: Retriever,
        llm: OllamaClient,
        top_k: int,
        cache: SemanticCache | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.cache = cache

    async def answer(self, question: str, top_k: int | None = None) -> QueryResult:
        k = top_k or self.top_k
        timings = StageTimings()

        with span("rag.query", **{"rag.question_len": len(question)}) as root:
            if self.cache is not None:
                hit = await self._cache_get(question)
                if hit is not None:
                    set_attrs(root, **{"rag.cache_hit": True})
                    return QueryResult(
                        answer=hit.answer,
                        sources=hit.sources,
                        tokens_in=0,
                        tokens_out=0,
                        timings=timings,
                        cached=True,
                        shadow_usd=0.0,  # a hit skips the LLM entirely
                    )
                set_attrs(root, **{"rag.cache_hit": False})

            t0 = perf_counter()
            with span("retrieve") as s:
                chunks = await self.retriever.retrieve(question, k)
                set_attrs(s, **{"retrieve.n_chunks": len(chunks)})
            timings.retrieve_ms = (perf_counter() - t0) * 1000

            if not chunks:
                return QueryResult(
                    answer=REFUSAL, sources=[], tokens_in=0, tokens_out=0, timings=timings
                )

            t0 = perf_counter()
            with span("generate") as s:
                response = await self.llm.chat(SYSTEM_ANSWER, build_user_prompt(question, chunks))
                set_attrs(
                    s,
                    **{
                        "gen.tokens_in": response.tokens_in,
                        "gen.tokens_out": response.tokens_out,
                    },
                )
            timings.generate_ms = (perf_counter() - t0) * 1000

            cost = shadow_usd(response.tokens_in, response.tokens_out)
            set_attrs(root, **{"rag.shadow_usd": cost})
            result = QueryResult(
                answer=response.text,
                sources=chunks,
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
                timings=timings,
                cached=False,
                shadow_usd=cost,
            )

            if self.cache is not None:
                await self.cache.put(
                    question, result.answer, chunks, response.tokens_in, response.tokens_out
                )
            return result

    async def _cache_get(self, question: str):
        with span("cache.get") as s:
            hit = await self.cache.get(question)
            set_attrs(s, **{"cache.hit": hit is not None})
            return hit
