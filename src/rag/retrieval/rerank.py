"""Cross-encoder reranking.

A bi-encoder (bge, used for dense retrieval) embeds query and passage
*separately* — fast, indexable, but it never sees the two together. A
cross-encoder scores the (query, passage) PAIR jointly, so it models term
interactions the bi-encoder can't — more accurate, but O(candidates) model
calls per query and no precomputation. So the standard split: bi-encoder +
BM25 retrieve a wide candidate pool cheaply, cross-encoder reranks the pool
precisely. This is where the precision hybrid traded away comes back.
"""

from __future__ import annotations

from typing import Protocol

from rag.models import RetrievedChunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = self._model.predict([(query, c.text) for c in chunks], show_progress_bar=False)
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda cs: cs[1], reverse=True)
        return [c.model_copy(update={"score": float(s)}) for c, s in ranked]


class FakeReranker:
    """Deterministic reranker for tests: scores by query-term overlap so a
    keyword-matching chunk provably gets promoted, no model download."""

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        terms = {t.lower() for t in query.split()}

        def overlap(chunk: RetrievedChunk) -> int:
            words = {w.lower() for w in chunk.text.split()}
            return len(terms & words)

        ranked = sorted(chunks, key=overlap, reverse=True)
        return [c.model_copy(update={"score": float(overlap(c))}) for c in ranked]


class RerankRetriever:
    """Wrap a base retriever: pull a wide pool, rerank it, return the top-k."""

    def __init__(self, base, reranker: Reranker, pool_size: int = 50):
        self._base = base
        self._reranker = reranker
        self._pool_size = pool_size

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        import anyio.to_thread

        candidates = await self._base.retrieve(query, self._pool_size)
        reranked = await anyio.to_thread.run_sync(self._reranker.rerank, query, candidates)
        return reranked[:k]


def build_reranker(backend: str, model_name: str) -> Reranker:
    if backend == "cross-encoder":
        return CrossEncoderReranker(model_name)
    if backend == "fake":
        return FakeReranker()
    raise ValueError(f"unknown reranker backend: {backend}")
