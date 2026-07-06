"""Reranking must be able to PROMOTE a better chunk the base ranked low.
Uses the deterministic FakeReranker (query-term overlap) so this runs in CI.
"""

import asyncio

from rag.models import RetrievedChunk
from rag.retrieval.rerank import FakeReranker, RerankRetriever


class StubBase:
    """A base retriever that returns chunks in a deliberately-bad order."""

    def __init__(self, chunks: list[RetrievedChunk]):
        self._chunks = chunks

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return self._chunks[:k]


def chunk(cid: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(id=cid, doc_id="d", ord=0, text=text, score=score)


def test_reranker_promotes_the_relevant_chunk():
    # base ranks an off-topic chunk first; the truly relevant one is buried
    base = StubBase(
        [
            chunk("c0", "completely unrelated text about weather and cooking", 0.9),
            chunk("c1", "vacuum reclaims dead tuple disk space in postgres", 0.5),
            chunk("c2", "some other tangential note about networking", 0.4),
        ]
    )
    retriever = RerankRetriever(base, FakeReranker(), pool_size=10)
    out = asyncio.run(retriever.retrieve("how does vacuum reclaim disk space", 3))
    assert out[0].id == "c1"  # promoted from rank 2 to rank 1


def test_reranker_truncates_to_k():
    base = StubBase([chunk(f"c{i}", f"text term{i} vacuum", 0.5) for i in range(10)])
    retriever = RerankRetriever(base, FakeReranker(), pool_size=10)
    out = asyncio.run(retriever.retrieve("vacuum", 3))
    assert len(out) == 3


def test_reranker_handles_empty_pool():
    retriever = RerankRetriever(StubBase([]), FakeReranker(), pool_size=10)
    assert asyncio.run(retriever.retrieve("anything", 5)) == []


def test_rerank_mode_built_with_injected_reranker():
    # build_retriever wires rerank over a base; injected fake avoids a model load
    from rag.retrieval.factory import build_retriever
    from rag.retrieval.rerank import RerankRetriever as RR

    r = build_retriever(
        "rerank", pool=None, embedder=None, chunk_config_hash="x", reranker=FakeReranker()
    )
    assert isinstance(r, RR)
