"""Reciprocal Rank Fusion (RRF) — combine dense + sparse rankings by RANK, not
score, so the two retrievers' incomparable score scales never need calibrating.

    RRF(chunk) = sum over rankers of 1 / (k + rank_in_that_ranker)

k (default 60, from Cormack et al. 2009) damps the contribution of low ranks so
a chunk near the top of either list gets most of the benefit. A chunk found by
BOTH retrievers accumulates from both, which is exactly the hybrid win: dense
supplies semantic matches, sparse supplies exact-term matches, and agreement is
rewarded.
"""

from __future__ import annotations

from rag.models import RetrievedChunk

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    fused: dict[str, float] = {}
    chunk_by_id: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (k + rank)
            chunk_by_id.setdefault(chunk.id, chunk)

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        ordered = ordered[:top_k]
    # carry the RRF score through so downstream (rerank, critique) can see it
    return [chunk_by_id[cid].model_copy(update={"score": score}) for cid, score in ordered]
