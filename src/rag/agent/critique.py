"""Retrieval-confidence self-critique.

Cheap heuristic FIRST (no LLM): the cross-encoder's top score is a calibrated-
ish relevance signal, so a low top score means "retrieval probably missed" and
is worth a re-query. Escalating to an LLM critique on every query would burn the
budget the agentic loop is supposed to spend sparingly — the whole point is to
act only when a free signal says it might help.

Threshold calibrated on the golden set's rerank scores: hits skew high (median
~3.9) and misses low (median ~3.0) with overlap, so ~2.0 catches the clearly-
weak retrievals without re-querying everything.
"""

from __future__ import annotations

from rag.models import RetrievedChunk

CONFIDENCE_THRESHOLD = 2.0


def top_score(chunks: list[RetrievedChunk]) -> float:
    return chunks[0].score if chunks else float("-inf")


def is_confident(chunks: list[RetrievedChunk], threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    return top_score(chunks) >= threshold
