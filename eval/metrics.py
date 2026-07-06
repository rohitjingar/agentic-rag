"""Retrieval metrics: recall@k, MRR, nDCG@k with graded relevance.

Definitions (fixed here so every phase compares apples to apples):

- recall@k   = |top-k ∩ primary gold| / |primary gold|
               "of the chunks that directly answer, how many did we retrieve?"
- MRR        = 1 / rank of the FIRST primary gold chunk (0 if none in top-k)
               "how high did the first right answer land?"
- nDCG@k     = DCG / ideal-DCG, gains primary=2, supporting=1
               "did we rank the important chunks above the merely-related ones?"

Graded relevance is why nDCG earns its place: recall/MRR treat every gold chunk
the same, nDCG rewards putting the primary chunk above a supporting one.
"""

from __future__ import annotations

from math import log2

GRADE_GAIN = {"primary": 2.0, "supporting": 1.0}


def recall_at_k(retrieved_ids: list[str], primary_gold_ids: set[str], k: int) -> float:
    if not primary_gold_ids:
        return float("nan")  # undefined for negative controls; callers exclude these
    hits = sum(1 for cid in retrieved_ids[:k] if cid in primary_gold_ids)
    return hits / len(primary_gold_ids)


def reciprocal_rank(retrieved_ids: list[str], primary_gold_ids: set[str], k: int) -> float:
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in primary_gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], grade_by_id: dict[str, str], k: int) -> float:
    if not grade_by_id:
        return float("nan")
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        gain = GRADE_GAIN.get(grade_by_id.get(cid, ""), 0.0)
        dcg += gain / log2(rank + 1)
    ideal_gains = sorted((GRADE_GAIN[g] for g in grade_by_id.values()), reverse=True)
    idcg = sum(g / log2(rank + 1) for rank, g in enumerate(ideal_gains[:k], start=1))
    return dcg / idcg if idcg else 0.0


def mean_ignoring_nan(values: list[float]) -> float:
    real = [v for v in values if v == v]  # NaN != NaN
    return sum(real) / len(real) if real else 0.0
