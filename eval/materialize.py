"""Resolve golden labels (doc_id, verbatim locator) to chunk ids for the live
chunk config, and enforce the no-leakage rule.

Materialization fails LOUDLY: a locator that matches zero chunks means the
label is stale (corpus or chunk config changed), and the runner must refuse to
report numbers computed against broken labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import psycopg

from eval.schema import GoldenItem

_WORD = re.compile(r"\w+")


@dataclass
class MaterializedItem:
    item: GoldenItem
    # chunk_id -> grade ("primary"/"supporting")
    grade_by_chunk: dict[str, str]
    leakage_overlap: float  # max n-gram overlap question vs any gold chunk

    @property
    def primary_chunk_ids(self) -> set[str]:
        return {cid for cid, g in self.grade_by_chunk.items() if g == "primary"}


class MaterializationError(RuntimeError):
    pass


def _shingles(text: str, n: int) -> set[str]:
    words = [w.lower() for w in _WORD.findall(text)]
    return (
        {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}
        if len(words) >= n
        else set()
    )


def ngram_overlap(question: str, chunk_text: str, n: int = 5) -> float:
    """Fraction of the question's n-gram shingles that also appear in the chunk.

    High overlap => the question was likely lifted from the chunk wording
    (leakage). We flag, never silently pass.
    """
    q = _shingles(question, n)
    if not q:
        return 0.0
    c = _shingles(chunk_text, n)
    return len(q & c) / len(q)


def materialize_item(
    conn: psycopg.Connection, item: GoldenItem, chunk_config_hash: str
) -> MaterializedItem:
    grade_by_chunk: dict[str, str] = {}
    max_overlap = 0.0

    for label in item.gold:
        rows = conn.execute(
            """
            SELECT id, text FROM chunks
            WHERE doc_id = %s AND chunk_config_hash = %s AND text LIKE %s
            ORDER BY ord
            """,
            (label.doc_id, chunk_config_hash, f"%{label.locator}%"),
        ).fetchall()
        if not rows:
            raise MaterializationError(
                f"{item.qid}: locator not found in any chunk of {label.doc_id!r} "
                f"(config {chunk_config_hash}): {label.locator!r}"
            )
        for chunk_id, text in rows:
            # primary wins if a chunk is claimed by both grades
            if grade_by_chunk.get(chunk_id) != "primary":
                grade_by_chunk[chunk_id] = label.grade
            max_overlap = max(max_overlap, ngram_overlap(item.question, text))

    return MaterializedItem(item=item, grade_by_chunk=grade_by_chunk, leakage_overlap=max_overlap)


def materialize_all(
    conn: psycopg.Connection,
    items: list[GoldenItem],
    chunk_config_hash: str,
    leakage_threshold: float = 0.5,
) -> list[MaterializedItem]:
    materialized = []
    leaks = []
    for item in items:
        if not item.answerable:
            materialized.append(MaterializedItem(item=item, grade_by_chunk={}, leakage_overlap=0.0))
            continue
        m = materialize_item(conn, item, chunk_config_hash)
        if m.leakage_overlap >= leakage_threshold:
            leaks.append(f"{item.qid} (overlap {m.leakage_overlap:.2f})")
        materialized.append(m)
    if leaks:
        raise MaterializationError(
            "possible question/chunk leakage (>= "
            f"{leakage_threshold} 5-gram overlap): {', '.join(leaks)}"
        )
    return materialized
