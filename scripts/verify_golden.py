"""Verify the golden set against the live corpus + chunk config.

For every answerable item this checks each locator (a) exists verbatim in its
source document and (b) materializes to at least one chunk, and reports the
5-gram leakage overlap. Exits non-zero if anything is broken so it can gate CI.

    uv run python scripts/verify_golden.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg

from eval.materialize import MaterializationError, ngram_overlap
from eval.schema import load_golden
from rag.api.app import chunk_config_from
from rag.config import get_settings
from rag.ingest.loader import load_corpus

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden" / "golden.jsonl"


def main() -> None:
    settings = get_settings()
    cfg_hash = chunk_config_from(settings).config_hash
    items = load_golden(GOLDEN)
    docs = {d.id: d.text for d in load_corpus(Path(settings.corpus_dir))}

    problems: list[str] = []
    max_overlap = 0.0
    n_answerable = 0

    with psycopg.connect(settings.database_url) as conn:
        for item in items:
            if not item.answerable:
                continue
            n_answerable += 1
            for label in item.gold:
                doc_text = docs.get(label.doc_id)
                if doc_text is None:
                    problems.append(f"{item.qid}: doc not found: {label.doc_id}")
                    continue
                if label.locator not in doc_text:
                    problems.append(
                        f"{item.qid}: locator NOT in doc {label.doc_id}: {label.locator!r}"
                    )
                    continue
                rows = conn.execute(
                    "SELECT text FROM chunks WHERE doc_id=%s AND chunk_config_hash=%s "
                    "AND text LIKE %s",
                    (label.doc_id, cfg_hash, f"%{label.locator}%"),
                ).fetchall()
                if not rows:
                    problems.append(
                        f"{item.qid}: locator in doc but spans chunk boundary "
                        f"(no single chunk) in {label.doc_id}: {label.locator!r}"
                    )
                    continue
                overlap = max(ngram_overlap(item.question, r[0]) for r in rows)
                max_overlap = max(max_overlap, overlap)
                if overlap >= 0.5:
                    problems.append(f"{item.qid}: LEAKAGE {overlap:.2f} vs {label.doc_id}")

    print(f"answerable questions: {n_answerable}")
    print(f"negative controls:    {sum(1 for i in items if not i.answerable)}")
    print(f"total:                {len(items)}")
    print(f"max leakage overlap:  {max_overlap:.3f} (threshold 0.50)")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nAll locators verified: exist in source doc, materialize to a chunk, no leakage.")


if __name__ == "__main__":
    try:
        main()
    except MaterializationError as exc:
        sys.exit(f"materialization error: {exc}")
