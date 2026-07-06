"""Retrieval eval runner — one command, reproducible, results logged over time.

    uv run python -m eval.run --mode dense --label dense-baseline

Loads the golden set, materializes labels against the LIVE chunk config (fails
loudly on stale/leaky labels), runs the chosen retriever per question, computes
recall@k / MRR / nDCG@10, and writes a timestamped JSON + appends a row to
eval/results/RESULTS.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import psycopg
from psycopg_pool import AsyncConnectionPool

from eval.materialize import MaterializedItem, materialize_all
from eval.metrics import mean_ignoring_nan, ndcg_at_k, recall_at_k, reciprocal_rank
from eval.retrievers import build_retriever
from eval.schema import load_golden
from rag.api.app import chunk_config_from
from rag.config import get_settings
from rag.generation.client import OllamaClient
from rag.ingest.embedder import build_embedder
from rag.ingest.pipeline import corpus_version

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVAL_DIR / "golden" / "golden.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
TOP_K_RETRIEVE = 10


@dataclass
class QuestionResult:
    qid: str
    qtype: str
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    top_score: float
    retrieve_ms: float  # end-to-end: embed query + retrieve (+ rerank, later phases)
    tokens: int = 0  # LLM tokens spent on agentic transforms (0 for non-agentic)
    iterations: int = 1
    classification: str = ""


async def _eval_question(retriever, m: MaterializedItem) -> QuestionResult:
    item = m.item
    t0 = perf_counter()
    chunks = await retriever.retrieve(item.question, TOP_K_RETRIEVE)
    retrieve_ms = (perf_counter() - t0) * 1000

    ids = [c.id for c in chunks]
    primary = m.primary_chunk_ids
    run = getattr(retriever, "last_run", None)  # populated by AgenticRetriever
    return QuestionResult(
        qid=item.qid,
        qtype=item.qtype,
        recall_at_5=recall_at_k(ids, primary, 5),
        recall_at_10=recall_at_k(ids, primary, 10),
        mrr=reciprocal_rank(ids, primary, TOP_K_RETRIEVE),
        ndcg_at_10=ndcg_at_k(ids, m.grade_by_chunk, 10),
        top_score=chunks[0].score if chunks else 0.0,
        retrieve_ms=retrieve_ms,
        tokens=run.tokens if run else 0,
        iterations=run.iterations if run else 1,
        classification=run.classification if run else "",
    )


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((p / 100) * (len(ordered) - 1)))
    return ordered[idx]


def _aggregate(results: list[QuestionResult]) -> dict:
    answerable = [r for r in results if not r.qtype.startswith("negative")]
    by_type: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in answerable:
        by_type[r.qtype].append(r)

    def block(rs: list[QuestionResult]) -> dict:
        return {
            "n": len(rs),
            "recall@5": round(mean_ignoring_nan([r.recall_at_5 for r in rs]), 4),
            "recall@10": round(mean_ignoring_nan([r.recall_at_10 for r in rs]), 4),
            "mrr": round(mean_ignoring_nan([r.mrr for r in rs]), 4),
            "ndcg@10": round(mean_ignoring_nan([r.ndcg_at_10 for r in rs]), 4),
        }

    retrieve_ms = [r.retrieve_ms for r in results]
    tokens = [r.tokens for r in answerable]
    mean_tokens = round(sum(tokens) / len(tokens), 1) if tokens else 0.0
    return {
        "overall": block(answerable),
        "by_qtype": {t: block(rs) for t, rs in sorted(by_type.items())},
        "latency_ms": {
            "retrieve_p50": round(_percentile(retrieve_ms, 50), 1),
            "retrieve_p95": round(_percentile(retrieve_ms, 95), 1),
        },
        "cost": {
            "mean_tokens_per_query": mean_tokens,
            # shadow-$: local run costs $0, but price tokens at a public small-
            # model blended rate (~$0.60 / 1M tokens) to keep the cost story
            # concrete. per 1k queries = tokens/q * 1000 * 0.60/1e6.
            "shadow_usd_per_1k_queries": round(mean_tokens * 1000 * 0.60 / 1_000_000, 4),
            "mean_iterations": round(sum(r.iterations for r in answerable) / len(answerable), 2)
            if answerable
            else 0.0,
        },
    }


async def run_eval(mode: str, label: str, provisional: bool) -> dict:
    settings = get_settings()
    chunk_config = chunk_config_from(settings)
    cfg_hash = chunk_config.config_hash
    items = load_golden(GOLDEN_PATH)

    # config-hash guard: labels must materialize against the live chunk config
    with psycopg.connect(settings.database_url) as conn:
        (chunk_total,) = conn.execute(
            "SELECT count(*) FROM chunks WHERE chunk_config_hash = %s", (cfg_hash,)
        ).fetchone()
        if chunk_total == 0:
            raise RuntimeError(f"no chunks for live config {cfg_hash} — ingest before evaluating")
        materialized = materialize_all(conn, items, cfg_hash)

    embedder = build_embedder(
        settings.embedding_backend, settings.embedding_model, settings.embedding_dim
    )
    pool = AsyncConnectionPool(settings.database_url, min_size=1, max_size=4, open=False)
    await pool.open()
    llm = None
    if mode == "agentic":
        llm = OllamaClient(
            settings.ollama_base_url, settings.generation_model, num_ctx=settings.llm_num_ctx
        )
    try:
        retriever = build_retriever(mode, pool, embedder, cfg_hash, llm=llm)
        results = [await _eval_question(retriever, m) for m in materialized if m.item.answerable]
    finally:
        if llm is not None:
            await llm.aclose()
        await pool.close()

    agg = _aggregate(results)
    report = {
        "label": label,
        "mode": mode,
        "provisional": provisional,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus_version": corpus_version(Path(settings.corpus_dir)),
        "chunk_config_hash": cfg_hash,
        "embedding_model": settings.embedding_model,
        "num_questions": len(items),
        "num_answerable": sum(1 for i in items if i.answerable),
        "num_negative": sum(1 for i in items if not i.answerable),
        "metrics": agg,
        "per_question": [vars(r) for r in results],
    }
    return report


def _write_report(report: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = report["timestamp"].replace(":", "").replace("-", "")
    out = RESULTS_DIR / f"run_{report['mode']}_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    _append_results_table(report)
    return out


def _append_results_table(report: dict) -> None:
    table = RESULTS_DIR / "RESULTS.md"
    header = (
        "# Retrieval eval results\n\n"
        "Appended by `eval/run.py`. Each row = one retriever config on the frozen\n"
        "golden set. Deltas are read down the table (baseline is the anchor).\n"
        "tokens/q = LLM tokens spent on agentic transforms (0 for non-agentic).\n\n"
        "| label | mode | recall@5 | recall@10 | MRR | nDCG@10 "
        "| retrieve p50/p95 ms | tokens/q | n | provisional |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    if not table.exists():
        table.write_text(header)
    m = report["metrics"]["overall"]
    lat = report["metrics"]["latency_ms"]
    tokens = report["metrics"].get("cost", {}).get("mean_tokens_per_query", 0.0)
    row = (
        f"| {report['label']} | {report['mode']} | {m['recall@5']} | {m['recall@10']} "
        f"| {m['mrr']} | {m['ndcg@10']} | {lat['retrieve_p50']}/{lat['retrieve_p95']} "
        f"| {tokens} | {m['n']} | {'yes' if report['provisional'] else 'no'} |\n"
    )
    with table.open("a") as fh:
        fh.write(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval eval suite.")
    parser.add_argument("--mode", default="dense")
    parser.add_argument("--label", default=None, help="row label (defaults to mode)")
    parser.add_argument(
        "--final", action="store_true", help="mark non-provisional (golden set signed off)"
    )
    args = parser.parse_args()
    report = asyncio.run(run_eval(args.mode, args.label or args.mode, provisional=not args.final))

    out = _write_report(report)
    m = report["metrics"]["overall"]
    print(f"[{report['label']}] mode={report['mode']} n={m['n']}")
    print(
        f"  recall@5={m['recall@5']}  recall@10={m['recall@10']}  "
        f"mrr={m['mrr']}  ndcg@10={m['ndcg@10']}"
    )
    print(
        f"  retrieve p50/p95 = {report['metrics']['latency_ms']['retrieve_p50']}"
        f"/{report['metrics']['latency_ms']['retrieve_p95']} ms"
    )
    cost = report["metrics"].get("cost", {})
    if cost.get("mean_tokens_per_query"):
        print(
            f"  cost: {cost['mean_tokens_per_query']} tokens/query, "
            f"{cost['mean_iterations']} iterations, "
            f"shadow ${cost['shadow_usd_per_1k_queries']}/1k queries"
        )
    print(f"  wrote {out.relative_to(EVAL_DIR.parent)}")
    if report["provisional"]:
        print("  NOTE: provisional — pending golden-set sign-off (rerun with --final)")


if __name__ == "__main__":
    main()
