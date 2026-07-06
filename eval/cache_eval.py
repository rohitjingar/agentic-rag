"""Semantic cache eval: warm the cache with a set of questions, then replay a
workload of close paraphrases + repeats + novel queries and measure hit-rate,
tokens saved, and latency saved.

    uv run python -m eval.cache_eval

The workload is derived from (but not identical to) the golden set — close
paraphrases model the real near-duplicate traffic a cache exists to absorb.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import redis.asyncio as aioredis
from psycopg_pool import AsyncConnectionPool

from eval.run import RESULTS_DIR
from rag.api.app import cache_namespace, chunk_config_from
from rag.cache.semantic import SemanticCache
from rag.config import get_settings
from rag.generation.client import OllamaClient
from rag.ingest.embedder import build_embedder
from rag.query import RAGService
from rag.retrieval.factory import build_retriever

# (seed question, [close paraphrases]). Seeds warm the cache; paraphrases +
# repeats + novel queries are the replay workload.
WORKLOAD = [
    (
        "What is the default value of hnsw.ef_search in pgvector?",
        ["In pgvector, what's the default hnsw.ef_search value?"],
    ),
    (
        "How do I write tests for a FastAPI application?",
        ["How to test a FastAPI app?", "What's the way to test FastAPI apps?"],
    ),
    (
        "Which index type does PostgreSQL create by default?",
        ["What is Postgres' default index type?"],
    ),
    (
        "What two standard transport mechanisms does MCP define?",
        ["Which transports does the Model Context Protocol define?"],
    ),
    (
        "How do I reclaim disk space after deleting rows in PostgreSQL?",
        ["How do I run VACUUM to free space in Postgres?"],
    ),
    (
        "How do I return a custom HTTP status code from a FastAPI path operation?",
        ["How to set a specific status code on a FastAPI route?"],
    ),
    (
        "When should I pick HNSW over IVFFlat in pgvector?",
        ["pgvector: HNSW vs IVFFlat, which to choose?"],
    ),
    (
        "How does an MCP client discover available tools?",
        ["Which MCP method lists a server's tools?"],
    ),
]
# genuinely novel queries that SHOULD miss (guard against false hits)
NOVEL = [
    "How do I create a partial index in PostgreSQL?",
    "What is write-ahead logging for?",
]


async def main() -> None:
    settings = get_settings()
    pool = AsyncConnectionPool(settings.database_url, min_size=1, max_size=4, open=False)
    await pool.open()
    redis = aioredis.from_url(settings.redis_url)
    embedder = build_embedder(
        settings.embedding_backend, settings.embedding_model, settings.embedding_dim
    )
    llm = OllamaClient(
        settings.ollama_base_url, settings.generation_model, num_ctx=settings.llm_num_ctx
    )
    cache = SemanticCache(
        redis,
        embedder,
        cache_namespace(settings),
        threshold=settings.cache_similarity_threshold,
        dim=settings.embedding_dim,
    )
    # isolate this run's namespace
    await redis.flushdb()
    await cache.ensure_index()
    retriever = build_retriever("rerank", pool, embedder, chunk_config_from(settings).config_hash)
    service = RAGService(retriever, llm, settings.top_k, cache=cache)

    # warm: seed questions (all miss, populate cache)
    print("warming cache...")
    for seed, _ in WORKLOAD:
        await service.answer(seed)

    # replay: paraphrases (expect hit) + repeats of seeds (expect hit) + novel (expect miss)
    replay = []
    for seed, paraphrases in WORKLOAD:
        replay.append(("repeat", seed))
        for p in paraphrases:
            replay.append(("paraphrase", p))
    for q in NOVEL:
        replay.append(("novel", q))

    print(f"replaying {len(replay)} queries...")
    rows = []
    for kind, q in replay:
        r = await service.answer(q)
        rows.append({"kind": kind, "query": q, "cached": r.cached, "shadow_usd": r.shadow_usd})

    await llm.aclose()
    await redis.aclose()
    await pool.close()

    # aggregate
    paraphrase = [r for r in rows if r["kind"] == "paraphrase"]
    repeats = [r for r in rows if r["kind"] == "repeat"]
    novel = [r for r in rows if r["kind"] == "novel"]
    hits = [r for r in rows if r["cached"]]

    def rate(rs):
        return round(sum(r["cached"] for r in rs) / len(rs), 3) if rs else 0.0

    # shadow savings: hits cost $0 vs the mean uncached cost
    uncached = [r for r in rows if not r["cached"]]
    mean_uncached_usd = (
        round(sum(r["shadow_usd"] for r in uncached) / len(uncached), 6) if uncached else 0.0
    )
    report = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "threshold": settings.cache_similarity_threshold,
        "workload_size": len(rows),
        "overall_hit_rate": rate(rows),
        "paraphrase_hit_rate": rate(paraphrase),
        "repeat_hit_rate": rate(repeats),
        "novel_false_hit_rate": rate(novel),  # must be 0.0 (no wrong answers)
        "hits": len(hits),
        "mean_uncached_shadow_usd": mean_uncached_usd,
        "shadow_usd_saved_per_hit": mean_uncached_usd,
        "rows": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = report["timestamp"].replace(":", "").replace("-", "")
    (RESULTS_DIR / f"cache_{stamp}.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\noverall hit-rate      : {report['overall_hit_rate']}")
    print(f"paraphrase hit-rate   : {report['paraphrase_hit_rate']}")
    print(f"repeat hit-rate       : {report['repeat_hit_rate']}")
    print(f"novel false-hit-rate  : {report['novel_false_hit_rate']} (must be 0.0)")
    print(f"mean uncached cost    : ${mean_uncached_usd} shadow / query")
    print(f"saved per hit         : ${mean_uncached_usd} shadow (+ retrieve+generate latency)")


if __name__ == "__main__":
    asyncio.run(main())
