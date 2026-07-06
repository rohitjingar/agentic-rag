"""CLI entry points: rag-ingest and rag-query."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rag.config import get_settings


def ingest_main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the corpus into Postgres.")
    parser.add_argument("--corpus-dir", type=Path, default=None)
    args = parser.parse_args()

    from rag.api.app import chunk_config_from
    from rag.db import run_migrations
    from rag.ingest.embedder import build_embedder
    from rag.ingest.pipeline import ingest_corpus

    settings = get_settings()
    corpus_dir = args.corpus_dir or Path(settings.corpus_dir)
    if not corpus_dir.exists():
        sys.exit(f"corpus dir not found: {corpus_dir} — run scripts/fetch_corpus.py first")

    run_migrations(settings.database_url)
    embedder = build_embedder(
        settings.embedding_backend, settings.embedding_model, settings.embedding_dim
    )
    stats = ingest_corpus(settings.database_url, corpus_dir, embedder, chunk_config_from(settings))
    print(
        f"ingested={stats.docs_ingested} skipped={stats.docs_skipped} "
        f"chunks_written={stats.chunks_written}"
    )


def query_main() -> None:
    parser = argparse.ArgumentParser(description="Ask the RAG pipeline a question.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(_query(args.question, args.top_k))


async def _query(question: str, top_k: int | None) -> None:
    from psycopg_pool import AsyncConnectionPool

    from rag.api.app import chunk_config_from
    from rag.generation.client import OllamaClient
    from rag.ingest.embedder import build_embedder
    from rag.query import RAGService

    settings = get_settings()
    pool = AsyncConnectionPool(settings.database_url, min_size=1, max_size=2, open=False)
    await pool.open()
    llm = OllamaClient(
        settings.ollama_base_url, settings.generation_model, num_ctx=settings.llm_num_ctx
    )
    embedder = build_embedder(
        settings.embedding_backend, settings.embedding_model, settings.embedding_dim
    )
    service = RAGService(pool, embedder, llm, chunk_config_from(settings), settings.top_k)
    try:
        result = await service.answer(question, top_k=top_k)
    finally:
        await llm.aclose()
        await pool.close()

    print(result.answer)
    print("\n--- sources ---")
    for i, chunk in enumerate(result.sources, start=1):
        print(f"[S{i}] {chunk.doc_id} (score={chunk.score:.3f})")
    timing = result.timings
    print(
        f"\ntokens_in={result.tokens_in} tokens_out={result.tokens_out} | "
        f"embed={timing.embed_ms:.0f}ms retrieve={timing.retrieve_ms:.0f}ms "
        f"generate={timing.generate_ms:.0f}ms"
    )
