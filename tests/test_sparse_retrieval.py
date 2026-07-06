"""Sparse (FTS) retrieval — the case dense retrieval blurs: exact identifiers.

Uses the generated text_tsv column (migration 002), applied to the test DB by
the conftest migration fixture.
"""

import asyncio

import psycopg
from psycopg_pool import AsyncConnectionPool

from rag.retrieval.sparse import sparse_search

CFG = "sparse-test"


def seed(url: str) -> None:
    rows = [
        ("s#0", 0, "The hnsw.ef_search parameter is 40 by default in pgvector."),
        ("s#1", 1, "Clients send a tools/list request to discover available tools."),
        ("s#2", 2, "FastAPI builds interactive documentation from Python type hints."),
        ("s#3", 3, "A partial index covers only a subset of table rows via a predicate."),
    ]
    with psycopg.connect(url) as conn:
        conn.execute(
            """
            INSERT INTO documents (id, source_path, title, meta, content_sha256, corpus_version)
            VALUES ('sdoc', 'sdoc.md', 'S', '{}', 'x', 'test')
            ON CONFLICT (id) DO NOTHING
            """
        )
        conn.execute("DELETE FROM chunks WHERE doc_id='sdoc' AND chunk_config_hash=%s", (CFG,))
        for cid, ord_, text in rows:
            conn.execute(
                """
                INSERT INTO chunks (id, doc_id, ord, text, token_count, char_start, char_end,
                                    chunk_config_hash, embedding)
                VALUES (%s, 'sdoc', %s, %s, 10, 0, 100, %s, NULL)
                """,
                (cid, ord_, text, CFG),
            )


async def run(url: str, query: str, k: int = 5):
    pool = AsyncConnectionPool(url, min_size=1, max_size=2, open=False)
    await pool.open()
    try:
        return await sparse_search(pool, query, CFG, k)
    finally:
        await pool.close()


def test_sparse_nails_exact_identifier(migrated_database_url):
    seed(migrated_database_url)
    # dense blurs 'ef_search'; sparse matches the literal token
    results = asyncio.run(run(migrated_database_url, "what is the default hnsw.ef_search value?"))
    assert results, "expected a lexical match"
    assert results[0].id == "s#0"


def test_sparse_matches_slash_identifier(migrated_database_url):
    seed(migrated_database_url)
    results = asyncio.run(run(migrated_database_url, "which request lists tools? tools/list"))
    assert results[0].id == "s#1"


def test_sparse_returns_empty_for_no_lexemes(migrated_database_url):
    seed(migrated_database_url)
    # a query of pure stopwords lexemizes to nothing -> no crash, empty result
    assert asyncio.run(run(migrated_database_url, "the of and to")) == []


def test_sparse_scoped_to_config(migrated_database_url):
    seed(migrated_database_url)
    assert asyncio.run(run(migrated_database_url, "ef_search", k=5)) != []
    pool_cfg_miss = asyncio.run(_run_other_cfg(migrated_database_url))
    assert pool_cfg_miss == []


async def _run_other_cfg(url: str):
    pool = AsyncConnectionPool(url, min_size=1, max_size=2, open=False)
    await pool.open()
    try:
        return await sparse_search(pool, "ef_search", "no-such-cfg", 5)
    finally:
        await pool.close()
