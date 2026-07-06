import asyncio

import psycopg
from psycopg_pool import AsyncConnectionPool

from rag.ingest.pipeline import to_pgvector
from rag.retrieval.dense import dense_search

DIM = 384
CFG = "retrieval-test"


def vec(index: int, value: float = 1.0, index2: int | None = None, value2: float = 0.0):
    v = [0.0] * DIM
    v[index] = value
    if index2 is not None:
        v[index2] = value2
    return v


def seed(url: str) -> None:
    with psycopg.connect(url) as conn:
        conn.execute(
            """
            INSERT INTO documents (id, source_path, title, meta, content_sha256, corpus_version)
            VALUES ('rdoc', 'rdoc.md', 'R Doc', '{}', 'x', 'test')
            ON CONFLICT (id) DO NOTHING
            """
        )
        rows = [
            ("rdoc#c0", 0, "exact match", vec(0)),  # cos = 1.0 vs query
            ("rdoc#c1", 1, "close match", vec(0, 0.6, 1, 0.8)),  # cos = 0.6
            ("rdoc#c2", 2, "orthogonal", vec(2)),  # cos = 0.0
        ]
        for chunk_id, ord_, text, v in rows:
            conn.execute(
                """
                INSERT INTO chunks (id, doc_id, ord, text, token_count, char_start, char_end,
                                    chunk_config_hash, embedding)
                VALUES (%s, 'rdoc', %s, %s, 3, 0, 10, %s, %s::vector)
                ON CONFLICT (id) DO NOTHING
                """,
                (chunk_id, ord_, text, CFG, to_pgvector(v)),
            )


async def run_search(url: str, k: int, config_hash: str = CFG):
    pool = AsyncConnectionPool(url, min_size=1, max_size=2, open=False)
    await pool.open()
    try:
        return await dense_search(pool, vec(0), config_hash, k)
    finally:
        await pool.close()


def test_dense_search_orders_by_cosine(migrated_database_url):
    seed(migrated_database_url)
    results = asyncio.run(run_search(migrated_database_url, 3))
    assert [r.id for r in results] == ["rdoc#c0", "rdoc#c1", "rdoc#c2"]
    assert results[0].score > 0.99
    assert abs(results[1].score - 0.6) < 0.01
    assert abs(results[2].score) < 0.01


def test_dense_search_respects_k(migrated_database_url):
    seed(migrated_database_url)
    results = asyncio.run(run_search(migrated_database_url, 2))
    assert len(results) == 2


def test_dense_search_scoped_to_chunk_config(migrated_database_url):
    seed(migrated_database_url)
    assert asyncio.run(run_search(migrated_database_url, 3, "no-such-config")) == []
