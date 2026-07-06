"""Dense retrieval: cosine top-k over pgvector's HNSW index."""

from __future__ import annotations

from psycopg_pool import AsyncConnectionPool

from rag.ingest.pipeline import to_pgvector
from rag.models import RetrievedChunk


async def dense_search(
    pool: AsyncConnectionPool,
    query_vector: list[float],
    chunk_config_hash: str,
    k: int,
) -> list[RetrievedChunk]:
    qv = to_pgvector(query_vector)
    async with pool.connection() as conn:
        # HNSW's default ef_search is 40; keep candidate breadth ahead of k
        # so deep top-k requests (reranking pulls 50) don't silently degrade.
        await conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(40, 2 * k)),))
        cur = await conn.execute(
            """
            SELECT id, doc_id, ord, text, 1 - (embedding <=> %(qv)s::vector) AS score
            FROM chunks
            WHERE chunk_config_hash = %(cfg)s
            ORDER BY embedding <=> %(qv)s::vector
            LIMIT %(k)s
            """,
            {"qv": qv, "cfg": chunk_config_hash, "k": k},
        )
        rows = await cur.fetchall()
    return [RetrievedChunk(id=r[0], doc_id=r[1], ord=r[2], text=r[3], score=r[4]) for r in rows]
