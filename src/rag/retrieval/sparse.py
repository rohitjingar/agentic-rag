"""Sparse (BM25-style) retrieval via PostgreSQL full-text search.

The query is lexemized by Postgres (same `english` config as the indexed
tsvector, so stemming/stopwords match) and the lexemes are OR-ed together. AND
semantics (websearch/plainto default) would demand every term appear in one
chunk — fatal for long natural-language questions. OR-ing lets ts_rank_cd rank
by how many query terms a chunk covers, which is what rescues exact-identifier
queries ("tools/list", "ef_search") that dense retrieval blurs.
"""

from __future__ import annotations

from psycopg_pool import AsyncConnectionPool

from rag.models import RetrievedChunk


async def sparse_search(
    pool: AsyncConnectionPool,
    query: str,
    chunk_config_hash: str,
    k: int,
) -> list[RetrievedChunk]:
    async with pool.connection() as conn:
        # lexemize the question with the same config as the index, then OR terms
        cur = await conn.execute("SELECT tsvector_to_array(to_tsvector('english', %s))", (query,))
        (lexemes,) = await cur.fetchone()
        if not lexemes:
            return []
        tsquery = " | ".join(lexemes)

        cur = await conn.execute(
            """
            SELECT id, doc_id, ord, text,
                   ts_rank_cd(text_tsv, query, 32) AS score
            FROM chunks, to_tsquery('english', %(q)s) AS query
            WHERE chunk_config_hash = %(cfg)s AND text_tsv @@ query
            ORDER BY score DESC
            LIMIT %(k)s
            """,
            {"q": tsquery, "cfg": chunk_config_hash, "k": k},
        )
        rows = await cur.fetchall()
    return [RetrievedChunk(id=r[0], doc_id=r[1], ord=r[2], text=r[3], score=r[4]) for r in rows]
