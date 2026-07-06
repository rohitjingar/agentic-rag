"""Idempotent ingestion: corpus files -> documents + embedded chunks in Postgres.

Re-running is safe: unchanged documents (same content hash, chunks already
present for the live chunk config) are skipped; changed ones have their
chunks for this config replaced atomically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import psycopg

from rag.ingest.chunker import ChunkConfig, chunk_document
from rag.ingest.embedder import Embedder
from rag.ingest.loader import load_corpus
from rag.models import Document


@dataclass
class IngestStats:
    docs_ingested: int = 0
    docs_skipped: int = 0
    docs_pruned: int = 0
    chunks_written: int = 0


def to_pgvector(vector: list[float]) -> str:
    # pgvector accepts its text representation; casting '%s::vector' keeps us
    # free of client-side adapter registration on every connection.
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


def corpus_version(corpus_dir: Path) -> str:
    manifest = corpus_dir / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())["corpus_version"]
    return "dev"


def ingest_corpus(
    database_url: str,
    corpus_dir: Path,
    embedder: Embedder,
    config: ChunkConfig | None = None,
    batch_size: int = 64,
) -> IngestStats:
    config = config or ChunkConfig()
    version = corpus_version(corpus_dir)
    stats = IngestStats()

    docs = load_corpus(corpus_dir)
    if not docs:
        # an empty corpus would prune the whole index — refuse loudly instead
        raise ValueError(f"no documents found in {corpus_dir}")
    with psycopg.connect(database_url) as conn:
        for doc in docs:
            if _is_current(conn, doc, config):
                stats.docs_skipped += 1
                continue
            stats.docs_ingested += 1
            stats.chunks_written += _ingest_document(
                conn, doc, config, embedder, version, batch_size
            )
            conn.commit()
        # documents removed from the corpus must not linger in the index
        cur = conn.execute(
            "DELETE FROM documents WHERE NOT (id = ANY(%s))", ([d.id for d in docs],)
        )
        stats.docs_pruned = cur.rowcount
        conn.commit()
    return stats


def _is_current(conn: psycopg.Connection, doc: Document, config: ChunkConfig) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM documents d
        WHERE d.id = %s AND d.content_sha256 = %s
          AND EXISTS (
            SELECT 1 FROM chunks c
            WHERE c.doc_id = d.id AND c.chunk_config_hash = %s
          )
        """,
        (doc.id, doc.content_sha256, config.config_hash),
    ).fetchone()
    return row is not None


def _ingest_document(
    conn: psycopg.Connection,
    doc: Document,
    config: ChunkConfig,
    embedder: Embedder,
    version: str,
    batch_size: int,
) -> int:
    conn.execute(
        """
        INSERT INTO documents (id, source_path, title, meta, content_sha256, corpus_version)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            meta = EXCLUDED.meta,
            content_sha256 = EXCLUDED.content_sha256,
            corpus_version = EXCLUDED.corpus_version
        """,
        (doc.id, doc.source_path, doc.title, json.dumps(doc.meta), doc.content_sha256, version),
    )
    # content changed (or first ingest for this config): replace this config's chunks
    conn.execute(
        "DELETE FROM chunks WHERE doc_id = %s AND chunk_config_hash = %s",
        (doc.id, config.config_hash),
    )

    chunks = chunk_document(doc, config)
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        vectors = embedder.encode_passages([c.text for c in batch])
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks
                    (id, doc_id, ord, text, token_count, char_start, char_end,
                     chunk_config_hash, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                """,
                [
                    (
                        c.id,
                        c.doc_id,
                        c.ord,
                        c.text,
                        c.token_count,
                        c.char_start,
                        c.char_end,
                        config.config_hash,
                        to_pgvector(v),
                    )
                    for c, v in zip(batch, vectors, strict=True)
                ],
            )
    return len(chunks)
