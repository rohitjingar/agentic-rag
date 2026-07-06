CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id             TEXT PRIMARY KEY,
    source_path    TEXT NOT NULL UNIQUE,
    title          TEXT NOT NULL,
    meta           JSONB NOT NULL DEFAULT '{}',
    content_sha256 TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id                TEXT PRIMARY KEY,
    doc_id            TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord               INT NOT NULL,
    text              TEXT NOT NULL,
    token_count       INT NOT NULL,
    -- char offsets into the source document: golden-set labels anchor to
    -- (doc_id, span) and get materialized to chunk ids per chunking config
    char_start        INT NOT NULL,
    char_end          INT NOT NULL,
    chunk_config_hash TEXT NOT NULL,
    embedding         vector(384),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, chunk_config_hash, ord)
);

CREATE INDEX chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
