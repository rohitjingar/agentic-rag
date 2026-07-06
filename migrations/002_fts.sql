-- Phase 4: BM25-style sparse retrieval via PostgreSQL full-text search.
-- Decision (b): PG FTS over rank-bm25/pg_search — one engine, persistent, no
-- new deps. ts_rank_cd is not textbook BM25, but RRF fusion consumes rank
-- ORDER, not raw scores, so exact score calibration barely matters.
--
-- A GENERATED column keeps the tsvector in lockstep with text automatically
-- (no trigger, no backfill drift) — the FTS analogue of a maintained index.

ALTER TABLE chunks
    ADD COLUMN text_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;

CREATE INDEX chunks_text_tsv_gin ON chunks USING gin (text_tsv);
