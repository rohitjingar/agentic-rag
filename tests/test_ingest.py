import psycopg

from rag.ingest.chunker import ChunkConfig
from rag.ingest.embedder import FakeEmbedder
from rag.ingest.pipeline import ingest_corpus

CONFIG = ChunkConfig(size_tokens=64, overlap_tokens=8)


def write_corpus(corpus_dir, doc_a_body="body of alpha. " * 40):
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "alpha.md").write_text(f"# Alpha\n\n{doc_a_body}")
    (corpus_dir / "beta.md").write_text("# Beta\n\n" + "content of beta. " * 40)


def chunk_count(url: str, doc_id: str) -> int:
    with psycopg.connect(url) as conn:
        row = conn.execute(
            "SELECT count(*) FROM chunks WHERE doc_id = %s AND chunk_config_hash = %s",
            (doc_id, CONFIG.config_hash),
        ).fetchone()
    return row[0]


def test_ingest_is_idempotent(migrated_database_url, tmp_path):
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    embedder = FakeEmbedder()

    first = ingest_corpus(migrated_database_url, corpus, embedder, CONFIG)
    assert first.docs_ingested == 2
    assert first.chunks_written > 0
    baseline = chunk_count(migrated_database_url, "alpha")

    second = ingest_corpus(migrated_database_url, corpus, embedder, CONFIG)
    assert second.docs_ingested == 0
    assert second.docs_skipped == 2
    assert second.chunks_written == 0
    assert chunk_count(migrated_database_url, "alpha") == baseline


def test_removed_doc_is_pruned(migrated_database_url, tmp_path):
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    embedder = FakeEmbedder()
    ingest_corpus(migrated_database_url, corpus, embedder, CONFIG)
    assert chunk_count(migrated_database_url, "beta") > 0

    (corpus / "beta.md").unlink()
    stats = ingest_corpus(migrated_database_url, corpus, embedder, CONFIG)
    assert stats.docs_pruned >= 1
    assert chunk_count(migrated_database_url, "beta") == 0


def test_changed_doc_gets_rechunked(migrated_database_url, tmp_path):
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    embedder = FakeEmbedder()
    ingest_corpus(migrated_database_url, corpus, embedder, CONFIG)

    write_corpus(corpus, doc_a_body="totally new alpha text. " * 10)
    stats = ingest_corpus(migrated_database_url, corpus, embedder, CONFIG)
    assert stats.docs_ingested == 1  # alpha changed
    assert stats.docs_skipped == 1  # beta untouched

    with psycopg.connect(migrated_database_url) as conn:
        (text,) = conn.execute(
            "SELECT text FROM chunks WHERE doc_id = 'alpha' AND chunk_config_hash = %s "
            "ORDER BY ord LIMIT 1",
            (CONFIG.config_hash,),
        ).fetchone()
    assert "totally new alpha text" in text
