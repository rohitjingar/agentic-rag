import psycopg
import pytest

from eval.materialize import MaterializationError, materialize_all, ngram_overlap
from eval.schema import GoldenItem

CFG = "materialize-test"


def seed_doc(url: str, chunk_id: str, text: str) -> None:
    with psycopg.connect(url) as conn:
        conn.execute(
            """
            INSERT INTO documents (id, source_path, title, meta, content_sha256, corpus_version)
            VALUES ('mdoc', 'mdoc.md', 'M', '{}', 'x', 'test')
            ON CONFLICT (id) DO NOTHING
            """
        )
        # replace any prior chunk for this doc+config so tests don't pollute each other
        conn.execute("DELETE FROM chunks WHERE doc_id = 'mdoc' AND chunk_config_hash = %s", (CFG,))
        conn.execute(
            """
            INSERT INTO chunks (id, doc_id, ord, text, token_count, char_start, char_end,
                                chunk_config_hash, embedding)
            VALUES (%s, 'mdoc', 0, %s, 5, 0, 100, %s, NULL)
            """,
            (chunk_id, text, CFG),
        )


def item(locator: str, question: str = "How do I reclaim disk space after deletes?") -> GoldenItem:
    return GoldenItem(
        qid="q1",
        question=question,
        expected_answer="run vacuum",
        qtype="how-to",
        gold=[{"doc_id": "mdoc", "locator": locator, "grade": "primary"}],
    )


def test_materialize_resolves_locator_to_chunk(migrated_database_url):
    seed_doc(migrated_database_url, "mdoc#0", "Run VACUUM to reclaim dead row space.")
    with psycopg.connect(migrated_database_url) as conn:
        [m] = materialize_all(conn, [item("reclaim dead row space")], CFG)
    assert m.primary_chunk_ids == {"mdoc#0"}


def test_missing_locator_fails_loudly(migrated_database_url):
    seed_doc(migrated_database_url, "mdoc#0", "Run VACUUM to reclaim dead row space.")
    with psycopg.connect(migrated_database_url) as conn:  # noqa: SIM117
        with pytest.raises(MaterializationError, match="locator not found"):
            materialize_all(conn, [item("this phrase is absent")], CFG)


def test_negative_control_materializes_empty(migrated_database_url):
    neg = GoldenItem(
        qid="n1",
        question="How do I shard MongoDB?",
        expected_answer="n/a",
        qtype="negative-control",
        answerable=False,
    )
    with psycopg.connect(migrated_database_url) as conn:
        [m] = materialize_all(conn, [neg], CFG)
    assert m.grade_by_chunk == {}


def test_leakage_is_detected(migrated_database_url):
    text = "Create a partial index by adding a WHERE clause to CREATE INDEX statement today."
    seed_doc(migrated_database_url, "mdoc#0", text)
    # question copies the chunk almost verbatim -> should trip the leakage guard
    leaky = item(locator="WHERE clause to CREATE INDEX", question=text)
    with psycopg.connect(migrated_database_url) as conn:  # noqa: SIM117
        with pytest.raises(MaterializationError, match="leakage"):
            materialize_all(conn, [leaky], CFG, leakage_threshold=0.5)


def test_ngram_overlap_math():
    assert ngram_overlap("a b c d e f", "a b c d e f", n=5) == 1.0
    assert (
        ngram_overlap("totally different words here now", "nothing shared at all here", n=5) == 0.0
    )
