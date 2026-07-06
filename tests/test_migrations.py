import pytest

from rag.db import discover_migrations, run_migrations


def test_migrations_apply_exactly_once(fresh_database_url):
    first = run_migrations(fresh_database_url)
    assert first == [m.version for m in discover_migrations()]
    second = run_migrations(fresh_database_url)
    assert second == []


def test_checksum_drift_fails_closed(fresh_database_url, tmp_path):
    migration = tmp_path / "900_drift_probe.sql"
    migration.write_text("CREATE TABLE drift_probe (id INT);")
    assert run_migrations(fresh_database_url, tmp_path) == ["900_drift_probe"]

    migration.write_text("CREATE TABLE drift_probe (id BIGINT);")
    with pytest.raises(RuntimeError, match="checksum drift"):
        run_migrations(fresh_database_url, tmp_path)


def test_schema_has_core_tables(migrated_database_url):
    import psycopg

    with psycopg.connect(migrated_database_url) as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
    tables = {name for (name,) in rows}
    assert {"documents", "chunks", "schema_migrations"} <= tables
