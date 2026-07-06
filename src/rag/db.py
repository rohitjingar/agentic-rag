"""Versioned, exactly-once SQL migrations.

An advisory lock serializes concurrent migrators (safe with multiple app
instances starting at once), schema_migrations records what ran with a
checksum, and an already-applied file that was edited afterwards fails
closed on checksum drift instead of silently diverging.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg

from rag.config import get_settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
_ADVISORY_LOCK_ID = 0x5241_4731  # "RAG1"


@dataclass(frozen=True)
class Migration:
    version: str
    path: Path
    checksum: str


def discover_migrations(migrations_dir: Path = MIGRATIONS_DIR) -> list[Migration]:
    return [
        Migration(
            version=path.stem,
            path=path,
            checksum=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(migrations_dir.glob("*.sql"))
    ]


def run_migrations(database_url: str, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending migrations in version order. Returns the versions applied now."""
    applied_now: list[str] = []
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                checksum TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute("SELECT pg_advisory_lock(%s)", (_ADVISORY_LOCK_ID,))
        try:
            rows = conn.execute("SELECT version, checksum FROM schema_migrations").fetchall()
            applied = dict(rows)
            for migration in discover_migrations(migrations_dir):
                if migration.version in applied:
                    if applied[migration.version] != migration.checksum:
                        raise RuntimeError(
                            f"migration {migration.version} changed after being applied "
                            "(checksum drift) — refusing to continue"
                        )
                    continue
                with psycopg.connect(database_url) as tx:
                    tx.execute(migration.path.read_text())
                    tx.execute(
                        "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)",
                        (migration.version, migration.checksum),
                    )
                applied_now.append(migration.version)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_ADVISORY_LOCK_ID,))
    return applied_now


def main() -> None:
    applied = run_migrations(get_settings().database_url)
    print(f"applied: {applied or 'nothing (up to date)'}", file=sys.stderr)
