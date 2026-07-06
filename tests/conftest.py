"""Shared fixtures. Tests run against real Postgres/Redis (compose locally,
service containers in CI), same pattern as agent-gateway."""

import os

import psycopg
import pytest

from rag.config import Settings
from rag.db import run_migrations

ADMIN_URL = os.environ.get("RAG_DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")
REDIS_URL = os.environ.get("RAG_REDIS_URL", "redis://localhost:6380/0")


def _recreate_database(name: str) -> str:
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{name}"')
    base, _, _ = ADMIN_URL.rpartition("/")
    return f"{base}/{name}"


@pytest.fixture(scope="session")
def migrated_database_url() -> str:
    """A fresh, fully migrated database shared by the session."""
    url = _recreate_database("rag_test")
    run_migrations(url)
    return url


@pytest.fixture
def fresh_database_url() -> str:
    """A fresh database with NO migrations applied (for migration-runner tests)."""
    return _recreate_database("rag_test_fresh")


@pytest.fixture
def settings(migrated_database_url: str) -> Settings:
    return Settings(
        _env_file=None,
        database_url=migrated_database_url,
        redis_url=REDIS_URL,
        require_llm=False,
        embedding_backend="fake",  # never download models inside tests
        cache_enabled=False,  # cache is exercised by dedicated tests, not everywhere
    )
