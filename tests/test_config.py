import os

from rag.config import Settings


def test_defaults_are_sane(monkeypatch):
    # clear RAG_* env so we test true field defaults, not the CI/.env environment
    # (CI sets RAG_REQUIRE_LLM=false, which pydantic-settings reads regardless
    # of _env_file)
    for key in list(os.environ):
        if key.startswith("RAG_"):
            monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.embedding_dim == 384
    assert settings.require_llm is True
    assert settings.generation_model != settings.judge_model  # cross-family judging


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("RAG_GENERATION_MODEL", "some-model:1b")
    monkeypatch.setenv("RAG_REQUIRE_LLM", "false")
    settings = Settings(_env_file=None)
    assert settings.generation_model == "some-model:1b"
    assert settings.require_llm is False
