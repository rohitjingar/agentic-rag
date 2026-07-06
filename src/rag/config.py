"""Application settings. All configuration comes from the environment (.env locally)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5433/rag"
    redis_url: str = "redis://localhost:6380/0"

    ollama_base_url: str = "http://localhost:11434"
    generation_model: str = "llama3.1:8b"
    judge_model: str = "qwen2.5:7b-instruct"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    # "sentence-transformers" (real) or "fake" (deterministic, for tests/CI)
    embedding_backend: str = "sentence-transformers"

    corpus_dir: str = "data/corpus"
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 60
    top_k: int = 5
    # dense | sparse | hybrid (P4) | rerank | rerank-dense (P5)
    retrieval_mode: str = "dense"
    reranker_backend: str = "cross-encoder"  # or "fake" (tests/CI)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_num_ctx: int = 8192

    # semantic cache. 0.90 tuned on paraphrase probes: bge-small close-paraphrase
    # sims cluster ~0.90-0.92; a *different* question scored 0.74, so 0.90 catches
    # near-duplicates while staying above false-hit territory. Precision > recall
    # for a cache — a false hit serves a wrong answer, a miss just recomputes.
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.90
    cache_ttl_seconds: int = 3600

    otel_enabled: bool = False  # export spans to Jaeger when true
    otel_endpoint: str = "http://localhost:4318"

    # CI has no Ollama: with require_llm=false the health probe skips the LLM
    # check instead of failing the whole service.
    require_llm: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
