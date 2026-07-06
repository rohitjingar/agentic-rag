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

    # CI has no Ollama: with require_llm=false the health probe skips the LLM
    # check instead of failing the whole service.
    require_llm: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
