"""FastAPI application: fail-closed lifespan wiring + health probe."""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from psycopg_pool import AsyncConnectionPool

from rag.api.routes import router
from rag.config import Settings, get_settings
from rag.db import discover_migrations, run_migrations
from rag.generation.client import OllamaClient
from rag.ingest.chunker import ChunkConfig
from rag.ingest.embedder import Embedder, build_embedder
from rag.query import RAGService
from rag.retrieval.factory import build_retriever


def chunk_config_from(settings: Settings) -> ChunkConfig:
    return ChunkConfig(
        size_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        tokenizer_model=settings.embedding_model,
    )


def create_app(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    llm: OllamaClient | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail closed: migrations applied + a live DB pool before serving anything.
        run_migrations(settings.database_url)
        pool = AsyncConnectionPool(settings.database_url, min_size=1, max_size=10, open=False)
        await pool.open()
        await pool.wait(timeout=10)
        redis = aioredis.from_url(settings.redis_url)

        app.state.settings = settings
        app.state.pool = pool
        app.state.redis = redis
        app.state.llm = llm or OllamaClient(
            settings.ollama_base_url, settings.generation_model, num_ctx=settings.llm_num_ctx
        )
        app.state.embedder = embedder or build_embedder(
            settings.embedding_backend, settings.embedding_model, settings.embedding_dim
        )
        retriever = build_retriever(
            settings.retrieval_mode,
            pool,
            app.state.embedder,
            chunk_config_from(settings).config_hash,
            reranker_backend=settings.reranker_backend,
            reranker_model=settings.reranker_model,
        )
        app.state.rag = RAGService(retriever=retriever, llm=app.state.llm, top_k=settings.top_k)
        try:
            yield
        finally:
            await app.state.llm.aclose()
            await redis.aclose()
            await pool.close()

    app = FastAPI(title="agentic-rag", lifespan=lifespan)
    app.include_router(router)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        checks: dict[str, str] = {}

        try:
            async with app.state.pool.connection() as conn:
                cur = await conn.execute("SELECT version FROM schema_migrations")
                applied = {version for (version,) in await cur.fetchall()}
            missing = [m.version for m in discover_migrations() if m.version not in applied]
            checks["postgres"] = "ok"
            checks["migrations"] = "ok" if not missing else f"pending: {missing}"
        except Exception as exc:  # a probe reports failures, it doesn't crash
            checks["postgres"] = f"error: {exc}"
            checks["migrations"] = "unknown"

        try:
            await app.state.redis.ping()
            checks["redis"] = "ok"
        except Exception as exc:
            checks["redis"] = f"error: {exc}"

        if settings.require_llm:
            checks["ollama"] = await _check_ollama(settings)
        else:
            checks["ollama"] = "skipped (require_llm=false)"

        ok = all(v == "ok" or v.startswith("skipped") for v in checks.values())
        return JSONResponse(
            status_code=200 if ok else 503,
            content={"status": "ok" if ok else "unhealthy", "checks": checks},
        )

    return app


async def _check_ollama(settings: Settings) -> str:
    try:
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=3) as client:
            resp = await client.get("/api/tags")
            resp.raise_for_status()
        available = {m["name"] for m in resp.json().get("models", [])}
    except Exception as exc:
        return f"error: {exc}"
    needed = {settings.generation_model, settings.judge_model}
    missing = sorted(needed - available)
    return "ok" if not missing else f"missing models: {missing}"


app = create_app()
