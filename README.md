# agentic-rag

Agentic RAG with an **eval-first** harness: the golden dataset and measurement
pipeline are built *before* any retrieval improvement, and every upgrade
(hybrid retrieval → cross-encoder reranking → agentic re-query loop) has to
earn its place with a measured delta on the golden set.

**Status: under construction.** The results table (baseline → hybrid → rerank
→ agentic) lands here as phases complete.

## Stack

FastAPI · PostgreSQL + pgvector (HNSW) · BM25 · sentence-transformers
(`bge-small-en-v1.5`) · cross-encoder reranker · Ollama (`llama3.1:8b`
generation, `qwen2.5:7b-instruct` judge — deliberately different families) ·
Redis 8 semantic cache · OpenTelemetry + Jaeger · Docker Compose · uv

Total spend: **$0** — every model runs locally, CI runs on the free tier.

## Quickstart

```bash
make up       # postgres+pgvector, redis, jaeger (docker compose)
make migrate  # apply versioned SQL migrations
make models   # pull local ollama models (~10 GB, one-time)
make test
make serve    # http://localhost:8000/healthz
```

Configuration is environment-only: copy `.env.example` to `.env` and adjust.
