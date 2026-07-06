"""Semantic cache over Redis 8 vector search.

A plain cache keys on the exact query string, so "how do I test FastAPI?" and
"how to write FastAPI tests?" miss each other. A semantic cache keys on the
query *embedding*: a near-duplicate question (cosine sim >= threshold) hits and
skips retrieval + generation entirely — the expensive stages.

Honest invalidation: entries are namespaced by corpus_version + pipeline-config
hash, so re-ingesting the corpus or changing the pipeline can never serve a
stale answer — those queries simply miss under the new namespace. Plus a TTL.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

import numpy as np

from rag.ingest.embedder import Embedder
from rag.models import RetrievedChunk

_PREFIX = "sc:"


def _sanitize(ns: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in ns)


@dataclass
class CacheHit:
    answer: str
    sources: list[RetrievedChunk]
    tokens_in: int
    tokens_out: int
    similarity: float
    cached_query: str


class SemanticCache:
    def __init__(
        self,
        redis,
        embedder: Embedder,
        namespace: str,
        *,
        threshold: float = 0.90,
        ttl_seconds: int = 3600,
        dim: int = 384,
    ):
        self._redis = redis
        self._embedder = embedder
        self._ns = _sanitize(namespace)
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._dim = dim
        # dim in the name so a dim-4 test index never collides with dim-384 prod
        self._index = f"idx:semantic_cache_d{dim}"

    async def ensure_index(self) -> None:
        try:
            await self._redis.execute_command(
                "FT.CREATE",
                self._index,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                _PREFIX,
                "SCHEMA",
                "ns",
                "TAG",
                "query",
                "TEXT",
                "embedding",
                "VECTOR",
                "FLAT",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                self._dim,
                "DISTANCE_METRIC",
                "COSINE",
            )
        except Exception as exc:  # "Index already exists" is fine
            if "already exists" not in str(exc).lower():
                raise

    def _to_bytes(self, vector: list[float]) -> bytes:
        return np.asarray(vector, dtype=np.float32).tobytes()

    async def get(self, query: str) -> CacheHit | None:
        vec = self._to_bytes(self._embedder.encode_query(query))
        q = f"(@ns:{{{self._ns}}})=>[KNN 1 @embedding $vec AS dist]"
        try:
            raw = await self._redis.execute_command(
                "FT.SEARCH",
                self._index,
                q,
                "PARAMS",
                "2",
                "vec",
                vec,
                "RETURN",
                "5",
                "dist",
                "answer",
                "sources",
                "tokens_in",
                "tokens_out",
                "SORTBY",
                "dist",
                "DIALECT",
                "2",
                "LIMIT",
                "0",
                "1",
            )
        except Exception:
            return None
        fields = _first_doc_fields(raw)
        if fields is None:
            return None
        distance = float(fields.get("dist", 1.0))
        similarity = 1.0 - distance
        if similarity < self._threshold:
            return None
        return CacheHit(
            answer=fields["answer"],
            sources=[RetrievedChunk(**c) for c in json.loads(fields.get("sources", "[]"))],
            tokens_in=int(fields.get("tokens_in", 0)),
            tokens_out=int(fields.get("tokens_out", 0)),
            similarity=similarity,
            cached_query=fields.get("query", ""),
        )

    async def put(
        self,
        query: str,
        answer: str,
        sources: list[RetrievedChunk],
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        vec = self._to_bytes(self._embedder.encode_query(query))
        key = f"{_PREFIX}{self._ns}:{uuid.uuid4().hex}"
        await self._redis.hset(
            key,
            mapping={
                "ns": self._ns,
                "query": query,
                "answer": answer,
                "sources": json.dumps([c.model_dump() for c in sources]),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "embedding": vec,
            },
        )
        await self._redis.expire(key, self._ttl)


def _decode(v):
    return v.decode() if isinstance(v, bytes) else v


def _first_doc_fields(raw) -> dict | None:
    # RESP3 (Redis 8 default): FT.SEARCH returns a map with 'results'.
    if isinstance(raw, dict):
        results = raw.get("results") or raw.get(b"results")
        if not results:
            return None
        attrs = results[0].get("extra_attributes") or results[0].get(b"extra_attributes") or {}
        return {_decode(k): _decode(v) for k, v in attrs.items()}
    # RESP2: [count, key, [f1, v1, f2, v2, ...], ...]
    if not raw or _decode(raw[0]) in (0, "0") or len(raw) < 3:
        return None
    flat = raw[2]
    return {_decode(flat[i]): _decode(flat[i + 1]) for i in range(0, len(flat), 2)}
