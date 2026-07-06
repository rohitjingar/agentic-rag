"""Semantic cache mechanics against real Redis vector search.

Uses a StubEmbedder with hand-built unit vectors of known cosine similarity, so
the threshold behaviour is deterministic (no model download). Paraphrase
behaviour with the real embedder is measured in the cache eval, not here.
"""

import asyncio
import math

import pytest
import redis.asyncio as aioredis

from rag.cache.semantic import SemanticCache
from rag.models import RetrievedChunk
from tests.conftest import REDIS_URL

DIM = 4


def unit(cos_to_e0: float) -> list[float]:
    """A 4-d unit vector whose cosine similarity to [1,0,0,0] is cos_to_e0."""
    return [cos_to_e0, math.sqrt(max(0.0, 1 - cos_to_e0**2)), 0.0, 0.0]


class StubEmbedder:
    dim = DIM

    def __init__(self, mapping: dict[str, list[float]]):
        self._m = mapping

    def encode_query(self, text: str) -> list[float]:
        return self._m[text]

    def encode_passages(self, texts):
        return [self._m[t] for t in texts]


VECTORS = {
    "orig": unit(1.0),  # [1,0,0,0]
    "near": unit(0.92),  # sim 0.92 to orig -> hits at 0.90
    "far": unit(0.50),  # sim 0.50 -> misses
    "ortho": [0.0, 0.0, 1.0, 0.0],  # sim 0 -> misses
}
SRC = [RetrievedChunk(id="d#1", doc_id="pgvector/README", ord=1, text="40 by default", score=0.9)]


async def _fresh_redis():
    r = aioredis.from_url(REDIS_URL)
    await r.flushdb()
    return r


def make_cache(redis, namespace="nsA", threshold=0.90):
    return SemanticCache(redis, StubEmbedder(VECTORS), namespace, threshold=threshold, dim=DIM)


async def _put_orig(cache):
    await cache.put("orig", "The default is 40. [S1]", SRC, tokens_in=100, tokens_out=20)


def test_exact_and_near_hit_far_miss():
    async def run():
        r = await _fresh_redis()
        cache = make_cache(r)
        await cache.ensure_index()
        await _put_orig(cache)

        exact = await cache.get("orig")
        near = await cache.get("near")
        far = await cache.get("far")
        ortho = await cache.get("ortho")
        await r.aclose()
        return exact, near, far, ortho

    exact, near, far, ortho = asyncio.run(run())
    assert exact is not None and exact.answer == "The default is 40. [S1]"
    assert exact.tokens_in == 100 and exact.sources[0].doc_id == "pgvector/README"
    assert near is not None and near.similarity >= 0.90  # near-duplicate hits
    assert far is None  # 0.50 similarity is a miss
    assert ortho is None


def test_threshold_is_respected():
    async def run(threshold):
        r = await _fresh_redis()
        cache = make_cache(r, threshold=threshold)
        await cache.ensure_index()
        await _put_orig(cache)
        hit = await cache.get("near")  # sim 0.92
        await r.aclose()
        return hit

    assert asyncio.run(run(0.90)) is not None  # 0.92 >= 0.90 -> hit
    assert asyncio.run(run(0.95)) is None  # 0.92 < 0.95 -> miss


def test_namespace_isolation_is_invalidation():
    async def run():
        r = await _fresh_redis()
        writer = make_cache(r, namespace="corpus_v1")
        await writer.ensure_index()
        await _put_orig(writer)
        # a different corpus/config namespace must not see v1's entry
        reader = make_cache(r, namespace="corpus_v2")
        miss = await reader.get("orig")
        same = make_cache(r, namespace="corpus_v1")
        hit = await same.get("orig")
        await r.aclose()
        return miss, hit

    miss, hit = asyncio.run(run())
    assert miss is None  # stale namespace -> miss (honest invalidation)
    assert hit is not None


@pytest.mark.parametrize("threshold", [0.90])
def test_empty_cache_misses(threshold):
    async def run():
        r = await _fresh_redis()
        cache = make_cache(r, threshold=threshold)
        await cache.ensure_index()
        hit = await cache.get("orig")
        await r.aclose()
        return hit

    assert asyncio.run(run()) is None
