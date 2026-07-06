"""Agentic loop: classification/transform parsing, confidence critique, and —
most importantly — that the re-query loop TERMINATES at the iteration cap even
when retrieval never looks confident. All LLM calls are mocked (CI-safe).
"""

import asyncio
import json

import httpx

from rag.agent.classifier import classify_query
from rag.agent.critique import CONFIDENCE_THRESHOLD, is_confident
from rag.agent.loop import ITER_CAP, AgenticRetriever
from rag.agent.transform import decompose
from rag.generation.client import OllamaClient
from rag.models import RetrievedChunk


def mock_llm(reply_content: str) -> OllamaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": reply_content},
                "prompt_eval_count": 50,
                "eval_count": 20,
                "total_duration": 1_000_000_000,
            },
        )

    return OllamaClient("http://llm.test", "m", transport=httpx.MockTransport(handler))


def chunk(cid: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(id=cid, doc_id="d", ord=0, text="text", score=score)


# --- classifier + transforms ---------------------------------------------


def test_classify_parses():
    llm = mock_llm(json.dumps({"query_type": "multi_hop", "reason": "two things"}))
    cls, tokens = asyncio.run(classify_query(llm, "a and b?"))
    assert cls.query_type == "multi_hop"
    assert cls.is_multi_hop
    assert tokens == 70


def test_classify_falls_back_on_garbage():
    llm = mock_llm("not json at all")
    cls, _ = asyncio.run(classify_query(llm, "q?"))
    assert cls.query_type == "factoid"  # degrade to simple, never crash


def test_decompose_parses_and_falls_back():
    llm = mock_llm(json.dumps({"sub_questions": ["q1?", "q2?"]}))
    subs, _ = asyncio.run(decompose(llm, "q1 and q2?"))
    assert subs == ["q1?", "q2?"]
    bad = mock_llm("garbage")
    subs2, _ = asyncio.run(decompose(bad, "original?"))
    assert subs2 == ["original?"]


# --- critique -------------------------------------------------------------


def test_confidence_threshold():
    assert is_confident([chunk("a", CONFIDENCE_THRESHOLD)])
    assert not is_confident([chunk("a", CONFIDENCE_THRESHOLD - 0.1)])
    assert not is_confident([])


# --- loop termination (the safety property) -------------------------------


class _StubAgentic(AgenticRetriever):
    """Override the DB-touching parts; force perpetually-low confidence so the
    loop would spin forever if the cap didn't hold."""

    def __init__(self, llm):
        super().__init__(pool=None, embedder=None, chunk_config_hash="x", reranker=None, llm=llm)
        self.rerank_calls = 0

    async def _gather(self, queries):
        return [chunk("c", 0.0)]

    async def _dense(self, text, k):
        return [chunk("c", 0.0)]

    async def _rerank(self, query, candidates):
        self.rerank_calls += 1
        return [chunk("c", -1.0)]  # always below CONFIDENCE_THRESHOLD


def test_loop_terminates_at_cap_under_low_confidence():
    # classifier says factoid; every hyde call returns a passage; confidence
    # never clears -> loop must still stop at ITER_CAP
    llm = mock_llm(json.dumps({"query_type": "factoid"}))
    r = _StubAgentic(llm)
    out = asyncio.run(r.retrieve("hopeless query", 5))
    assert out  # still returns whatever it found
    assert r.last_run.iterations == ITER_CAP  # capped, did not exceed
    assert r.last_run.iterations <= ITER_CAP


def test_confident_first_pass_costs_one_classification_only():
    # reranker returns a high score -> no re-query, only the classify call
    class _Confident(_StubAgentic):
        async def _rerank(self, query, candidates):
            return [chunk("c", 9.0)]

    llm = mock_llm(json.dumps({"query_type": "factoid"}))
    r = _Confident(llm)
    asyncio.run(r.retrieve("easy query", 5))
    assert r.last_run.iterations == 1
    assert r.last_run.steps == ["classify"]  # no hyde/decompose spent
