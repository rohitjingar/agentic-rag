"""Judge plumbing — parsing, prompt versioning, context formatting — with a
mocked LLM so this runs in CI (no Ollama). The judge's *judgment* is validated
separately in test_judge_regression.py against the real model.
"""

import asyncio
import json

import httpx

from eval.judge.judge import JUDGE_SCHEMA, Judge, JudgeScores
from rag.generation.client import OllamaClient
from rag.models import RetrievedChunk


def judge_with_reply(reply: dict, captured: dict | None = None) -> Judge:
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": json.dumps(reply)},
                "prompt_eval_count": 100,
                "eval_count": 20,
                "total_duration": 1_000_000_000,
            },
        )

    llm = OllamaClient("http://judge.test", "qwen", transport=httpx.MockTransport(handler))
    return Judge(llm)


def test_parses_scores_and_computes_mean():
    reply = {
        "faithfulness": 5,
        "groundedness": 4,
        "relevance": 3,
        "faithfulness_reason": "all claims supported",
    }
    scores = asyncio.run(judge_with_reply(reply).score("q?", "a", []))
    assert (scores.faithfulness, scores.groundedness, scores.relevance) == (5, 4, 3)
    assert scores.mean == 4.0


def test_sends_schema_and_versioned_prompt():
    captured: dict = {}
    j = judge_with_reply({"faithfulness": 3, "groundedness": 3, "relevance": 3}, captured)
    chunks = [
        RetrievedChunk(id="d#1", doc_id="pgvector/README", ord=1, text="40 by default", score=0.9)
    ]
    asyncio.run(j.score("What is the default?", "It is 40. [S1]", chunks))

    payload = captured["payload"]
    assert payload["format"] == JUDGE_SCHEMA  # decoding constrained to the schema
    assert payload["options"]["temperature"] == 0
    user_msg = payload["messages"][1]["content"]
    assert "What is the default?" in user_msg
    assert "[S1] (from pgvector/README)" in user_msg  # context formatted with tags
    assert "It is 40. [S1]" in user_msg


def test_empty_context_is_marked():
    captured: dict = {}
    j = judge_with_reply({"faithfulness": 5, "groundedness": 5, "relevance": 5}, captured)
    asyncio.run(j.score("q?", "refusal", []))
    assert "(no context retrieved)" in captured["payload"]["messages"][1]["content"]


def test_scores_model_bounds():
    s = JudgeScores(faithfulness=1, groundedness=2, relevance=3)
    assert s.mean == 2.0
