"""The judge is under test: it must separate a good answer from a seeded
hallucination and an off-topic answer BY A MARGIN, or its scores can't be
trusted. Runs against the real local judge; skipped where Ollama is absent
(e.g. CI) — this suite is the local validity gate for generation numbers.
"""

import asyncio

import httpx
import pytest

from eval.judge.judge import Judge
from rag.config import get_settings
from rag.generation.client import OllamaClient
from rag.models import RetrievedChunk

pytestmark = pytest.mark.judge


def _ollama_ready(settings) -> bool:
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        models = {m["name"] for m in resp.json().get("models", [])}
        return settings.judge_model in models
    except Exception:
        return False


QUESTION = "In pgvector, what is the default value of hnsw.ef_search?"
CONTEXT = [
    RetrievedChunk(
        id="pgvector/README#1",
        doc_id="pgvector/README",
        ord=1,
        text="Specify the size of the dynamic candidate list for search (40 by default). "
        "A higher value provides better recall at the cost of speed.",
        score=0.8,
    )
]

GOOD = "The default value of hnsw.ef_search is 40. [S1]"
HALLUCINATED = "The default is 100, and it must be a power of two for the graph to balance. [S1]"
OFFTOPIC = "Create a FastAPI app by instantiating FastAPI() and adding path operation decorators."


@pytest.fixture(autouse=True)
def _require_ollama():
    if not _ollama_ready(get_settings()):
        pytest.skip("judge model not available on Ollama")


async def _score(answer, context=CONTEXT):
    # fresh client per call: reusing one httpx.AsyncClient across separate
    # asyncio.run() loops raises "Event loop is closed"
    settings = get_settings()
    llm = OllamaClient(settings.ollama_base_url, settings.judge_model, num_ctx=settings.llm_num_ctx)
    try:
        return await Judge(llm).score(QUESTION, answer, context)
    finally:
        await llm.aclose()


def score(answer, context=CONTEXT):
    return asyncio.run(_score(answer, context))


def test_good_answer_scores_high():
    s = score(GOOD)
    assert s.faithfulness >= 4
    assert s.relevance >= 4
    assert s.groundedness >= 4


def test_hallucination_scores_low_faithfulness():
    s = score(HALLUCINATED)
    assert s.faithfulness <= 2, f"judge missed the hallucination: {s}"


def test_offtopic_scores_low_relevance():
    s = score(OFFTOPIC)
    assert s.relevance <= 2, f"judge missed the off-topic answer: {s}"


def test_judge_separates_good_from_bad_by_margin():
    good = score(GOOD)
    bad = score(HALLUCINATED)
    assert good.faithfulness - bad.faithfulness >= 2, (
        f"insufficient separation: good={good.faithfulness} bad={bad.faithfulness}"
    )
