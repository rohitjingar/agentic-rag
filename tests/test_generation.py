import asyncio
import json

import httpx

from rag.generation.client import OllamaClient
from rag.generation.prompts import REFUSAL, SYSTEM_ANSWER, build_user_prompt
from rag.models import RetrievedChunk


def make_transport(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "FastAPI uses Starlette. [S1]"},
                "prompt_eval_count": 42,
                "eval_count": 17,
                "total_duration": 2_500_000_000,
            },
        )

    return httpx.MockTransport(handler)


def test_chat_payload_and_parsing():
    captured: dict = {}
    client = OllamaClient(
        "http://ollama.test", "llama3.1:8b", num_ctx=4096, transport=make_transport(captured)
    )
    response = asyncio.run(client.chat("sys prompt", "user prompt"))

    assert captured["path"] == "/api/chat"
    payload = captured["payload"]
    assert payload["model"] == "llama3.1:8b"
    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0, "num_ctx": 4096}
    assert payload["messages"][0] == {"role": "system", "content": "sys prompt"}
    assert "format" not in payload

    assert response.text == "FastAPI uses Starlette. [S1]"
    assert response.tokens_in == 42
    assert response.tokens_out == 17
    assert response.duration_ms == 2500.0


def test_chat_json_schema_forwarded():
    captured: dict = {}
    client = OllamaClient("http://ollama.test", "m", transport=make_transport(captured))
    schema = {"type": "object", "properties": {"score": {"type": "integer"}}}
    asyncio.run(client.chat("s", "u", json_schema=schema))
    assert captured["payload"]["format"] == schema


def test_grounded_prompt_structure():
    chunks = [
        RetrievedChunk(id="a#1", doc_id="fastapi/testing", ord=0, text="TestClient.", score=0.9),
        RetrievedChunk(id="b#2", doc_id="postgres/indexes", ord=1, text="B-tree.", score=0.5),
    ]
    prompt = build_user_prompt("How do I test?", chunks)
    assert "[S1] (from fastapi/testing)" in prompt
    assert "[S2] (from postgres/indexes)" in prompt
    assert prompt.index("[S1]") < prompt.index("[S2]")
    assert prompt.rstrip().endswith("Question: How do I test?")
    assert REFUSAL in SYSTEM_ANSWER
