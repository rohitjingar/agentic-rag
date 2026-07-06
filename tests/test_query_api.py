import httpx
from fastapi.testclient import TestClient

from rag.api.app import chunk_config_from, create_app
from rag.generation.client import OllamaClient
from rag.ingest.embedder import FakeEmbedder
from rag.ingest.pipeline import ingest_corpus


def mock_llm(answer: str) -> OllamaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": answer},
                "prompt_eval_count": 10,
                "eval_count": 5,
                "total_duration": 1_000_000_000,
            },
        )

    return OllamaClient("http://ollama.test", "m", transport=httpx.MockTransport(handler))


def test_query_end_to_end(settings, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "guide.md").write_text("# Guide\n\nDependency injection uses Depends().")
    ingest_corpus(settings.database_url, corpus, FakeEmbedder(), chunk_config_from(settings))

    app = create_app(settings, embedder=FakeEmbedder(), llm=mock_llm("Use Depends(). [S1]"))
    with TestClient(app) as client:
        resp = client.post("/query", json={"question": "How does DI work?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Use Depends(). [S1]"
    assert body["sources"], "expected at least one retrieved source"
    assert body["tokens_in"] == 10
    assert body["timings"]["retrieve_ms"] >= 0


def test_query_refuses_when_nothing_indexed(settings):
    # a chunk-size tweak changes the config hash -> no chunks exist for it
    lonely = settings.model_copy(update={"chunk_size_tokens": 123})
    app = create_app(lonely, embedder=FakeEmbedder(), llm=mock_llm("unused"))
    with TestClient(app) as client:
        resp = client.post("/query", json={"question": "Anything at all here?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"] == []
    assert "does not contain the answer" in body["answer"]


def test_query_validates_input(settings):
    app = create_app(settings, embedder=FakeEmbedder(), llm=mock_llm("unused"))
    with TestClient(app) as client:
        assert client.post("/query", json={"question": "hi"}).status_code == 422
        assert client.post("/query", json={"question": "valid?", "top_k": 0}).status_code == 422
