from fastapi.testclient import TestClient

from rag.api.app import create_app


def test_healthz_ok_without_llm(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checks"]["postgres"] == "ok"
    assert body["checks"]["migrations"] == "ok"
    assert body["checks"]["redis"] == "ok"
    assert body["checks"]["ollama"].startswith("skipped")


def test_healthz_unhealthy_when_llm_required_but_unreachable(settings):
    broken = settings.model_copy(
        update={"require_llm": True, "ollama_base_url": "http://localhost:1"}
    )
    app = create_app(broken)
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 503
    assert resp.json()["checks"]["ollama"].startswith("error")
