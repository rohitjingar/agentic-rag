"""Ollama chat client. temperature=0 everywhere: reproducible evals trump variety."""

from __future__ import annotations

from typing import Any

import httpx

from rag.models import LLMResponse


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        num_ctx: int = 8192,
        timeout: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model = model
        self.num_ctx = num_ctx
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)

    async def chat(
        self, system: str, user: str, *, json_schema: dict[str, Any] | None = None
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }
        if json_schema is not None:
            payload["format"] = json_schema  # Ollama constrains decoding to the schema
        resp = await self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(
            text=data["message"]["content"],
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
            duration_ms=data.get("total_duration", 0) / 1e6,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
