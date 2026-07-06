"""LLM-as-judge: scores an answer against its retrieved context.

Prompts are versioned on disk (prompts/<version>/); the version is recorded in
every result so a rubric change is a visible, attributable event, not a silent
drift. Decoding is schema-constrained so the judge must return well-formed
scores (no regex-scraping a free-text rating).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from rag.generation.client import OllamaClient
from rag.models import RetrievedChunk

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
        "faithfulness_reason": {"type": "string"},
        "groundedness": {"type": "integer", "minimum": 1, "maximum": 5},
        "groundedness_reason": {"type": "string"},
        "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
        "relevance_reason": {"type": "string"},
    },
    "required": ["faithfulness", "groundedness", "relevance"],
}


class JudgeScores(BaseModel):
    faithfulness: int
    groundedness: int
    relevance: int
    faithfulness_reason: str = ""
    groundedness_reason: str = ""
    relevance_reason: str = ""

    @property
    def mean(self) -> float:
        return (self.faithfulness + self.groundedness + self.relevance) / 3


class Judge:
    def __init__(self, llm: OllamaClient, prompt_version: str = "v1"):
        self.llm = llm
        self.prompt_version = prompt_version
        base = PROMPTS_DIR / prompt_version
        self._system = (base / "system.txt").read_text(encoding="utf-8")
        self._user_template = (base / "user_template.txt").read_text(encoding="utf-8")

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "(no context retrieved)"
        return "\n\n".join(f"[S{i}] (from {c.doc_id})\n{c.text}" for i, c in enumerate(chunks, 1))

    async def score(self, question: str, answer: str, context: list[RetrievedChunk]) -> JudgeScores:
        user = self._user_template.format(
            question=question, context=self._format_context(context), answer=answer
        )
        response = await self.llm.chat(self._system, user, json_schema=JUDGE_SCHEMA)
        return JudgeScores.model_validate(json.loads(response.text))
