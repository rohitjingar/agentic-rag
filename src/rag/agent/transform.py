"""Query transformations — turn a query the retriever struggles with into one
it handles. Each returns (result, tokens_used) so the loop can budget.

- HyDE: generate a hypothetical *answer passage* and retrieve with that. The
  hypothetical answer is written in documentation vocabulary, closing the
  vocabulary gap between how a user asks and how the docs phrase it (Failure-2).
- decompose: split a multi-hop question into standalone sub-questions so each
  facet can be retrieved independently, then merged.
"""

from __future__ import annotations

import json

from rag.generation.client import OllamaClient

_HYDE_SYSTEM = """Write a short, factual passage (2-4 sentences) that would answer
the user's question, as if it were an excerpt from technical documentation for
FastAPI, PostgreSQL, pgvector, or the Model Context Protocol. Use precise
technical vocabulary and identifiers. If you are unsure of specifics, still write
a plausible documentation-style passage. Output only the passage, no preamble."""

_DECOMPOSE_SYSTEM = """Break the user's question into 2-3 standalone sub-questions,
each retrievable on its own. Keep them minimal and non-overlapping. Output only
JSON matching the schema."""

_DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_questions": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["sub_questions"],
}


async def hyde_passage(llm: OllamaClient, query: str) -> tuple[str, int]:
    resp = await llm.chat(_HYDE_SYSTEM, f"Question: {query}")
    return resp.text.strip(), resp.tokens_in + resp.tokens_out


async def decompose(llm: OllamaClient, query: str) -> tuple[list[str], int]:
    resp = await llm.chat(_DECOMPOSE_SYSTEM, f"Question: {query}", json_schema=_DECOMPOSE_SCHEMA)
    tokens = resp.tokens_in + resp.tokens_out
    try:
        subs = json.loads(resp.text).get("sub_questions", [])
        subs = [s.strip() for s in subs if s.strip()]
        return (subs or [query]), tokens
    except Exception:
        return [query], tokens
