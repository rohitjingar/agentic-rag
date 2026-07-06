"""Query classification — decides which agentic moves (if any) a query needs.

Cheap up-front call: most queries are simple and need no extra work, so
classification lets the loop spend LLM budget only where it might pay
(decomposition for multi-hop, HyDE for vocabulary mismatch).
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from rag.generation.client import OllamaClient

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "query_type": {
            "type": "string",
            "enum": ["factoid", "how_to", "multi_hop", "out_of_scope"],
        },
        "reason": {"type": "string"},
    },
    "required": ["query_type"],
}

_SYSTEM = """You classify a user's question about a technical documentation corpus
(FastAPI, PostgreSQL, pgvector, and the Model Context Protocol). Choose one:

- factoid: asks for a single specific fact (a default value, a method name).
- how_to: asks how to accomplish one task.
- multi_hop: needs information about TWO or more distinct things combined
  (e.g. "override a dependency AND use an async session").
- out_of_scope: about a technology NOT in the corpus (Django, Kubernetes,
  Kafka, React, Rust, MongoDB, etc.).

Output only the JSON the schema requires."""


class QueryClass(BaseModel):
    query_type: str
    reason: str = ""

    @property
    def is_multi_hop(self) -> bool:
        return self.query_type == "multi_hop"

    @property
    def is_out_of_scope(self) -> bool:
        return self.query_type == "out_of_scope"


async def classify_query(llm: OllamaClient, query: str) -> tuple[QueryClass, int]:
    """Returns (classification, tokens_used)."""
    resp = await llm.chat(_SYSTEM, f"Question: {query}", json_schema=CLASSIFY_SCHEMA)
    tokens = resp.tokens_in + resp.tokens_out
    try:
        return QueryClass.model_validate(json.loads(resp.text)), tokens
    except Exception:
        # a malformed classification must not break retrieval — treat as simple
        return QueryClass(query_type="factoid", reason="classify parse failed"), tokens
