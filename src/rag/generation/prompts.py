"""Prompt assembly for grounded answering.

The refusal sentence is a stable constant: evals and the agentic loop detect
"couldn't answer" by matching it instead of guessing from free text.
"""

from __future__ import annotations

from rag.models import RetrievedChunk

REFUSAL = "The indexed documentation does not contain the answer to this question."

SYSTEM_ANSWER = f"""You answer questions strictly from the provided documentation excerpts.

Rules:
- Use ONLY the numbered context excerpts below. No outside knowledge.
- Cite every claim with its excerpt tag, e.g. [S1] or [S1][S3].
- Be concise and technical; include code only when the excerpts contain it.
- If the excerpts do not contain the answer, reply with exactly:
  "{REFUSAL}"
"""


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[S{i}] (from {chunk.doc_id})\n{chunk.text}")
    context = "\n\n".join(blocks)
    return f"Context:\n\n{context}\n\nQuestion: {question}"
