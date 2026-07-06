"""Shadow cost accounting.

Everything runs locally, so real spend is $0. But "free" hides the real
engineering cost of a design choice, so we price LLM tokens at public
small-model rates ("shadow-$") to keep cost a first-class, comparable number.
Embeddings and reranking are local compute (no per-token price) — their cost
shows up as latency, tracked separately.
"""

from __future__ import annotations

# public small-model (Haiku-class) blended reference rates, USD per token
USD_PER_INPUT_TOKEN = 0.25 / 1_000_000
USD_PER_OUTPUT_TOKEN = 1.25 / 1_000_000


def shadow_usd(tokens_in: int, tokens_out: int) -> float:
    return tokens_in * USD_PER_INPUT_TOKEN + tokens_out * USD_PER_OUTPUT_TOKEN


def shadow_usd_per_1k(tokens_in: int, tokens_out: int) -> float:
    return round(shadow_usd(tokens_in, tokens_out) * 1000, 4)
