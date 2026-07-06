"""Fixed-size token chunking with overlap — the deliberately naive baseline.

Token counts use the embedding model's own tokenizer so every chunk is
guaranteed to fit the encoder's 512-token window. Chunks carry char offsets
into the source document: golden-set labels anchor to (doc_id, span) and get
materialized to chunk ids for whatever chunk config is live.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache

from rag.models import Chunk, Document

MIN_TAIL_TOKENS = 20  # a trailing window shorter than this merges into nothing


@dataclass(frozen=True)
class ChunkConfig:
    size_tokens: int = 400
    overlap_tokens: int = 60
    strategy: str = "fixed"
    tokenizer_model: str = "BAAI/bge-small-en-v1.5"

    @property
    def config_hash(self) -> str:
        payload = json.dumps(
            {
                "size": self.size_tokens,
                "overlap": self.overlap_tokens,
                "strategy": self.strategy,
                "tokenizer": self.tokenizer_model,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


@lru_cache(maxsize=2)
def _tokenizer(model_name: str):
    from transformers import AutoTokenizer

    # model_max_length lifted so encoding whole documents doesn't warn/truncate
    return AutoTokenizer.from_pretrained(model_name, model_max_length=10**9)


def chunk_document(doc: Document, config: ChunkConfig) -> list[Chunk]:
    if config.strategy != "fixed":
        raise ValueError(f"unknown chunking strategy: {config.strategy}")

    encoding = _tokenizer(config.tokenizer_model)(
        doc.text, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets: list[tuple[int, int]] = encoding["offset_mapping"]
    if not offsets:
        return []

    step = config.size_tokens - config.overlap_tokens
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk size")

    chunks: list[Chunk] = []
    for start_tok in range(0, len(offsets), step):
        window = offsets[start_tok : start_tok + config.size_tokens]
        is_tail = start_tok + config.size_tokens >= len(offsets)
        if chunks and len(window) < MIN_TAIL_TOKENS:
            break  # tiny tail already covered by the previous window's overlap
        char_start = window[0][0]
        char_end = window[-1][1]
        text = doc.text[char_start:char_end]
        if not text.strip():
            continue
        ord_ = len(chunks)
        chunks.append(
            Chunk(
                id=f"{doc.id}#{config.config_hash}:{ord_}",
                doc_id=doc.id,
                ord=ord_,
                text=text,
                token_count=len(window),
                char_start=char_start,
                char_end=char_end,
            )
        )
        if is_tail:
            break
    return chunks
