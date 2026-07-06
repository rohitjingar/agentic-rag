"""Domain models shared across ingestion, retrieval, and generation."""

from __future__ import annotations

from pydantic import BaseModel


class Document(BaseModel):
    id: str
    source_path: str
    title: str
    text: str
    meta: dict[str, str] = {}
    content_sha256: str


class Chunk(BaseModel):
    id: str
    doc_id: str
    ord: int
    text: str
    token_count: int
    char_start: int
    char_end: int


class RetrievedChunk(BaseModel):
    id: str
    doc_id: str
    ord: int
    text: str
    score: float


class LLMResponse(BaseModel):
    text: str
    tokens_in: int
    tokens_out: int
    duration_ms: float


class StageTimings(BaseModel):
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0


class QueryResult(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
    tokens_in: int
    tokens_out: int
    timings: StageTimings
