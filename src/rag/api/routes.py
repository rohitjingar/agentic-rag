"""HTTP surface for querying the pipeline."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from rag.models import QueryResult

router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


@router.post("/query")
async def query(request: Request, body: QueryRequest) -> QueryResult:
    service = getattr(request.app.state, "rag", None)
    if service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    return await service.answer(body.question, top_k=body.top_k)
