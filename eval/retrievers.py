"""Retriever registry for the eval runner.

Thin re-export of rag.retrieval.factory so the measurement path runs the exact
same retrieval code as serving. Modes: dense (P2), sparse/hybrid (P4),
rerank (P5), agentic (P6) register there.
"""

from rag.retrieval.factory import (
    DenseRetriever,
    HybridRetriever,
    Retriever,
    SparseRetriever,
    build_retriever,
)

__all__ = [
    "DenseRetriever",
    "HybridRetriever",
    "Retriever",
    "SparseRetriever",
    "build_retriever",
]
