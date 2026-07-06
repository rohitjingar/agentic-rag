# Agentic RAG — Project Constitution

## What this project is
An Agentic RAG system with a REAL evaluation harness: hybrid retrieval
(pgvector dense + BM25 sparse) with cross-encoder reranking, an agentic
query planner that critiques and re-queries low-confidence retrievals,
a golden dataset, LLM-as-judge faithfulness/groundedness scoring,
regression tests wired into CI as eval gates, and per-query cost/latency
observability with semantic caching.

Full month context, career goals, and skills tracker: read `docs/CONTEXT.md`
Current spec and task plan: read `specs/002-agentic-rag-spec.md`

## Who you are working with (IMPORTANT — read every session)
Rohit: strong backend engineer (Python, distributed systems, Redis,
PostgreSQL, AWS) who just shipped agent-gateway (production MCP gateway:
JWT/RBAC, rate limiting, OTel, HITL approvals, config-as-data, Redis
pub/sub policy sync). He is NEW to: retrieval systems, embeddings,
evaluation of LLM outputs. Reuse patterns he already knows from
agent-gateway (FastAPI structure, OTel wiring, Docker Compose, CI with
service containers) instead of re-explaining them.

## The one big idea of this project (repeat it often)
EVAL-FIRST. The measurement harness is built BEFORE retrieval
improvements, then every upgrade (hybrid, reranking, agentic loop)
must EARN its place with a measured delta on the golden set. "You
can't improve what you can't measure" — this inversion is the whole
interview story. Preserve this ordering even in continuous build mode.

## Build & learning protocol (non-negotiable)
1. BUILD MODE: implement phases autonomously in spec order without
   interleaved teaching. Still required: Plan Mode master plan first
   (all phases, acceptance gates, checkpoint locations) approved by
   Rohit before any code; one atomic commit per phase minimum; tests
   + evals green before moving to the next phase; every retrieval or
   generation change logs its eval delta in the results table.
2. DECISION CHECKPOINTS: stop and consult Rohit ONLY for these four:
   (a) corpus choice, (b) BM25 approach (PG full-text vs rank-bm25),
   (c) generation/judge LLM budget (Anthropic API vs Ollama),
   (d) golden-set methodology. Present tradeoffs in ~5 lines each,
   wait for his call, record the decision and rationale in
   docs/DECISIONS.md.
3. GOLDEN SET: you may draft questions, but Rohit must personally
   review/edit at least 15-20 of them before evals are considered
   valid. No leakage of chunk wording into questions.
4. LEARNING PACK: after the build passes all evals, write
   docs/LEARNING.md — for each phase: the problem, the design
   decision, why the alternatives lost, one small concrete example
   (mapped to backend ideas he knows: indexes, caches, test suites,
   CI, monitoring), and the eval delta it produced. End with the full
   results table (baseline → hybrid → rerank → agentic).
5. VIVA: after Rohit reads LEARNING.md, run an interview-style quiz —
   one probing question at a time, he answers, you correct and drill
   deeper where he is shaky. Cover: eval methodology, hybrid-vs-dense,
   reranking tradeoff, agentic-loop cost justification, CI eval gates.
   End with a list of his weak spots and what to re-read.

## Workflow rules
- Spec-driven: no code outside the spec's scope. Scope changes get
  written into the spec first.
- One phase = at least one atomic commit (conventional commits).
- Tests alongside features. pytest. CI must stay green.
- No secrets in code — .env + pydantic-settings, .env gitignored.
- eval/ is a first-class top-level package, not an afterthought.
- Never mark a phase done without its eval delta recorded.

## Stack
Python 3.12+, FastAPI, PostgreSQL + pgvector, BM25 (decision
checkpoint b), sentence-transformers for embeddings (local, free),
cross-encoder reranker (local), Anthropic API or Ollama for
generation + LLM-as-judge (decision checkpoint c), Redis (semantic
cache), OpenTelemetry + Jaeger, Docker Compose, uv, ruff.

## Conventions
- src/ layout; small single-purpose modules; type hints everywhere;
  pydantic at all boundaries.
