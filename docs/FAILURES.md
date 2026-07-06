# Baseline failure analysis — where naive dense RAG breaks

Phase 1 ships deliberately naive RAG: fixed-size chunking → bge-small dense
embeddings → pgvector cosine top-5 → grounded generation. It *works* on
well-phrased questions whose wording matches the docs. This document records
where it **fails**, with reproducible evidence, so later phases have concrete
targets to beat. Each failure is tagged with the phase meant to fix it.

Corpus at time of capture: `corpus_version` per `data/corpus/manifest.json`,
399 docs / 3,125 chunks (FastAPI 0.139.0, MCP spec, pgvector, 36 PostgreSQL
pages). Retrieval scores are cosine similarity (1.0 = identical direction).

> These are *diagnostic anecdotes*, not the eval set. The measured golden-set
> baseline is recorded in Phase 2. The point here is to see the failure modes
> with our own eyes before we build anything to fix them — eval-first, but
> motivated by real breakage.

---

## Failure 1 — False "I don't know": exact identifier blurred by dense retrieval

**Question:** *"What is the default value of hnsw.ef_search in pgvector?"*

**Ground truth:** The answer **is in the corpus.** `pgvector/README` states
"the dynamic candidate list (`hnsw.ef_search`), which is 40 by default."

**What naive RAG did — live, end-to-end:**
```
answer: "The indexed documentation does not contain the answer to this question."
sources:
  0.713 pgvector/README            (chunk about bit_hamming_ops / Jaccard distance)
  0.711 postgres/textsearch-intro
  0.696 fastapi/tutorial/body
  0.691 fastapi/tutorial/response-model
  0.689 pgvector/README            (chunk about iterative index scans)
```

**Why it fails:** the query is dominated by the exact token `hnsw.ef_search`.
A single 384-dim dense vector smears that identifier into a generic
"vector-index configuration" direction, so it pulls *other* pgvector/README
chunks (and even FastAPI chunks at 0.69) while the one sentence that answers
the question never enters the top-5. The generator, correctly refusing to
invent, returns the refusal — **a false negative caused entirely by retrieval.**
This is the worst kind of RAG bug: the system looks honest ("I don't know")
while sitting on the answer.

**This is the money case for hybrid retrieval.** BM25 ranks documents by exact
lexical overlap, so `hnsw.ef_search` + `default` would score the answer chunk
at the top regardless of semantic blur. → **Fixed in Phase 4 (hybrid + RRF).**

---

## Failure 2 — Vocabulary mismatch: the right doc never surfaces

**Question:** *"How do I stop my API from blocking while a slow external
service call finishes?"*

**Ground truth:** `fastapi/async.md` (24 KB) is the canonical explainer —
it covers `async`/`await`, concurrency, coroutines, and exactly the
"don't block on external calls" scenario.

**What naive RAG did (retrieval):**
```
  0.727 fastapi/advanced/stream-data                 (streaming responses — wrong topic)
  0.719 mcp/specification/2025-06-18/basic/lifecycle (MCP shutdown timeouts — wrong)
  0.718 mcp/specification/2025-11-25/basic/lifecycle
  0.705 mcp/seps/2260-Require-Server-requests...
  0.702 mcp/extensions/tasks/overview
```
`fastapi/async.md` does not appear at all.

**Why it fails:** the *question's* vocabulary ("blocking", "slow external
service call") and the *doc's* vocabulary ("concurrency", "await", "coroutine")
barely overlap — lexically **or** in embedding space, because bge-small maps
the surface terms to different regions. The retriever latches onto the word
"finishes"/"timeout"/"stream" and wanders into MCP lifecycle docs.

Note BM25 **won't** save this one — the question and the answer doc share
almost no literal terms either. This is the case for **query transformation**:
rewrite the user's phrasing toward doc vocabulary, or generate a hypothetical
answer and search with that (HyDE). → **Fixed in Phase 6 (agentic transform).**

---

## Failure 3 — No confidence gate: refusal works only by the LLM's good grace

**Question:** *"How do I configure connection pooling in Django ORM?"*
(out of scope — the corpus has FastAPI/MCP/pgvector/Postgres, no Django).

**What naive RAG did — live, end-to-end:**
```
answer: "The indexed documentation does not contain the answer to this question."
sources:
  0.680 fastapi/advanced/settings
  0.671 fastapi/advanced/settings
  0.664 fastapi/reference/httpconnection
  0.651 fastapi/tutorial/query-param-models
  0.650 mcp/docs/develop/clients/client-best-practices
```

**Why it's still a failure (even though it refused correctly):** nothing in the
pipeline *noticed* that the best match scored 0.68 — barely above the
low-relevance noise floor. The refusal happened only because llama3.1 read five
FastAPI chunks and judged them irrelevant to Django. There is **no
retrieval-confidence signal** gating the generator. Swap in a weaker or more
eager model and the same low-score context produces a confident hallucination,
and — because nothing is measured — nobody notices.

The fix is to make confidence explicit: score the retrieval (top score,
score distribution, rerank agreement) and refuse or re-query *before* trusting
the generator. → **Fixed in Phase 6 (retrieval-confidence self-critique).**

---

## What already works (honest contrast)

Naive dense retrieval is not useless — it handles well-phrased, vocabulary-
aligned questions fine, which is exactly why teams ship it and then get
surprised. Examples that retrieved the correct chunk at high confidence:

- *"Which PostgreSQL index types exist and when should I prefer BRIN over
  B-tree?"* → `postgres/indexes-types` @ 0.858 ✓
- *"What transports does MCP define...?"* → `mcp/.../basic/transports/index`
  @ 0.837 ✓
- *"How do I override a dependency in tests?"* → `fastapi/advanced/testing-
  dependencies` @ 0.839 ✓ (though the "async session" half of the question
  splits into a second doc — a mild multi-hop case for Phase 6 decomposition)

## Aside — data hygiene bug found and fixed during Phase 1

The first ingest pulled MCP's `schema.mdx` files: machine-generated TypeDoc
**HTML dumps** (~1.4 MB of `<div>`/`<svg>` markup, no prose). They produced
~1,900 junk chunks that dominated retrieval for several probes (e.g.
`ts_rank_cd normalization` returned pure HTML). Excluding them dropped the
corpus from 5,037 → 3,125 chunks and *fixed* those queries outright — a
reminder that **retrieval quality starts at ingestion**, before any fancy
retriever. Recorded in `scripts/fetch_corpus.py` (`MCP_EXCLUDE`).
