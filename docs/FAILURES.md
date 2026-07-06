# Baseline failure analysis — where naive dense RAG breaks

Phase 1 ships RAG that is basic on purpose: fixed-size chunking (cut docs into
equal-size pieces) → bge-small dense embeddings (turn each chunk into one vector
of numbers that stands for its meaning) → pgvector cosine top-5 (keep the 5
chunks whose vectors point most nearly the same way as the question) → grounded
generation (answer only from those chunks). It *works* on well-phrased questions
whose wording matches the docs. This document records where it **fails**, with
evidence anyone can reproduce, so later phases have concrete targets to beat.
Each failure is tagged with the phase meant to fix it.

Corpus at time of capture: `corpus_version` per `data/corpus/manifest.json`,
399 docs / 3,125 chunks (FastAPI 0.139.0, MCP spec, pgvector, 36 PostgreSQL
pages). Retrieval scores are cosine similarity — a measure of how closely two
vectors point the same way (1.0 = identical direction).

> These are *diagnostic anecdotes* (hand-picked examples), not the eval set. The
> measured golden-set baseline is recorded in Phase 2. The point here is to see
> the failure modes with our own eyes before we build anything to fix them —
> eval-first, but driven by real breakage.

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

**Why it fails:** the question is built around one exact term, `hnsw.ef_search`.
Dense retrieval (search by meaning, not exact words) squeezes the whole question
into a single 384-dim vector — 384 numbers that stand for its meaning — and that
blurs the exact term into a vague "vector-index configuration" direction. So the
search pulls back *other* pgvector/README chunks (and even FastAPI chunks at
0.69) while the one sentence that answers the question never enters the top-5
(the 5 closest chunks). The generator correctly refuses to invent an answer, so
it returns the refusal — **a false negative caused entirely by retrieval.** A
false negative here means a correct answer wrongly reported as missing. This is
the worst kind of RAG bug: the system looks honest ("I don't know") while
sitting on the answer.

**This is the strongest case for hybrid retrieval** (mixing meaning search with
keyword search). BM25 ranks documents by exact word overlap — plain keyword
matching — so `hnsw.ef_search` + `default` would put the answer chunk at the top
no matter how the meaning got blurred. → **Fixed in Phase 4 (hybrid + RRF).**

---

## Failure 2 — Vocabulary mismatch: the right doc never surfaces

**Question:** *"How do I stop my API from blocking while a slow external
service call finishes?"*

**Ground truth:** `fastapi/async.md` (24 KB) is the doc that explains this —
it covers `async`/`await`, concurrency, coroutines, and exactly the
"don't block on external calls" case.

**What naive RAG did (retrieval):**
```
  0.727 fastapi/advanced/stream-data                 (streaming responses — wrong topic)
  0.719 mcp/specification/2025-06-18/basic/lifecycle (MCP shutdown timeouts — wrong)
  0.718 mcp/specification/2025-11-25/basic/lifecycle
  0.705 mcp/seps/2260-Require-Server-requests...
  0.702 mcp/extensions/tasks/overview
```
`fastapi/async.md` does not appear at all.

**Why it fails:** this is a *vocabulary mismatch* — the question and the doc name
the same idea with different words. The *question's* words ("blocking", "slow
external service call") and the *doc's* words ("concurrency", "await",
"coroutine") barely overlap. They don't match as plain text, **and** they don't
match in embedding space either (the number-vector view of meaning), because
bge-small maps these surface terms to different regions. So the retriever latches
onto the word "finishes"/"timeout"/"stream" and wanders into MCP lifecycle docs.

Note that BM25 **won't** save this one — the question and the answer doc share
almost no literal words either. This is the case for **query transformation**
(reword the question before searching): rewrite the user's phrasing toward the
doc's vocabulary, or write a fake ideal answer and search with that (HyDE). →
**Fixed in Phase 6 (agentic transform).**

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
pipeline *noticed* that the best match scored only 0.68 — barely above the
noise floor for low-relevance results (the score level where nothing really
matches). The refusal happened only because llama3.1 read five FastAPI chunks
and judged them irrelevant to Django. There is **no retrieval-confidence
signal** — a check on how good the matches are — gating the generator. Swap in a
weaker or more eager model and the same low-score context produces a confident
hallucination (a made-up answer stated as fact), and — because nothing is
measured — nobody notices.

The fix is to make confidence explicit: score the retrieval (top score, spread
of scores, rerank agreement — whether a second, more careful scorer agrees) and
refuse or re-query *before* trusting the generator. → **Fixed in Phase 6
(retrieval-confidence self-critique).**

---

## What already works (honest contrast)

Naive dense retrieval is not useless — it handles well-phrased questions whose
words line up with the docs just fine, which is exactly why teams ship it and
then get surprised later. Examples that retrieved the correct chunk at high
confidence:

- *"Which PostgreSQL index types exist and when should I prefer BRIN over
  B-tree?"* → `postgres/indexes-types` @ 0.858 ✓
- *"What transports does MCP define...?"* → `mcp/.../basic/transports/index`
  @ 0.837 ✓
- *"How do I override a dependency in tests?"* → `fastapi/advanced/testing-
  dependencies` @ 0.839 ✓ (though the "async session" half of the question
  lands in a second doc — a mild multi-hop case, i.e. the full answer is spread
  across more than one doc, for Phase 6 decomposition)

## Aside — data hygiene bug found and fixed during Phase 1

The first ingest (loading docs into the system) pulled MCP's `schema.mdx` files:
machine-generated TypeDoc **HTML dumps** (~1.4 MB of `<div>`/`<svg>` markup, no
real text). They produced ~1,900 junk chunks that took over retrieval for
several probes (e.g. `ts_rank_cd normalization` returned pure HTML). Removing
them dropped the corpus from 5,037 → 3,125 chunks and *fixed* those queries
outright — a reminder that **retrieval quality starts at ingestion**, before any
fancy retriever. Recorded in `scripts/fetch_corpus.py` (`MCP_EXCLUDE`).
