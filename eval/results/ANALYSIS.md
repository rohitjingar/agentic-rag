# Results analysis — what each upgrade earned

Companion to RESULTS.md / GENERATION_RESULTS.md: the *interpretation* behind
each row. Every retrieval change must earn its place with a measured delta; this
is where we're honest about what it did and did not buy.

## Retrieval (golden v1, 46 answerable, top-10)

| config | recall@5 | recall@10 | MRR | nDCG@10 | vs dense |
|---|---|---|---|---|---|
| dense (baseline) | 0.412 | 0.540 | 0.433 | 0.395 | — |
| sparse (BM25/FTS) | 0.289 | 0.332 | 0.259 | 0.237 | worse alone |
| hybrid (RRF) | 0.427 | 0.563 | 0.402 | 0.389 | recall ↑, MRR ↓ |

### Phase 4 — hybrid: a MIXED result, recorded honestly

Hybrid is **not** a clean win, and eval-first is exactly what surfaces that:

- **Recall improved**: recall@10 0.540 → 0.563 (+4.2%), recall@5 +3.6%. Hybrid
  pulls more primary chunks into the candidate pool.
- **Ranking regressed**: MRR 0.433 → 0.402 (−7%), nDCG@10 −1.7%. The first
  correct chunk lands *lower* on average.

Why: sparse alone is weak here (recall@10 0.33 — most questions are semantic,
not lexical). RRF fuses by rank, so when dense already ranks the answer #1 but
sparse ranks something else #1, the fusion *demotes* dense's correct hit.
Measured: on ~6 questions dense had MRR 1.0 and hybrid dropped it (q002 1.0 →
0.12, q031 1.0 → 0.25).

But it **rescued 7 questions dense missed entirely** (recall@10 0 → >0), exactly
the lexical/exact-identifier cases the failure analysis predicted:

| qid | question gist | dense | hybrid | why |
|---|---|---|---|---|
| q038 | "two standard transport mechanisms" (MCP) | 0.0 | 1.0 | exact phrase |
| q073 | WAL: why commit is cheap | 0.0 | 1.0 | lexical term overlap |
| q066 | declare a request body (Pydantic BaseModel) | 0.0 | 1.0 | exact term |
| q063 | GIN "inverted indexes" | 0.0 | 0.5 | exact identifier |
| q037, q044, q022 | | 0.0 | >0 | lexical |

...at the cost of demoting 2 (q005, q070) that dense had.

**Verdict:** hybrid earns its place as a **recall widener**, not a precision
improver. That's the classic *retrieve-then-rerank* division of labor: widen the
pool now (done), restore precision with the cross-encoder next (Phase 5). We did
NOT tune RRF k/fanout against the golden set — that would overfit the eval. The
MRR the reranker must recover is a known, quantified target: back above 0.433.

_Interview line:_ "Hybrid alone traded 7% MRR for 4% recall — a bad deal on its
own. It only pays off because reranking turns that wider pool back into
precision. Eval-first is how I knew to keep going rather than ship hybrid as a
'win.'"

### Phase 5 — reranking: the real win, AND hybrid fails to earn its keep

| config | recall@5 | recall@10 | MRR | nDCG@10 | retrieve p50 ms |
|---|---|---|---|---|---|
| dense (baseline) | 0.412 | 0.540 | 0.433 | 0.395 | 14 |
| hybrid | 0.427 | 0.563 | 0.402 | 0.389 | 36 |
| **rerank (hybrid base)** | **0.573** | 0.627 | 0.477 | 0.459 | 526 |
| rerank (dense base) | 0.569 | 0.653 | 0.488 | 0.475 | 504 |

**Reranking is the biggest single win.** rerank(hybrid) vs dense baseline:
recall@5 +39% (0.412 → 0.573), nDCG@10 +16%, MRR +10%. It recovered the
precision hybrid lost *and* surpassed the baseline — the cross-encoder scores
each (query, chunk) pair jointly and pulls the right chunks from the 50-wide
pool up into the top-5.

**The cost, quantified:** retrieve latency 14 ms → ~510 ms p50 (~36×). The
cross-encoder runs 50 model inferences per query with no precomputation. That
is the precision-for-latency trade, measured — for a latency-sensitive service
you'd shrink the pool, cache, or rerank only low-confidence queries (Phase 6).

**Hybrid does NOT earn its place in the final pipeline (the honest headline).**
rerank(dense) ties — actually slightly beats — rerank(hybrid): recall@10 0.653
vs 0.627, MRR 0.488 vs 0.477. Per-question, the exact-identifier cases hybrid
rescued over *plain* dense (q037, q038, q063, q073) are **all recovered by
dense+rerank too** — those chunks were in dense's top-50 pool all along, just
ranked 11–50; the reranker surfaces them without BM25's help. So on this
corpus BM25 + RRF adds an FTS index and fusion complexity for ~zero gain once
reranking is present.

Kept honestly in the table, not silently dropped. This does not mean hybrid is
useless — on a lexical/code/log corpus, or with a smaller rerank pool, BM25
would likely still pay. On *this* documentation corpus, measured, it did not.

_Interview line:_ "I added BM25 hybrid retrieval and it improved recall. Then I
measured the full pipeline and found dense+rerank matched hybrid+rerank — my own
addition didn't earn its place once the reranker was in. I kept the number in
the table anyway. That willingness to measure your own work out of the pipeline
is the whole point."

_Pipeline choice:_ the results-table narrative follows baseline → hybrid →
rerank → agentic (rerank over the hybrid stack). In production on this corpus
I'd default to **dense + rerank** (simpler, equal-or-better) and keep hybrid
behind a flag for lexical-heavy deployments.

### Phase 6 — agentic loop: pays off ONLY for vocabulary mismatch

| config | recall@5 | recall@10 | MRR | nDCG@10 | p50/p95 ms | tokens/q |
|---|---|---|---|---|---|---|
| rerank (hybrid) | 0.573 | 0.627 | 0.477 | 0.459 | 526/653 | 0 |
| **agentic** | 0.595 | 0.648 | 0.488 | 0.473 | 1516/8246 | 219 |

Aggregate delta vs rerank is small (+3–4%). But the *per-type* breakdown is the
whole story — the gain is entirely in one bucket:

| qtype | rerank r@10 | agentic r@10 | delta | mean iters |
|---|---|---|---|---|
| factoid | 0.730 | 0.730 | **+0.000** | 1.10 |
| how-to | 0.564 | 0.564 | **+0.000** | 1.07 |
| multi-hop | 0.750 | 0.750 | **+0.000** | 1.25 |
| vocab-mismatch | 0.417 | **0.542** | **+0.125** | 1.38 |

The loop earns its keep on **vocabulary mismatch and nowhere else**. The HyDE
re-query — write a hypothetical answer in documentation vocabulary, retrieve
with that — closes exactly the gap that sinks vocab-mismatch queries. Concretely
it fixed **q064** (the Failure-2 case: "stop my API blocking on a slow call" →
FastAPI's async/concurrency docs), 0.00 → 1.00 recall@10. It lost nothing.

For the other 83% of queries the loop added **zero retrieval gain** — rerank
already handled them — so the classify call and any re-query were pure overhead.
Decomposition on multi-hop likewise added cost without moving recall (rerank
already got multi-hop to 0.75).

**When does the agentic loop pay for itself?** When the corpus/query mix has a
real vocabulary gap. Measured here: worth it for ~17% of queries, overhead for
the rest. **What stops it looping forever?** The iteration cap (2) and the
per-query token budget (6000) — verified by `test_loop_terminates_at_cap`.

**Cost, and the loop-engineering lesson.** Mean 219 tokens/query, but a
classification call fires on *every* query while only vocab-mismatch benefits.
The efficient redesign: drop the always-on classifier and trigger the HyDE
re-query purely off the confidence heuristic (retrieve → rerank → if top score
< 2.0, re-query) — paying LLM cost only on the low-confidence tail (~15% of
queries) instead of all of them. That would cut mean tokens ~5× for the same
retrieval gain. (Left as a documented improvement — the current version shows
the full classify/decompose/critique/HyDE machinery the spec asked for.)

_Interview line:_ "The agentic loop improved recall 3% overall — but averaged
across types that hides the real result: +12.5% on vocabulary-mismatch queries,
0% everywhere else. It fixed the exact async-vs-blocking failure I found in
Phase 1. The honest cost read is that classifying every query to help 17% of
them is wasteful; the confidence heuristic alone should gate the LLM spend. Cap
+ token budget guarantee termination."

## Full ladder (retrieval, vs dense baseline)

| config | recall@5 | recall@10 | nDCG@10 | Δ recall@5 |
|---|---|---|---|---|
| dense (baseline) | 0.412 | 0.540 | 0.395 | — |
| hybrid | 0.427 | 0.563 | 0.389 | +3.6% |
| rerank | 0.573 | 0.627 | 0.459 | +38.9% |
| agentic | 0.595 | 0.648 | 0.473 | +44.3% |

Reranking is the load-bearing upgrade; the agentic loop adds a targeted
vocab-mismatch gain on top. Hybrid's standalone contribution washes out once
reranking is present.

### Phase 7 — observability, cost, semantic cache

**Semantic cache (Redis 8 vector search).** Warm 8 questions, replay 19
(repeats + close paraphrases + novel):

| workload slice | hit rate |
|---|---|
| exact repeat | **1.00** |
| close paraphrase | 0.44 |
| novel (must not hit) | **0.00** |
| overall | 0.63 |

Each hit skips retrieve + generate entirely: saves the generation cost
(~$0.0006 shadow/query at Haiku-class rates) **and** the full retrieve+generate
latency (rerank ~510 ms + generation seconds). The **0.00 false-hit rate on
novel queries is the number that matters** — a semantic cache that serves a
wrong answer to a different question is worse than no cache.

Threshold tuned to **0.90** on measured probes, not guessed: bge-small scores
close paraphrases ~0.90–0.92 but a genuinely different question ("HNSW index"
vs "partial index") at 0.74, and heavy rewordings drop to 0.66–0.80. 0.90 is the
precision-first line — it catches near-duplicates and rejects the 0.74 near-miss.
That's why paraphrase hit-rate is only 0.44: the looser paraphrases fall below
0.90 and correctly miss rather than risk a wrong answer.

**Honest invalidation.** Cache entries are namespaced by
`corpus_version : chunk_config_hash : retrieval_mode`, so re-ingesting or
changing the pipeline can never serve a stale answer — those queries miss under
the new namespace. Plus a TTL. Verified by `test_namespace_isolation`.

**Quality invariance.** A hit returns the *exact stored answer + sources*, so
the cache changes cost and latency, never answer content — no re-eval needed to
know retrieval/generation quality is unaffected by definition.

**Cost + tracing.** `shadow_usd` is on every `/query` response (LLM tokens
priced at public rates; local compute shows as latency). OTel spans per stage
(cache.get / retrieve / generate) carry token + cost attributes and export to
Jaeger — verified a live trace with nested `rag.query → retrieve → generate`
spans. Per-stage p95 (from the eval runs): retrieve ≈ 25 ms (dense) → 653 ms
(rerank) → 8.2 s (agentic re-query); generation dominates end-to-end at
seconds (local llama3.1).
