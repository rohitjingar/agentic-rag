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
