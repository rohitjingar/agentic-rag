# Learning Pack — Agentic RAG, explained for a backend engineer

This is your interview study guide. Every section maps a retrieval/eval concept
to something you already own — indexes, caches, test suites, CI, monitoring —
then gives the design decision, why the alternatives lost, and the **measured
delta** it produced. The numbers are the golden-set results in
`eval/results/RESULTS.md` and `ANALYSIS.md`.

## The one big idea: eval-first

We built the **measurement harness before improving retrieval**. Baseline naive
RAG got measured first; then hybrid, reranking, and the agentic loop each had to
*earn their place* with a recorded delta.

**Backend analogy:** shipping RAG without evals is deploying a distributed
service with no tests and no monitoring. The golden set is your test suite;
faithfulness scoring is your prod monitoring; the CI eval gate is your
"tests-must-pass-to-merge" rule — but for probabilistic outputs where "correct"
is a distribution, not a boolean. The whole project is that inversion.

The payoff of doing it this way: we caught **two of our own upgrades not earning
their keep** (hybrid post-rerank; the agentic loop outside vocab-mismatch). An
eyeball-5-questions workflow ships both as "wins."

---

## Phase 1 — Embeddings, vector indexes, chunking

**The problem.** An LLM knows nothing about your private/recent docs and
hallucinates confidently. RAG grounds it: retrieve relevant chunks, stuff them
in the prompt, cite them.

**Embeddings from first principles.** An embedding maps text to a point in
384-dimensional space such that *similar meaning → nearby points*. "How do I
test FastAPI?" and "writing FastAPI tests" land close together even with zero
shared words. Similarity = cosine of the angle between two vectors (1.0 = same
direction, 0 = orthogonal). We normalize vectors to length 1, so cosine is just
a dot product.

*Tiny 2-D example:* imagine "cat"→(0.9, 0.1), "kitten"→(0.86, 0.15),
"database"→(0.05, 0.99). cat·kitten ≈ 0.79 (close); cat·database ≈ 0.14 (far).
Scale that to 384 dims and you have semantic search.

**The vector index — your B-tree-vs-hash instinct, ported.** Comparing the
query against all 3,125 chunks (exact scan) is O(n) — fine at our scale, a
non-starter at millions. pgvector offers two approximate indexes:

- **IVFFlat** partitions vectors into lists (clusters) and only searches the
  nearest few lists. Like a **hash index**: fast lookups, needs a training step
  to learn the buckets, degrades if data shifts.
- **HNSW** builds a navigable small-world graph and greedily walks it toward the
  query. Like a **B-tree**: no training, better recall-speed tradeoff, higher
  memory and slower build.

We chose HNSW (cosine). Honest note: at 3k chunks an exact scan would be fine;
HNSW is here to *demonstrate* the tradeoff. `hnsw.ef_search` (default 40) is the
search-time "how hard do I look" knob — higher = better recall, slower.

**Chunking = your index granularity decision.** Too-large chunks bury the answer
in noise; too-small chunks lose context. We use fixed 400-token windows with
60-token overlap (overlap so an answer straddling a boundary isn't split). Each
chunk stores its **character offsets** into the source doc — this matters for
the golden set (labels anchor to a doc span, then materialize to whatever chunk
config is live, so re-chunking never silently breaks labels).

**The delta.** This phase *is* the baseline (nothing to beat yet). We
deliberately found 3 failures (`docs/FAILURES.md`): a false "I don't know" on an
exact identifier (`hnsw.ef_search`), a vocabulary-mismatch miss (async docs
never surfaced for a "blocking" question), and an ungated out-of-scope query.
Each became a target for a later phase.

---

## Phase 2 — Golden set + retrieval metrics

**The problem.** "It feels better" is not shippable. We need numbers.

**The metrics (know these cold).**
- **recall@k** — of the chunks that truly answer the question, how many are in
  the top-k? "Did we retrieve the answer at all?"
- **MRR** (mean reciprocal rank) — 1/(rank of the first correct chunk). "How
  high did the first right answer land?" MRR 1.0 = always rank 1; 0.5 = rank 2.
- **nDCG@10** — graded: rewards ranking a *primary* chunk above a merely
  *supporting* one, discounted by position. This is why we grade labels
  primary/supporting instead of binary — nDCG is meaningless without grades.

**Building a golden set honestly.** The cardinal sin is writing a question by
paraphrasing the chunk you know is the answer — the retriever then matches
wording no real user would type. Defenses: (1) phrase questions like an engineer
would ask (often with *different* vocabulary — that's the `vocab-mismatch`
type), (2) a mechanical **n-gram leakage check** between each question and its
gold chunk (max observed 0.15, threshold 0.50). 53 questions, 5 types, ~15%
negative controls (out-of-scope, no valid answer — they measure "does it know
when it doesn't know?").

**Backend analogy:** the golden set is a **fixture-based test suite** for a
probabilistic function. The config-hash guard (the runner refuses to score if
the live chunk config doesn't match the labels' materialization) is the same
instinct as a migration that fails closed on checksum drift.

**The delta (the anchor for everything).** Dense baseline: **recall@10 0.540,
MRR 0.433, nDCG@10 0.395**. Reproducible to the digit across runs.

---

## Phase 3 — LLM-as-judge (generation quality)

**The problem.** Retrieval metrics don't tell you if the *answer* is good. But
"good" for free-text has no gold string to `assert ==` against.

**The idea.** Use a second LLM as a grader against a written rubric, scoring
three axes 1–5: **faithfulness** (are the claims supported by the retrieved
context?), **groundedness** (does it stay in-context and cite?), **relevance**
(does it answer the question asked?). Reference-free — the judge sees the same
context the generator saw, mirroring production where you have no gold answer at
serving time.

**Why LLM judges are dangerous, and the fixes.**
- *Self-preference bias* (a model loves its own style) → **cross-family**:
  llama generates, qwen judges.
- *Verbosity bias* (longer = better) → rubric scores support/attribution, not
  length.
- *The judge might just be wrong* → we **regression-test the judge itself**: it
  must separate a known-good answer from a seeded hallucination and an off-topic
  answer *by a margin* before its scores count. (It did: 5/5/5 vs faithfulness=1
  vs 1/1/1.)

**Backend analogy:** the judge is a flaky dependency, so you write a contract
test for it before trusting it in your pipeline.

**The delta.** Naive dense RAG: **faithfulness 4.65, groundedness 4.59,
relevance 4.13, refusal accuracy 1.00**. Relevance sits *below* faithfulness —
because bad retrieval yields answers faithful to the *wrong* context. That gap
is the opening for retrieval upgrades to raise generation quality.

---

## Phase 4 — Hybrid retrieval (BM25 + RRF)

**The problem.** Dense retrieval blurs exact tokens. `hnsw.ef_search`,
`tools/list`, `<=>` — a single 384-dim vector smears these into a generic
region and misses the exact chunk.

**Lexical vs semantic.** BM25 (here via PostgreSQL full-text search) ranks by
*literal term overlap* — perfect for identifiers, useless for paraphrases.
Dense is the opposite. **Fusion** combines them.

**RRF = your composite-index instinct.** Reciprocal Rank Fusion scores a chunk
by `Σ 1/(k + rank_in_each_list)` (k=60). It combines by **rank, not score**, so
the two retrievers' incomparable score scales never need calibrating — like a
composite index letting two access paths contribute without normalizing their
costs. A chunk both retrievers rank highly wins.

**The delta — and the honest catch.** Hybrid improved recall@10 (0.540 → 0.563,
+4.2%) but *hurt* MRR (0.433 → 0.402): fusing a weak sparse ranker demotes
dense's confident top-1 hits. It rescued 7 exact-identifier questions dense
missed entirely (MCP transports, GIN, WAL) but demoted ~6 others. **Verdict:
hybrid earns its place as a recall widener, not a precision improver** — which
is exactly the setup reranking needs. We did *not* tune RRF against the golden
set (that's overfitting the eval).

---

## Phase 5 — Cross-encoder reranking

**The problem.** Hybrid widened the pool but scrambled precision. Recover it.

**Bi-encoder vs cross-encoder (the key distinction).** The bi-encoder (bge, used
for dense retrieval) embeds query and passage **separately** — fast, indexable,
precomputed, but it never sees the two *together*. A **cross-encoder** feeds
`(query, passage)` through the model **jointly** and scores the pair, modeling
term interactions the bi-encoder can't. It's far more accurate but O(candidates)
model calls per query with **no precomputation** — you can't index it.

So the standard split: bi-encoder + BM25 cheaply retrieve a wide pool (top-50),
cross-encoder precisely reranks it to top-k. Fast filter, then expensive precise
sort — the same shape as a cheap index scan feeding an expensive recheck.

**The delta — the big win.** rerank vs dense baseline: **recall@5 0.412 → 0.573
(+39%), nDCG@10 +16%, MRR +10%.** Cost: retrieve latency **14 ms → ~510 ms** (50
cross-encoder inferences/query). The precision-for-latency trade, quantified.

**The second honest finding.** `dense+rerank` **ties/beats** `hybrid+rerank`
(recall@10 0.653 vs 0.627). The exact-identifier cases hybrid rescued were
already in dense's top-50 pool — reranking surfaces them **without BM25**. So on
this corpus, **hybrid does not earn its place once reranking exists.** Kept in
the table anyway. On a lexical/code/log corpus BM25 would likely still pay.

---

## Phase 6 — Agentic loop

**The problem.** Some queries fail no matter how good the retriever, because the
user's words and the doc's words don't overlap (vocabulary mismatch).

**The moves.** Classify the query (factoid/how-to/multi-hop/out-of-scope) →
optionally **decompose** multi-hop into sub-questions → retrieve + rerank →
**self-critique** the confidence (cheap heuristic: is the cross-encoder top
score ≥ 2.0?) → if weak, **HyDE re-query**: ask the LLM to *write a hypothetical
answer* in documentation vocabulary and retrieve with *that*, closing the
vocab gap. Bounded by an **iteration cap (2)** and a **token budget (6000)**.

**Backend analogy:** this is a retry loop with a circuit breaker. The cap +
budget are your max-retries and timeout — they're what stop it looping forever
(verified by `test_loop_terminates_at_cap`). HyDE is a fallback path taken only
when the primary looks unhealthy.

**The delta — precisely targeted.** Overall +3% vs rerank, but that average
hides the real result: the entire gain is in **vocab-mismatch (+12.5% recall@10,
0.417 → 0.542); every other type moved +0.000.** It fixed q064 — the exact
async-vs-blocking failure from Phase 1 — 0 → 1.0. Cost: 219 tokens/query, p95
latency 8.2 s on re-queried questions.

**When does the loop pay for itself?** When there's a real vocabulary gap —
here, ~17% of queries. For the other 83% the classifier is pure overhead. The
documented redesign: drop the always-on classifier, gate the LLM spend on the
confidence heuristic alone → ~5× fewer tokens for the same gain. Knowing *when
not* to be agentic is the senior insight.

---

## Phase 7 — Observability, cost, semantic cache

**Semantic cache = a cache keyed on meaning.** A normal cache keys on the exact
query string, so "how to test FastAPI?" and "writing FastAPI tests" miss each
other. A semantic cache keys on the query **embedding**: a near-duplicate
(cosine ≥ threshold) hits and skips retrieve+generate — the expensive stages.
Redis 8 does the vector search.

**The threshold is a precision/recall dial, and precision wins.** Measured
bge-small similarities: close paraphrases ~0.90–0.92, but a *different* question
("HNSW index" vs "partial index") scored 0.74, and heavy rewordings 0.66–0.80 —
overlapping the different-question range. There's no clean cutoff. We set **0.90
(precision-first)**: a cache that serves the *wrong* cached answer is worse than
no cache. Result: exact-repeat hit 1.00, close-paraphrase 0.44, **novel
false-hit 0.00**.

**Cache invalidation = the honesty problem.** Entries are namespaced by
`corpus_version : chunk_config_hash : retrieval_mode`. Re-ingest or change the
pipeline → new namespace → old entries simply miss. No stale answers, ever. Plus
a TTL. (The two hard things in CS are cache invalidation and naming; we made
invalidation a key-derivation problem so it can't rot.)

**Cost as a first-class metric.** Everything runs locally ($0), so we price LLM
tokens at public rates — **shadow-$** — on every `/query` response. "Free" hides
the real cost of a design choice; shadow-$ makes reranking's latency and the
agentic loop's tokens comparable dollar figures.

**Observability.** OTel spans per stage (cache/retrieve/generate) carry token +
cost + latency attributes, exported to Jaeger — same wiring as agent-gateway.
Per-stage p95: retrieve 25 ms (dense) → 653 ms (rerank) → 8.2 s (agentic);
generation dominates end-to-end.

---

## Phase 8 — CI eval gates

**The idea.** A PR runs the eval suite; a quality regression **blocks merge**,
exactly like a failing unit test — but for retrieval quality. Thresholds live in
`eval/thresholds.yaml` (config-as-data). Retrieval runs fully in CI (bge-small
on CPU, committed corpus, pgvector container). Generation can't (no Ollama in
CI), so we verify the newest **committed, hash-stamped** generation result is
fresh (its `golden_hash` still matches the repo) and above floors — stale or
regressed → blocked. Same enforcement, zero cloud LLM.

**The demonstration (PR #1).** A PR that drops the bge query-instruction prefix
**passes `lint` and `test`** but the `retrieval-gate` catches it —
`GATE FAIL: recall@10 0.489 < floor 0.5` — and blocks the merge. That's the
whole point: unit tests check code correctness; only the golden set catches a
*quality* regression.

**How this changes team velocity.** Without the gate, a well-meaning "small
retrieval tweak" silently degrades quality and nobody notices until users
complain. With it, the regression is caught in CI in minutes, with a number.
It turns "please be careful with retrieval changes" (unenforceable) into a
mechanical gate — the same shift as going from "please write tests" to
"coverage must not drop."

---

## Full results ladder

**Retrieval** (golden set, 46 answerable, top-10):

| config | recall@5 | recall@10 | MRR | nDCG@10 | retrieve p50 | verdict |
|---|---|---|---|---|---|---|
| dense (baseline) | 0.412 | 0.540 | 0.433 | 0.395 | 14 ms | anchor |
| + hybrid (BM25+RRF) | 0.427 | 0.563 | 0.402 | 0.389 | 36 ms | recall↑ MRR↓; washed out by rerank |
| + rerank (cross-encoder) | 0.573 | 0.627 | 0.477 | 0.459 | 526 ms | **the win** (+39% recall@5) |
| + agentic loop | 0.595 | 0.648 | 0.488 | 0.473 | 1516 ms | +12.5% on vocab-mismatch only |

**Generation** (LLM-as-judge, 1–5):

| pipeline | faithfulness | groundedness | relevance |
|---|---|---|---|
| naive dense RAG | 4.65 | 4.59 | 4.13 |
| rerank pipeline | 4.65 | 4.72 | **4.74** |

The headline: **relevance +15% (4.13 → 4.74)** from better retrieval alone —
faithfulness was never the problem (llama grounds well), *relevance* was, because
naive retrieval fed it the wrong context. Fix retrieval, relevance follows. Both
refuse 100% of out-of-scope questions.

**Cache:** overall hit-rate 0.63, exact-repeat 1.00, novel false-hit 0.00.

## The five interview narratives

1. **Why eval-first?** It caught two of my own upgrades not earning their keep.
   Show the table.
2. **Why did hybrid beat dense — then stop mattering?** BM25 rescued exact
   identifiers, but reranking already recovered them from the pool.
3. **When is reranking worth it?** +16% nDCG for ~500 ms — yes for quality-
   sensitive, gate it behind confidence for latency-sensitive.
4. **When does the agentic loop pay for itself?** Only on vocabulary mismatch
   (+12.5%); a cap + token budget stop it looping forever.
5. **How do CI eval gates change how a team ships RAG?** They turn "be careful"
   into a mechanical, numeric merge gate — PR #1 proves it.
