# Learning Pack — Agentic RAG, explained for a backend engineer

This is your interview study guide. **Start with the plain-English section right
below** — it explains the whole system in simple words. Everything after it goes
deeper. Each part maps to something you already know (indexes, caches, test
suites, CI, monitoring), with the design decision, why the other options lost,
and the **measured delta** — the change in score — it produced (numbers in
`eval/results/RESULTS.md` + `ANALYSIS.md`).

---

## Start here — the whole thing in plain English

**What this project does.** You ask a question; the system finds the right pages
from a big pile of documents and has an AI write an answer from them, with
citations. Plus — the real point — it can **prove with numbers** whether the
answers are actually good.

**Why bother?** An AI model on its own has two problems: it confidently makes
things up, and it doesn't know *your* documents. So we don't let it answer from
memory — we *fetch the relevant pages first* and say "answer using only these."
Think of the AI as a brilliant new hire who hasn't read the company wiki: instead
of letting them guess, you hand them the 5 right wiki pages first. That's RAG
(Retrieval-Augmented Generation).

**How it finds the right pages — the three "modes."** Imagine a librarian
finding pages for you, from fast-and-dumb to slow-and-smart:

- **`dense`** — searches by *gist*. Every page and your question become a
  "meaning fingerprint" (a list of numbers); she grabs the pages whose
  fingerprints are closest. Fast and decent, but bad with exact terms — ask for
  `ef_search` and she blurs it and grabs the wrong shelf. **(fast, rough)**
- **`rerank`** — dense, then a smarter second pass. The fast librarian hands over
  ~50 maybe-right pages; a slower, smarter model reads each one *next to your
  question* and re-sorts them so the best lands on top. Much more accurate — the
  big quality jump. (It literally "re-ranks" the 50.) **(the win)**
- **`agentic`** — rerank, plus the system checks its own work. If its best result
  looks weak, it doesn't give up: it asks the AI to *write a pretend perfect
  answer in the documents' own words*, then searches again with that — bridging
  the "you said *freezing*, the docs say *async*" gap. Capped at 2 tries so it
  can't loop forever. "Agentic" = it **makes decisions** (retry? how? stop?)
  instead of blindly running one fixed path. **(smartest, slowest, costs tokens)**

  One line: **dense = fast & rough → rerank = add a smart re-sorter → agentic =
  add self-checking + a smart retry.** (You'll also see `sparse` = plain keyword
  search like Ctrl-F, and `hybrid` = dense+sparse blended — those two didn't earn
  their keep, so ignore them.)

**The two halves that can each break.** Answering has two steps, and either can
fail on its own: (1) **finding** the right page, and (2) the AI actually
**reading** it and answering. In the live demo, reranking *found* the page saying
"ef_search is 40 by default," but the small local AI still fumbled and refused.
That's why we score the two halves separately — you need both to work.

**The one big idea — measure before you improve.** Most people build a fancy
pipeline, try 5 questions by hand, and ship. We did the opposite: first we built
a **scorecard** — 53 real questions with their known-correct pages marked — and
made every upgrade *prove* it helps by moving the score. A test suite, but for a
fuzzy system. It's how we knew reranking was real and two of our own upgrades
were not. Same instinct as: never ship a service with no tests and no monitoring.

---

## The one big idea: eval-first

We built the **measurement harness before improving retrieval** — the scoring
tools came first. We measured the plain, basic RAG setup first. Then hybrid,
reranking, and the agentic loop each had to *earn their place* by moving the
recorded score (the delta).

**Backend analogy:** shipping RAG without evals (quality measurements) is like
deploying a service with no tests and no monitoring. The golden set is your test
suite; faithfulness scoring (does the answer match its sources?) is your
production monitoring; the CI eval gate is your "tests must pass to merge" rule —
but for fuzzy outputs, where "correct" is a spread of scores, not a yes/no. The
whole project is that flip in order.

Why this paid off: we caught **two of our own upgrades not earning their keep**
(hybrid after reranking; the agentic loop outside vocab-mismatch questions). If
you just eyeball 5 questions by hand, you ship both as "wins."

---

## Phase 1 — Embeddings, vector indexes, chunking

**The problem.** An LLM knows nothing about your private or recent docs, and it
makes things up with confidence. RAG fixes this: fetch the relevant chunks, paste
them into the prompt, and cite them.

**Embeddings from first principles.** An embedding (turning text into a list of
numbers that captures its meaning) maps text to a point in 384-dimensional space,
so that *similar meaning → nearby points*. "How do I test FastAPI?" and "writing
FastAPI tests" land close together even with zero shared words. To measure how
close two points are, we take the cosine of the angle between the two vectors
(1.0 = same direction, 0 = at right angles, i.e. unrelated). We scale every
vector to length 1, so the cosine is just a dot product (multiply matching
numbers, add them up).

*Tiny 2-D example:* imagine "cat"→(0.9, 0.1), "kitten"→(0.86, 0.15),
"database"→(0.05, 0.99). cat·kitten ≈ 0.79 (close); cat·database ≈ 0.14 (far).
Scale that to 384 dims and you have semantic search.

**The vector index — your B-tree-vs-hash instinct, ported.** Comparing the query
against all 3,125 chunks one by one (an exact scan) is O(n) — the work grows with
the data. Fine at our size, hopeless at millions. pgvector gives you two
approximate indexes (fast shortcuts that are a little less exact):

- **IVFFlat** splits the vectors into lists (clusters) and searches only the few
  nearest lists. Like a **hash index**: fast lookups, needs a training step to
  learn the buckets, gets worse if the data shifts.
- **HNSW** builds a graph you can walk toward the query (a "navigable small-world
  graph"). Like a **B-tree**: no training, a better speed-vs-quality tradeoff,
  but more memory and a slower build.

We chose HNSW (using cosine). Honest note: at 3k chunks a plain exact scan would
be fine; HNSW is here to *show* the tradeoff. `hnsw.ef_search` (default 40) is the
"how hard do I look" knob at search time — higher = finds more, but slower.

**Chunking = your index granularity decision.** Chunking means splitting the docs
into pieces. Too-large chunks bury the answer in noise; too-small chunks lose the
surrounding context. We use fixed 400-token windows with 60-token overlap (the
overlap means an answer sitting on a boundary isn't cut in half). Each chunk
stores its **character offsets** (where it starts and ends in the source doc).
This matters for the golden set: labels point at a span in the doc, then get
mapped onto whatever chunk config is live, so re-chunking never quietly breaks
the labels.

**The delta.** This phase *is* the baseline — there's nothing to beat yet. We
deliberately dug up 3 failures (`docs/FAILURES.md`): a false "I don't know" on an
exact identifier (`hnsw.ef_search`), a vocabulary-mismatch miss (the async docs
never showed up for a "blocking" question), and an out-of-scope query that wasn't
filtered out. Each one became a target for a later phase.

---

## Phase 2 — Golden set + retrieval metrics

**The problem.** "It feels better" is not something you can ship. We need
numbers.

**The metrics (know these cold).**
- **recall@k** — of the chunks that truly answer the question, how many showed up
  in the top-k results? "Did we fetch the answer at all?"
- **MRR** (mean reciprocal rank) — 1/(rank of the first correct chunk). "How high
  did the first right answer land?" MRR 1.0 = always rank 1; 0.5 = rank 2.
- **nDCG@10** — a graded score: it rewards putting a *primary* chunk above a
  merely *supporting* one, and counts hits further down for less. This is why we
  tag labels primary/supporting instead of just yes/no — nDCG means nothing
  without those grades.

**Building a golden set honestly.** The golden set is our list of test questions
with the known-correct chunks marked. The worst mistake is writing a question by
rewording the chunk you already know is the answer — then the retriever just
matches wording no real user would type. Our defenses: (1) word the questions the
way an engineer really would (often using *different* words than the doc — that's
the `vocab-mismatch` type), (2) an automatic **n-gram leakage check** — how much
wording a question shares with its gold chunk (max observed 0.15, threshold
0.50). 53 questions, 5 types, ~15% negative controls (out-of-scope, no valid
answer — they measure "does it know when it doesn't know?").

**Backend analogy:** the golden set is a **fixture-based test suite** (fixed
inputs with known expected outputs) for a fuzzy function. The config-hash guard —
the runner refuses to score if the live chunk config doesn't match the config the
labels were built against — is the same instinct as a migration that fails shut
when a checksum drifts.

**The delta (the anchor for everything).** Dense baseline: **recall@10 0.540,
MRR 0.433, nDCG@10 0.395**. The same to the last digit on every run.

---

## Phase 3 — LLM-as-judge (generation quality)

**The problem.** Retrieval metrics don't tell you if the *answer* is any good.
But for free-form text there's no single correct string to `assert ==` against.

**The idea.** Use a second LLM as the grader (this is "LLM-as-judge") against a
written rubric (a fixed set of grading rules), scoring three axes 1–5:
**faithfulness** (are the claims backed by the fetched context?),
**groundedness** (does it stay inside that context and cite it?), **relevance**
(does it answer the question that was asked?). It's reference-free — the judge
sees only the same context the writer saw, with no gold answer, mirroring
production, where at serving time you have no correct answer to compare against.

**Why LLM judges are dangerous, and the fixes.**
- *Self-preference bias* (a model favors its own writing style) → use
  **different model families**: llama writes, qwen grades.
- *Verbosity bias* (longer looks better) → the rubric scores support and
  citation, not length.
- *The judge might just be wrong* → we **regression-test the judge itself**
  (re-run a fixed check to catch it drifting): before we trust its scores, it
  must rank a known-good answer clearly above a planted made-up answer and an
  off-topic one *by a margin*. (It did: 5/5/5 vs faithfulness=1 vs 1/1/1.)

**Backend analogy:** the judge is a flaky dependency, so you write a contract
test for it (a check that it behaves as promised) before you trust it in your
pipeline.

**The delta.** Naive dense RAG: **faithfulness 4.65, groundedness 4.59,
relevance 4.13, refusal accuracy 1.00**. Relevance sits *below* faithfulness —
because bad retrieval hands the model the *wrong* context, and it writes an answer
that is faithful to that. That gap is the opening for retrieval upgrades to lift
generation quality.

---

## Phase 4 — Hybrid retrieval (BM25 + RRF)

**The problem.** Dense retrieval blurs exact tokens (specific words or symbols).
`hnsw.ef_search`, `tools/list`, `<=>` — one 384-dim vector smears these into a
vague general area and misses the exact chunk.

**Lexical vs semantic.** BM25 (a classic keyword-scoring method, here run through
PostgreSQL full-text search) ranks by *literal word overlap* — perfect for exact
identifiers, useless for reworded phrases. Dense is the opposite: strong on
meaning, weak on exact words. **Fusion** (blending two result lists into one)
combines them.

**RRF = your composite-index instinct.** RRF (Reciprocal Rank Fusion) scores a
chunk by `Σ 1/(k + rank_in_each_list)` (k=60). It combines by **rank, not raw
score**, so the two retrievers' score scales — which aren't comparable — never
need lining up. It's like a composite index that lets two access paths both
contribute without converting their costs to a common unit. A chunk that both
retrievers rank highly wins.

**The delta — and the honest catch.** Hybrid raised recall@10 (0.540 → 0.563,
+4.2%) but *hurt* MRR (0.433 → 0.402): mixing in a weak sparse (keyword) ranker
pushes down dense's confident top-1 hits. It rescued 7 exact-identifier questions
dense missed entirely (MCP transports, GIN, WAL) but pushed down ~6 others.
**Verdict: hybrid earns its place by widening recall, not by improving precision**
(getting the very top result right) — which is exactly the setup reranking needs.
We did *not* tune RRF against the golden set (that would be overfitting to the
eval).

---

## Phase 5 — Cross-encoder reranking

**The problem.** Hybrid widened the pool but scrambled precision (the order of
the top few). Get it back.

**Bi-encoder vs cross-encoder (the key distinction).** The bi-encoder (bge, the
model used for dense retrieval) embeds the query and each passage **separately** —
fast, and you can index and precompute them, but it never sees the two
*together*. A **cross-encoder** feeds `(query, passage)` through the model **as a
pair** and scores that pair, so it can weigh how the words interact — which the
bi-encoder can't. It's far more accurate, but it costs O(candidates) model calls
per query with **nothing precomputed** — you can't index it.

So the standard split: bi-encoder + BM25 cheaply pull a wide pool (top-50), then
the cross-encoder carefully re-sorts (reranks) it down to the top-k. A fast
filter, then an expensive precise sort — the same shape as a cheap index scan
feeding an expensive recheck.

**The delta — the big win.** rerank vs dense baseline: **recall@5 0.412 → 0.573
(+39%), nDCG@10 +16%, MRR +10%.** Cost: retrieve latency **14 ms → ~510 ms** (50
cross-encoder runs per query). The precision-for-latency trade, put in numbers.

**The second honest finding.** `dense+rerank` **ties or beats** `hybrid+rerank`
(recall@10 0.653 vs 0.627). The exact-identifier cases hybrid rescued were already
sitting in dense's top-50 pool — reranking lifts them up **without BM25**. So on
this corpus (our document collection), **hybrid does not earn its place once
reranking exists.** We kept it in the table anyway. On a corpus full of code,
logs, or exact terms, BM25 would likely still pay.

---

## Phase 6 — Agentic loop

**The problem.** Some queries fail no matter how good the retriever, because the
user's words and the doc's words just don't overlap (a vocabulary mismatch).

**The moves.** Sort the query into a type (factoid/how-to/multi-hop/out-of-scope)
→ if it's multi-hop (needs several facts joined), **decompose** it into
sub-questions → retrieve + rerank → **self-critique** the confidence with a cheap
rule of thumb (heuristic: is the cross-encoder top score ≥ 2.0?) → if weak, do a
**HyDE re-query**: ask the LLM to *write a made-up ideal answer* in the docs' own
vocabulary and retrieve with *that*, which closes the vocab gap. Bounded by an
**iteration cap (2)** and a **token budget (6000)**.

**Backend analogy:** this is a retry loop with a circuit breaker. The cap and
budget are your max-retries and timeout — they're what stop it looping forever
(verified by `test_loop_terminates_at_cap`). HyDE is a fallback path it takes only
when the main one looks unhealthy.

**The delta — precisely targeted.** Overall +3% vs rerank, but that average hides
the real result: the entire gain lands on **vocab-mismatch (+12.5% recall@10,
0.417 → 0.542); every other type moved +0.000.** It fixed q064 — the exact
async-vs-blocking failure from Phase 1 — 0 → 1.0. Cost: 219 tokens/query, p95
latency (95% of requests are faster than this) 8.2 s on re-queried questions.

**When does the loop pay for itself?** When there's a real vocabulary gap — here,
~17% of queries. For the other 83% the classifier is pure overhead. The redesign
we wrote down: drop the always-on classifier, and trigger the LLM spend on the
confidence heuristic alone → ~5× fewer tokens for the same gain. Knowing *when
not* to be agentic is the senior insight.

---

## Phase 7 — Observability, cost, semantic cache

**Semantic cache = a cache keyed on meaning.** A normal cache keys on the exact
query string, so "how to test FastAPI?" and "writing FastAPI tests" miss each
other. A semantic cache keys on the query **embedding** (its meaning-vector)
instead: a near-duplicate (cosine ≥ threshold) is a hit and skips retrieve+generate
— the expensive stages. Redis 8 does the vector search.

**The threshold is a precision/recall dial, and precision wins.** Measured
bge-small similarities: close rewordings ~0.90–0.92, but a *different* question
("HNSW index" vs "partial index") scored 0.74, and heavy rewordings 0.66–0.80 —
overlapping the different-question range. There's no clean cutoff. We set **0.90
(precision-first)**: a cache that serves the *wrong* saved answer is worse than
no cache at all. Result: exact-repeat hit 1.00, close-paraphrase 0.44, **novel
false-hit 0.00**.

**Cache invalidation = the honesty problem.** Each entry's key is namespaced
(prefixed) by `corpus_version : chunk_config_hash : retrieval_mode`. Re-ingest the
docs or change the pipeline → new prefix → old entries simply miss. No stale
answers, ever. Plus a TTL (an expiry time). (The two hard things in CS are cache
invalidation and naming; we made invalidation part of how the key is built, so it
can't rot.)

**Cost as a first-class metric.** Everything runs locally (zero real spend), so
we price LLM tokens at public rates — **shadow-dollars** — on every `/query`
response. "Free" hides the real cost of a design choice; shadow-dollars put
reranking's latency and the agentic loop's tokens on the same scale so you can
compare them.

**Observability.** Observability means being able to see what the system did
inside. Per stage (cache/retrieve/generate) we record an OTel span (a timed trace
segment) carrying token + cost + latency attributes, exported to Jaeger — the same
wiring as agent-gateway. Per-stage p95: retrieve 25 ms (dense) → 653 ms (rerank)
→ 8.2 s (agentic); generation dominates end-to-end.

---

## Phase 8 — CI eval gates

**The idea.** When someone opens a PR, CI runs the eval suite; a drop in quality
(a regression) **blocks the merge**, exactly like a failing unit test — but for
retrieval quality. The pass/fail thresholds live in `eval/thresholds.yaml`
(config-as-data — the rules are data, not hard-coded). Retrieval runs fully in CI
(bge-small on CPU, committed corpus, pgvector container). Generation can't (no
Ollama in CI), so instead we check that the newest **committed, hash-stamped**
generation result is fresh (its `golden_hash` still matches the repo) and above
the floors (minimum scores) — stale or regressed → blocked. Same enforcement,
zero cloud LLM.

**The demonstration (PR #1).** A PR that removes the bge query-instruction prefix
**passes `lint` and `test`** but the `retrieval-gate` catches it —
`GATE FAIL: recall@10 0.489 < floor 0.5` — and blocks the merge. That's the
whole point: unit tests check that the code is correct; only the golden set
catches a drop in *quality*.

**How this changes team velocity.** Without the gate, a well-meaning "small
retrieval tweak" quietly makes quality worse and nobody notices until users
complain. With it, the regression is caught in CI within minutes, with a number
attached. It turns "please be careful with retrieval changes" (which you can't
enforce) into a mechanical gate — the same shift as going from "please write
tests" to "coverage must not drop."

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
faithfulness was never the problem (llama grounds its answers well), *relevance*
was, because naive retrieval fed it the wrong context. Fix the retrieval and
relevance follows. Both pipelines refuse 100% of out-of-scope questions.

**Cache:** overall hit-rate 0.63, exact-repeat 1.00, novel false-hit 0.00.

## The five interview narratives

1. **Why eval-first?** It caught two of my own upgrades that weren't earning
   their keep. Show the table.
2. **Why did hybrid beat dense — then stop mattering?** BM25 rescued exact
   identifiers, but reranking already pulled them back from the pool on its own.
3. **When is reranking worth it?** +16% nDCG for ~500 ms — yes when quality
   matters; when latency matters, run it only when confidence is low.
4. **When does the agentic loop pay for itself?** Only on vocabulary mismatch
   (+12.5%); a cap plus a token budget stop it looping forever.
5. **How do CI eval gates change how a team ships RAG?** They turn "be careful"
   into a mechanical, numeric merge gate — PR #1 proves it.
