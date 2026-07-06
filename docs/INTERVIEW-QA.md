# Interview Q&A — agentic-rag

The likely interview questions about this project, with plain-English answers.
This is your script: read the **What to say** for each, and memorize the
**one-liner**. Real numbers are in `eval/results/RESULTS.md` and `ANALYSIS.md`.

---

## 1. What is this project, in one minute?

**What to say:** It's a question-answering system over a set of documents
(RAG — fetch the relevant pages, then have an AI write an answer from them). But
the real point isn't the pipeline — it's that I built the **measurement system
first** and made every improvement *prove* it helped with a number. So I can say
"recall went from 0.54 to 0.65," not "it feels better." Along the way I found
that two of my own upgrades didn't actually help, and I kept them in the results
table honestly.

**One-liner:** "A RAG system built eval-first — every upgrade had to earn its
place with a measured number, and I was honest when some didn't."

---

## 2. Why build the evaluation FIRST, before improving the search?

**What to say:** Because without a scorecard you're flying blind. The amateur
workflow is: build a fancy pipeline, try 5 questions by hand, it looks good,
ship it. Then it quietly breaks on real traffic and nobody notices, because
nothing is measured. Building the test set first means every change after it is
judged against the same fixed yardstick, so I can tell a real improvement from a
lucky-looking one. It's the same reason you don't ship a service with no tests
and no monitoring.

**What goes wrong the other way:** you can't tell which of your five upgrades
actually helped, you can't catch a change that quietly makes answers worse, and
you argue from opinions instead of numbers.

**One-liner:** "You can't improve what you can't measure — so I measured first."

---

## 3. What's in the "golden set," and how do you keep it honest?

**What to say:** It's my test set — 53 real questions, each tagged with the
document text that actually answers it (like unit-test fixtures, but for a fuzzy
system). 46 are answerable; 7 are "negative controls" — questions the documents
*can't* answer, to check the system correctly says "I don't know."

The big risk is **leakage**: if I write a question by rewording the exact
sentence I know is the answer, the search wins by matching words a real user
would never type — so the score is fake. I guard against it two ways: (1) I
phrase questions the way an engineer really would, often with *different* words
than the docs; (2) an automatic check measures word overlap between each
question and its answer chunk and refuses to run if it's too high. Max overlap
in my set is 0.15, well under the 0.50 limit.

**One-liner:** "Questions are tagged to the real answer text, and an automatic
leakage check stops me from accidentally writing the answer into the question."

---

## 4. What do recall@k, MRR, and nDCG measure — in plain terms?

**What to say:**
- **recall@k** — of the pages that truly answer the question, how many showed up
  in the top k? ("Did we even find the answer?")
- **MRR** (mean reciprocal rank) — how high did the *first* correct page land? 1
  means always rank 1; 0.5 means rank 2 on average. ("How near the top?")
- **nDCG** — a ranking score that rewards putting the *most* useful page above a
  merely-related one. It needs graded labels (I mark chunks "primary" = directly
  answers vs "supporting" = context), which is why I graded them.

**One-liner:** "recall = did we find it; MRR = how high; nDCG = did we rank the
best one on top."

---

## 5. You use an AI to grade an AI's answers. Why trust it?

**What to say:** That's the LLM-as-judge — a second model scores each answer 1–5
on faithfulness (did it stick to the sources / not make things up),
groundedness (did it stay in the sources and cite them), and relevance (did it
actually answer). AI judges have known biases, so I handled them: I use a
**different model family** for the judge than the generator (so it doesn't just
love its own style), I score against support/citations rather than length (so it
doesn't reward long waffle), and temperature 0 so scores are repeatable.

Most importantly, I **tested the judge itself** — I fed it a known-good answer, a
planted false answer, and an off-topic answer, and required it to separate them
by a clear margin before I trusted its scores. It's a contract test for a flaky
dependency.

**One-liner:** "Cross-family judge, scored on evidence not length, and I
regression-tested the judge before trusting it."

---

## 6. Why add keyword search (hybrid) when you already have meaning-search?

**What to say:** Meaning-search (dense) is great at "same idea, different words,"
but it *blurs exact terms*. Ask for `hnsw.ef_search` and it smears that into a
vague "vector settings" direction and grabs the wrong chunk. Keyword search
(BM25) matches the literal token, so it nails exact identifiers like
`ef_search`, `tools/list`, `GIN`. I combine the two ranked lists with Reciprocal
Rank Fusion — which merges by *rank position*, not score, so I never have to
calibrate the two systems' incompatible score scales.

**One-liner:** "Meaning-search blurs exact terms; keyword search nails them; I
fuse them by rank so the scales don't need calibrating."

---

## 7. Reranking — what is it, why is it the biggest win, and what's the cost?

**What to say:** The fast search grabs ~50 rough candidates. A **reranker** is a
slower, smarter model that reads the question *together with* each candidate and
re-sorts them, floating the best to the top. The key distinction: the fast
search uses a **bi-encoder** — it turns query and page into fingerprints
*separately*, so it never sees them together; the reranker is a **cross-encoder**
— it looks at the pair *jointly*, which is far more accurate but can't be
pre-computed. So the standard split: cheap wide search first, expensive precise
sort second — like an index scan feeding an exact recheck.

It was my biggest single win: recall@5 went from 0.41 to 0.57 (+39%), quality
score nDCG +16%. The cost is latency: retrieval went from ~14 ms to ~510 ms,
because it runs 50 model comparisons per query with no shortcut. That's the
precision-for-speed trade, measured.

**One-liner:** "Bi-encoder finds 50 fast; cross-encoder re-sorts them precisely.
+39% recall@5 for ~500 ms — worth it for quality, gate it for latency."

---

## 8. (The killer question) You added hybrid search — did it earn its place?

**What to say:** On its own, yes — it improved recall. But once I added
reranking, I measured the *full* pipeline and found **dense + rerank ties
hybrid + rerank** (recall@10 0.653 vs 0.627). The exact-term cases keyword search
rescued were already sitting in the dense top-50 pool — just ranked low — and the
reranker pulls them up *without* needing keyword search. So on this data, the
keyword-search complexity earned ~nothing once reranking was in. I kept the
number in the table anyway rather than hide it.

That willingness to measure my own addition *out* of the pipeline is the whole
point of eval-first. (On a code/logs corpus full of exact terms, keyword search
would likely still pay — so I'd keep it behind a flag.)

**One-liner:** "Hybrid helped alone, but reranking made it redundant on this
corpus — I measured my own feature out and kept the honest number."

---

## 9. The agentic loop — when does it pay for itself, and what stops it looping forever?

**What to say:** The agentic mode adds self-checking: after searching, it asks
"do these results look confident?" If not, it rewrites the question in the docs'
own vocabulary (HyDE — write a fake ideal answer, search with that) and retries.

The honest result: overall it only helped ~3%, but that average hides the truth
— the entire gain was on **vocabulary-mismatch questions** (+12.5%), and **zero**
on every other type. It fixed the exact "my API is blocking / the docs say async"
failure from early on. So it pays for itself only when there's a real
vocabulary gap — about 17% of my queries. For the rest it just burned tokens
(~219/query), because a classifier runs on *every* query while only a few
benefit. The senior insight: I'd redesign it to spend tokens only when the
confidence check fires.

**What stops it looping forever:** a hard iteration cap (2 tries) and a per-query
token budget — verified by a test that forces perpetual low confidence and
confirms it still stops.

**One-liner:** "It only helps vocabulary-mismatch questions (+12.5%, 0%
elsewhere); a 2-try cap and token budget guarantee it terminates."

---

## 10. The semantic cache — why the 0.90 threshold, and what's the danger?

**What to say:** It caches answers by *meaning*, not exact text, so a reworded
repeat of an earlier question hits and skips the expensive work. The threshold
is how similar two questions must be to count as "the same." I tuned it to 0.90
from real measurements: close paraphrases scored ~0.90–0.92, but a genuinely
*different* question scored 0.74 — and there's no clean gap. So I chose
precision over recall: **a cache that serves the wrong answer to a different
question is worse than no cache.** Result: it never served a wrong answer (0%
false hits), at the cost of a modest 44% paraphrase hit-rate. Invalidation is
honest — cache keys include the corpus version and pipeline config, so a
re-index or config change simply misses instead of serving something stale.

**One-liner:** "Keyed on meaning; tuned to 0.90 for zero false hits, because a
wrong cached answer is worse than a miss; keys include corpus+config so it can't
go stale."

---

## 11. How do CI eval gates change how a team ships?

**What to say:** Normally a small tweak to the search can quietly make answers
worse and slip through, because regular tests only check the code *runs*, not
that quality held. The eval gate runs the test set on every pull request and
**blocks the merge if quality drops below a floor**. It turns "please be careful
with retrieval changes" — which is unenforceable — into a mechanical, numeric
gate, the same way coverage-can't-drop turns "please write tests" into a rule.

I proved it live: a pull request that secretly degrades the search **passes the
normal code tests but the eval gate catches it** — `GATE FAIL: recall@10 0.489 <
floor 0.5` — and blocks it. (No cloud AI needed in CI: the search test runs
there directly, and the answer-quality test is verified against a committed,
hash-stamped result so a stale one is rejected.)

**One-liner:** "It blocks merges on a quality drop — turning 'be careful' into a
mechanical gate. I demoed it: a bad PR passed code tests but the gate blocked it."

---

## 12. Bonus: what did the live demo teach about where RAG breaks?

**What to say:** Answering has two halves that can each fail on their own:
*finding* the right page (retrieval) and the AI *reading* it correctly
(generation). In the demo, reranking successfully found the page that said
"ef_search is 40 by default" — retrieval was fixed — but the small free local
model still fumbled and refused to answer. That's exactly why I score the two
halves **separately**: a retrieval metric would call that a win, while the
answer-quality judge flags the bad answer. A bigger model fixes the second half;
the point is you must measure them apart to know which one to fix.

**One-liner:** "Retrieval and generation fail independently — reranking found the
answer, the small model still fumbled it — so I measure them separately."

---

## The 5 points that are trickiest to nail (drill these)

1. **The honest hybrid finding (Q8)** — "I measured my own feature out of the
   pipeline." This is your strongest signal of senior judgment; practice saying
   it confidently.
2. **Bi-encoder vs cross-encoder (Q7)** — separate fingerprints vs looking at the
   pair jointly. Know *why* one is fast-and-indexable and the other is
   accurate-but-not.
3. **When the agentic loop pays off (Q9)** — the "+12.5% on one type, 0%
   elsewhere, and here's how I'd make it cheaper" story.
4. **Why the cache threshold is precision-first (Q10)** — "a wrong cached answer
   is worse than a miss."
5. **Retrieval vs generation split (Q12)** — two independent failure points is
   *why* the eval harness has two separate scorers.

Read `docs/LEARNING.md` for the deeper version of any of these.
