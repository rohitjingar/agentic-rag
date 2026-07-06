# Golden set — methodology

The golden set — our hand-labeled test set, like a unit-test fixture — is the
measuring stick for this whole project. If it is biased, leaky, or can't be
verified, every number that follows (hybrid delta, rerank delta, agentic delta)
is worthless. This document explains how it was built and why you can trust it.

## What a golden item is

```json
{
  "qid": "q001",
  "question": "In pgvector, what is the default size of the dynamic candidate list ...",
  "expected_answer": "40. The search-time dynamic candidate list hnsw.ef_search defaults to 40.",
  "qtype": "factoid",
  "answerable": true,
  "gold": [{"doc_id": "pgvector/README", "locator": "40 by default", "grade": "primary"}]
}
```

- **`locator`** is an exact run of text copied straight from the source
  document — the spot where the answer lives — **not** a chunk id. (A chunk is
  one small slice of a document that the retriever searches.) Because labels
  point at document text instead of chunks, re-chunking the docs (a different
  size or overlap) never breaks them. The label just re-materializes — gets
  re-linked to whatever new chunk ids now hold that text. `eval/materialize.py`
  takes each locator and finds the chunk(s) that contain it for the current
  chunk config. A locator that matches zero chunks is a hard error that stops
  the run, never a silent zero-score.
- **`grade`** says how useful a chunk is (its primary vs supporting label).
  `primary` = the chunk directly answers the question; `supporting` = needed
  context, but not the answer itself. This graded labeling is what makes
  nDCG@10 meaningful — nDCG is a ranking score that rewards putting primary
  chunks above supporting ones. recall@k (did a right chunk land in the top k?)
  and MRR (mean reciprocal rank — how high the first right chunk sits) are
  computed on primaries only.

## Size and composition

53 questions total: **46 answerable + 7 negative controls** (~13%). A negative
control is a question the corpus can't answer.

The answerable questions cover the four corpus sources (pgvector, PostgreSQL,
FastAPI, MCP) — the corpus is the collection of documents we search — and five
types we chose on purpose:

| qtype | what it stresses |
|---|---|
| `factoid` | a single specific fact (default value, method name) |
| `how-to` | a procedure spread over a section |
| `multi-hop` | needs two chunks/docs combined |
| `vocab-mismatch` | question wording differs from doc wording — the case dense retrieval and BM25 handle differently |
| `negative-control` | **no** answer in the corpus (Django, Kubernetes, Kafka, …) |

Negative controls have no gold chunk, so they don't count toward retrieval
metrics. They are there to measure the one thing amateur RAG never checks:
**does the system know when it doesn't know?** (That is: does it correctly
refuse, and how confident was retrieval — scored by the Phase 3 judge and
exercised by the Phase 6 self-critique.)

## Anti-leakage (the honesty guarantee)

The worst mistake in RAG evals is writing a question by rewording the chunk you
already know is the answer. This is leakage — the answer's wording leaks into
the question, so the retriever wins by matching words a real user would never
type. Two things stop that here:

1. **Authoring rule:** we phrase each question the way a backend engineer would
   ask it — often with *different* words than the doc (that's the whole point
   of the `vocab-mismatch` type) — never by copying doc sentences.
2. **Mechanical check:** `eval/materialize.py` measures the 5-gram overlap
   between each question and its gold chunk text — how much of the 5-word runs
   of text they share. The runner refuses to evaluate if any question goes over
   0.50. Current max across the set: **0.154**.

## Verification & provenance

`scripts/verify_golden.py` (also a CI gate — a check that must pass in CI)
confirms, for every answerable item: its locator appears word-for-word in the
source document, it materializes to at least one chunk under the current
config, and its leakage is under the threshold. Every result JSON records
`corpus_version` + `chunk_config_hash` — a short fingerprint of the exact chunk
settings — and the runner won't score against a config whose chunks the labels
can't materialize into.

## Authorship & sign-off

Following the build protocol, the set was written against the committed corpus,
with every answer grep-verified against the source docs. It now goes to Rohit
for review/sign-off; until he signs off, recorded baselines are marked
`provisional: true`. This keeps the protocol's validity gate in place without
blocking the build.
