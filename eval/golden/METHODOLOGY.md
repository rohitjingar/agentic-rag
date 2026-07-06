# Golden set — methodology

The golden set is the measuring stick for this whole project. If it is biased,
leaky, or unverifiable, every downstream number (hybrid delta, rerank delta,
agentic delta) is worthless. This document is how it was built and why it can
be trusted.

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

- **`locator`** is a *verbatim substring of the source document* where the
  answer lives — **not** a chunk id. Labels anchor to document content, so
  re-chunking (different size/overlap) never invalidates them; it just
  re-materializes to new chunk ids. `eval/materialize.py` resolves each locator
  to the chunk(s) containing it for the live chunk config; a locator that
  matches zero chunks is a hard error, never a silent zero-score.
- **`grade`**: `primary` = the chunk directly answers the question;
  `supporting` = needed context but not the answer itself. This graded labeling
  is what makes nDCG@10 meaningful (it rewards ranking primary above
  supporting); recall@k and MRR are computed on primaries only.

## Size and composition

53 questions total: **46 answerable + 7 negative controls** (~13%).

Answerable questions span the four corpus sources (pgvector, PostgreSQL,
FastAPI, MCP) and five deliberately-chosen types:

| qtype | what it stresses |
|---|---|
| `factoid` | a single specific fact (default value, method name) |
| `how-to` | a procedure spread over a section |
| `multi-hop` | needs two chunks/docs combined |
| `vocab-mismatch` | question wording differs from doc wording — the case dense retrieval and BM25 handle differently |
| `negative-control` | **no** answer in the corpus (Django, Kubernetes, Kafka, …) |

Negative controls have no gold chunk; they don't enter retrieval metrics.
They exist to measure the thing amateur RAG never measures: **does the system
know when it doesn't know?** (refusal correctness + retrieval confidence,
scored by the Phase 3 judge and exercised by the Phase 6 self-critique.)

## Anti-leakage (the honesty guarantee)

The cardinal sin of RAG evals is questions written by paraphrasing the chunk
you already know is the answer — the retriever then matches on wording the user
would never have used. Two mechanisms prevent it here:

1. **Authoring rule:** questions are phrased the way a backend engineer would
   ask (often with *different* vocabulary than the doc — that's the point of
   the `vocab-mismatch` type), never by copying doc sentences.
2. **Mechanical check:** `eval/materialize.py` computes the 5-gram overlap
   between each question and its gold chunk text; the runner refuses to
   evaluate if any question exceeds 0.50. Current max across the set: **0.154**.

## Verification & provenance

`scripts/verify_golden.py` (also a CI gate) asserts, for every answerable item:
its locator exists verbatim in the source document, it materializes to at least
one chunk under the live config, and leakage is under threshold. Every result
JSON records `corpus_version` + `chunk_config_hash`, and the runner won't score
against a config whose chunks the labels don't materialize into.

## Authorship & sign-off

Per the build protocol, the set was authored to the committed corpus with every
answer grep-verified against the source docs. It is presented to Rohit for
review/sign-off; until he signs off, recorded baselines are marked
`provisional: true`. This preserves the protocol's validity gate without
blocking the build.
