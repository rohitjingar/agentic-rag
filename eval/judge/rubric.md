# Generation eval rubric (v1)

The judge scores a `(question, retrieved context, answer)` triple on three
independent 1–5 dimensions. Scoring is **reference-free**: the judge sees the
same context the generator saw and judges the answer against *that context*,
not against a gold answer. This mirrors production — you never have the gold
answer at serving time, only the retrieved context.

The judge model (`qwen2.5:7b-instruct`) is a **different family** from the
generator (`llama3.1:8b`) to blunt self-preference bias (a model rating its own
style highly). temperature = 0, decoding constrained to a JSON schema.

## Dimensions

### Faithfulness — are the answer's claims supported by the context?
The anti-hallucination score. Every factual claim in the answer must be
entailed by the retrieved context.

| score | meaning |
|---|---|
| 5 | Every claim is directly supported by the context. |
| 4 | All claims supported; one minor detail is a reasonable paraphrase/inference. |
| 3 | Mostly supported, but one claim goes beyond the context. |
| 2 | A significant claim is unsupported or contradicts the context. |
| 1 | Largely fabricated / contradicts the context. |

A correct refusal ("the documentation does not contain the answer") when the
context truly lacks the answer is **faithful (5)** — it invents nothing.

### Groundedness — does the answer stay within the context and attribute?
Penalizes using outside knowledge (even if correct) and missing citations.
Faithfulness asks "are the claims true to the context?"; groundedness asks
"did the answer restrict itself to the context and cite it?"

| score | meaning |
|---|---|
| 5 | Fully derived from context; claims carry citations ([S#]). |
| 4 | Derived from context; citations mostly present. |
| 3 | Mostly grounded but leans on some unstated outside knowledge, or citations absent. |
| 2 | Substantial reliance on outside knowledge not in the context. |
| 1 | Ignores the provided context. |

### Answer-relevance — does the answer address the question?
Independent of correctness: a fluent answer to the wrong question scores low.

| score | meaning |
|---|---|
| 5 | Directly and completely answers the question asked. |
| 4 | Answers the question with minor omissions or padding. |
| 3 | Partially answers; misses part of the question. |
| 2 | Tangentially related; does not really answer. |
| 1 | Off-topic. |

For a **negative-control** question (no answer in the corpus), a refusal is the
correct behavior and scores 5 on relevance (it correctly addresses that the
answer isn't available); a confident fabricated answer scores 1.

## Known LLM-judge biases and our mitigations
- **Self-preference** → cross-family judge (qwen judges llama).
- **Verbosity bias** (longer = better) → the rubric scores support/attribution,
  not length; reasons must cite specific claims.
- **Position bias** → single-answer absolute scoring, not pairwise A/B here.
- **Judge unreliability** → the judge is itself regression-tested
  (`tests/test_judge_regression.py`): it must separate a known-good answer from
  a seeded-hallucination and an off-topic answer by a margin before its scores
  are trusted. If it can't, the rubric/prompt is iterated (or the larger local
  model used) — recorded honestly.
