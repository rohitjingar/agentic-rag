# Generation eval rubric (v1)

The judge — a second AI model that scores each answer — looks at a
`(question, retrieved context, answer)` triple. It gives three separate 1–5
scores, one per dimension. Scoring is **reference-free** (no gold/correct
answer to compare against): the judge sees the same context the generator saw,
and it scores the answer against *that context*, not against a gold answer.
This matches production — at serving time you never have the gold answer, only
the retrieved context.

The judge model (`qwen2.5:7b-instruct`) comes from a **different family** than
the generator (`llama3.1:8b`). This blunts self-preference bias — a model
rating its own style highly. We set temperature = 0 (no randomness, so the
scores repeat) and force the output to follow a fixed JSON schema.

## Dimensions

### Faithfulness — are the answer's claims supported by the context?
This is the anti-hallucination score — it checks the answer isn't made up.
Every factual claim in the answer must be backed by the retrieved context.

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
This docks points for using outside knowledge (even when it's correct) and for
missing citations. Faithfulness asks "are the claims true to the context?";
groundedness asks "did the answer stay within the context and cite it?"

| score | meaning |
|---|---|
| 5 | Fully derived from context; claims carry citations ([S#]). |
| 4 | Derived from context; citations mostly present. |
| 3 | Mostly grounded but leans on some unstated outside knowledge, or citations absent. |
| 2 | Substantial reliance on outside knowledge not in the context. |
| 1 | Ignores the provided context. |

### Answer-relevance — does the answer address the question?
This ignores whether the answer is correct — a fluent answer to the wrong
question still scores low.

| score | meaning |
|---|---|
| 5 | Directly and completely answers the question asked. |
| 4 | Answers the question with minor omissions or padding. |
| 3 | Partially answers; misses part of the question. |
| 2 | Tangentially related; does not really answer. |
| 1 | Off-topic. |

For a **negative-control** question (one with no answer in the corpus, used as a
trap), a refusal is the correct behavior and scores 5 on relevance (it correctly
addresses that the answer isn't available); a confident fabricated answer scores
1.

## Known LLM-judge biases and our mitigations
- **Self-preference** → cross-family judge, a judge from a different model
  family (qwen judges llama).
- **Verbosity bias** (longer = better) → the rubric scores support/attribution,
  not length; reasons must cite specific claims.
- **Position bias** (answer order swaying the score) → single-answer absolute
  scoring: we score one answer on its own, not pairwise A/B here.
- **Judge unreliability** → the judge is itself regression-tested, tested like
  code (`tests/test_judge_regression.py`): it must separate a known-good answer
  from a seeded-hallucination (a planted false claim) and an off-topic answer by
  a margin before its scores are trusted. If it can't, the rubric/prompt is
  iterated (or the larger local model used) — recorded honestly.
