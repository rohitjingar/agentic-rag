# Retrieval eval results

Appended by `eval/run.py`. Each row = one retriever config on the frozen
golden set. Deltas are read down the table (baseline is the anchor).

| label | mode | recall@5 | recall@10 | MRR | nDCG@10 | retrieve p50/p95 ms | n | provisional |
|---|---|---|---|---|---|---|---|---|
| dense-baseline | dense | 0.4121 | 0.5401 | 0.4328 | 0.3954 | 13.8/25.1 | 46 | yes |
