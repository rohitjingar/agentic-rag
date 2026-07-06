# Retrieval eval results

Appended by `eval/run.py`. Each row = one retriever config on the frozen
golden set. Deltas are read down the table (baseline is the anchor).

| label | mode | recall@5 | recall@10 | MRR | nDCG@10 | retrieve p50/p95 ms | n | provisional |
|---|---|---|---|---|---|---|---|---|
| dense-baseline | dense | 0.4121 | 0.5401 | 0.4328 | 0.3954 | 13.8/25.1 | 46 | yes |
| sparse-bm25 | sparse | 0.2885 | 0.3318 | 0.2593 | 0.2374 | 9.5/22.7 | 46 | yes |
| hybrid-rrf | hybrid | 0.4271 | 0.5629 | 0.4024 | 0.3887 | 35.9/60.3 | 46 | yes |
