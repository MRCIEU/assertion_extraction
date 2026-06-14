# Step 11 — Round-one encoder comparison

Compares nine encoders on in-distribution benchmark and out-of-distribution CIViC ranking at a single checkpoint.

Method: Score seventy-two fine-tuned runs plus nine untrained-floor references on both axes; variance decomposition and seed-level association bootstrap.

| Axis | Between / within encoder variance | Fine-tuned mean |
| --- | --- | ---: |
| Benchmark F1 | 36% / 64% | spread 0.025 |
| KB MRR gene-drug | 23% / 77% | 0.676 |
| KB MRR gene-disease | 13% / 87% | 0.625 |

Seed-level benchmark–KB Spearman: gene-drug negative; gene-disease negative (interval-heavy).
