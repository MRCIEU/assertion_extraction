# Step 20 — Training dynamics diagnostic

Scores per-epoch checkpoints on benchmark and CIViC ranking to test within-model training effects.

Method: Pairwise comparison from epoch 1 to best validation-F1 checkpoint, split by gene-drug and gene-disease; mundane-explanation and qualitative deepening.

| Metric | Value |
| --- | ---: |
| Epoch checkpoints | 498 |
| Pairable seeds | 65 |
| Pooled hard KB delta | -0.0016 |
| Gene-drug KB delta | +0.0080 |
| Gene-disease KB delta | -0.0569 (48/65 fall) |
| Gene-disease-hard bootstrap P(negative) | 99.1% |

Pushing the in-distribution benchmark erodes out-of-distribution gene-disease ranking for biomedically pretrained encoders in a regular, predictable pattern; gene-drug stays flat or positive.
