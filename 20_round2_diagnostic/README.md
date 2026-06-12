# Step 20 — Training dynamics diagnostic

Scores per-epoch checkpoints on benchmark and CIViC ranking to test within-model training effects.

Method: Pairwise comparison from epoch 1 to best validation-F1 checkpoint, split by gene-drug and gene-disease.

Results: 498 epoch checkpoints; 65 pairable seeds. Pooled hard-subset KB delta -0.0016; gene-disease -0.0569 (48/65 fall); gene-drug +0.0080. Verdict: mixed_gene_disease_signal.
