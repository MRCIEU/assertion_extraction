# Round 2 diagnostic

Read-only analysis on Round 1 to judge whether a training-configuration Round 2 is worth compute.

## Key findings (fill after run)

Checkpoint policy on the main matrix: **val_f1-best weights only** (one recoverable epoch per run). Matched sweep (lr 2e-5, no warmup, seed 42) retains **val_loss-best and val_f1-best** weights for PubMedBERT, RoBERTa, DistilBERT.
