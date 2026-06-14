# Step 04 — Pilot study

Minimal-training pilot on three encoders with step-03 pool scoring under the pre-fix pipeline.

Method: Short training run on BioRED plus DrugProt; score frozen pool at best checkpoint.

| Model / baseline | MRR |
| --- | ---: |
| Random | 0.322 |
| Distance ranker | 0.489 |
| PubMedBERT-base (pilot) | 0.469 |
| Reference benchmark F1 (PubMedBERT) | 0.893 |

Not comparable to post-fix step-10 matrix.
