# Step 04 — Pilot study

Minimal-training pilot on three encoders with step-03 pool scoring under the pre-fix pipeline.

Method: Short training run on BioRED plus DrugProt; score frozen pool at best checkpoint.

Results: PubMedBERT MRR 0.469 versus random 0.322 and distance ranker 0.489; reference benchmark F1 0.893. Hard-subset ranking beats the distance ranker. Not comparable to post-fix step-10 matrix.
