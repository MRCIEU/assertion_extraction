# Step 10 — Recipe sweep and training matrix

Selects a stable recipe and trains nine encoders by eight seeds.

Method: Learning-rate and warmup sweep with DeBERTa health gate; full matrix at confirmed recipe with per-epoch checkpoints.

Results: DeBERTa gate failure at 3e-5/warmup; confirmed recipe 5e-6/none. Matrix benchmark F1 spread 0.025 across nine encoders (means roughly 0.72 to 0.75).
