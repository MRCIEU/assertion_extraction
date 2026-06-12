# Step 10 — Recipe sweep and training matrix

Selects a stable recipe and trains nine encoders by eight seeds at benchmark-only monitoring; saves per-epoch checkpoints for step 20.

**Result:** DeBERTa gate failure at 3e-5/warmup; confirmed recipe 5e-6/none. Matrix mean benchmark F1 spread ~0.72–0.75 across nine encoders. KB scoring happens in folders 11 and 20.
