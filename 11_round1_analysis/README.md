# Step 11 — Round-one encoder comparison

Compares nine encoders on in-distribution benchmark and out-of-distribution CIViC ranking at a single checkpoint.

Method: Score seventy-two fine-tuned runs plus nine untrained-floor references on both axes; variance decomposition, association bootstrap, and calibration diagnostics.

Results: Benchmark spread 0.025; variance shares 36/64 benchmark, 23/77 gene-drug KB, 13/87 gene-disease KB. Fine-tuned KB means 0.676 gene-drug and 0.625 gene-disease. Seed-level benchmark–KB association negative on both pair types.
