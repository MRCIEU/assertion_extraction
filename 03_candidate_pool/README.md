# Step 03 — Candidate pool

Builds PubTator3 candidate pools for the 1812 frozen targets.

Method: Per-abstract pool construction with frozen matching rules; trivial ranking baselines on the primary pool.

Results: 18911 primary candidates; 1590 matched and 222 missed recall (87.7%). Distance ranker MRR 0.489 versus random 0.322. Entity-type granularity gaps inflate pools common-mode across encoders.
