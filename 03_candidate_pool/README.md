# Step 03 — Candidate pool

Builds per-abstract PubTator3 candidate pools for ranking evaluation on the 1,812 evaluable targets frozen in step 02.

**Result:** **87.7%** of gene–drug / gene–disease positives have both entities found by PubTator3 (1,589 / 1,812). Mean pool size ~10.3; mean positive fraction **14.8%** (adequate ranking room). Variant coverage **0.0%** (genuine — CIViC variant strings vs PubTator3 tmVar3). ~18,911 primary-scope candidates across 1,047 PMIDs. Trivial baselines on the real pool: random MRR **0.322**, constant **0.321**, distance ranker **0.489**.