# Step 02 — Ranking evaluation protocol

Freezes the evaluable ranking target set and defines metrics only.

**Result:** **1,812** abstract-grounded gene–drug and gene–disease positives across **915** PMIDs (1,230 gene–drug; 582 gene–disease). The step-00 inventory contains **2,074** abstract-grounded pairs in total; the remaining **262 variant pairs are not evaluable** (PubTator3 cannot build variant candidate pools — 0% coverage in step 03). Metrics: Mean Reciprocal Rank (MRR), Recall@k, and area under the precision-recall curve (AUC-PR). Trivial baselines and tie-handling verification run in step 03 on the real frozen pool.