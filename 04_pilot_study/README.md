# Step 04 — Pilot study

Train three encoders (PubMedBERT-base, BioLinkBERT-base, RoBERTa-base) on BioRED + DrugProt with excluded evaluation PMIDs enforced, scores the frozen step-03 pool, and reports descriptive preliminary evidence on ranking, benchmark–KB decoupling, and calibration. Includes a distance-confound diagnostic (no re-training) to explain why the best model trails the distance ranker on the full pool.

**Run result:** Best trained model PubMedBERT MRR **0.469** vs random **0.322** and distance ranker **0.489** on the full pool. Distance-confound diagnostic: **53.3%** of CIViC positives are co-sentence; on **cross-sentence (hard)** pairs PubMedBERT MRR **0.427** beats the distance ranker (**0.369**), favouring **under-training** over a purely distance-dominated task at pilot scale. Non-degenerate score distributions; benchmark–KB order and calibration decoupling not observed at n=3. 

**Reference:** constant MRR ≈ random ≈ 0.32; best trained MRR ≈ 0.47; non-degenerate score distributions (means ~0.31–0.37).
