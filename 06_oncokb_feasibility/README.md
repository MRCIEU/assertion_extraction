# Step 06 — OncoKB feasibility (corrected annotation probe)

Read-only probe using batch POST on the authenticated production OncoKB annotation API over the full cancer-gene list.

**Verdict:** GO

**Key counts**
- OncoKB-annotated genes queried: 1010
- Total annotation API calls: 10112
- Unique associations: 678
- gene–drug single-PMID / evaluable: 244 / 244
- gene–disease single-PMID / evaluable: 76 / 76
- Abstract retrievable (evaluable PMIDs): 302

Run: `bash -lc 'source ~/.bashrc && conda activate hf-hpc && python project_1/06_oncokb_feasibility/run.py --force-fetch'`
