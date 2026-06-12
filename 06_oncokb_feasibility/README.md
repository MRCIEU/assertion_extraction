# Step 06 — OncoKB feasibility

Read-only probe of OncoKB API abstract grounding for a parallel gene–drug and gene–disease ranking target set.

**Verdict:** GO

**Key counts**
- Therapeutic genes queried: 89
- Annotation queries: 906
- Unique associations retrieved: 532
- Single-PMID associations: 261 (49.1%)
- Multi-PMID associations: 249 (46.8%)
- Evaluable single-PMID triples (training PMIDs excluded): 261

Run: `bash -lc 'source ~/.bashrc && conda activate hf-hpc && python project_1/06_oncokb_feasibility/run.py'`
