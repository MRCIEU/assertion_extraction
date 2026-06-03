# Step 01 — Corpus relevance

Loads BioRED, DrugProt, and BC5CDR via BigBio and scores each corpus against CIViC pair types, label granularity (RQ1), and trainable volume (RQ3). PMID diagnostics check corpus overlap, binary-presence agreement on shared PMIDs, and train/eval leakage. An oncology-subset analysis quantifies cancer-related training signal under three independent criteria (NCIt neoplasm disease mapping, CIViC gene set, PubMed MeSH neoplasm).

**Run result:** BioRED admissible (4/4 pair types, 100% CIViC-relevance); DrugProt partial (gene–drug only, 53.5%); BC5CDR inadmissible (0/4). BioRED∩DrugProt overlap: **2 PMIDs** (Jaccard 0.0004). **3 DrugProt PMIDs leak** into the evaluation set — recorded in `excluded_pmids.json`; clean lists in `training_pmids_clean.json`. Conservative oncology intersection: **1,086** BioRED gene–disease training relations meet all three criteria (of 31,834).
