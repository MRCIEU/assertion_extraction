# Author-level IAA labeling pack

- **`author_iaa_labeling_workbook.xlsx`** — Sheet `1_blind_labeling`: PMID, abstract, entity pair, empty columns for human labels. Sheet `2_REF_post_blind`: heuristic reference (open only after blind pass).
- **`author_iaa_labeling_template.tsv`** — Same as sheet 1 (tab-delimited).
- **`REFERENCE_only_after_blind_labeling__heuristic_labels.tsv`** — Heuristic labels for post-hoc merge (keep separate during labeling).
- **`AUTHOR_IAA_审核指南.md`** — Full protocol and blinding rules (Chinese).
- **`build_author_iaa_template.py`** — Regenerates the three files from  
  `data/processed/inter_annotator_audit/sampled_targets.csv` (expects **n = 30**).

Population and sampling match Supplement **S3** (`supp:iaa`): stratified sample `random.Random(42)`, **27** `gene_drug` + **3** `variant_disease`.
