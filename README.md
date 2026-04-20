# Cancer Assertion Extraction — Code Repository

Code accompanying an in-progress study on **heterogeneous supervision and evaluation validity for cancer-focused biomedical relation extraction**.

This repository contains source code only. Generated artifacts (training data shards, model checkpoints, evaluation results, run logs) live on the cluster filesystem and are not version-controlled. The full design document (research questions, statistical plan, hypothesis registry) is maintained locally and is not part of this repository.


## Directory layout

```
data_pipeline/          legacy; superseded by dataset_inventory + oncology_projection
dataset_inventory/      raw-data audits, leakage detection, corpus statistics
oncology_projection/    MeSH C04 oncology subset derivation, cancer lexicon
schema_exploration/     schema definitions (S_flat / S_pair / S_mech), package remapping
training_data_generation/   T1/T2/T3/T4 data preparation (output JSONL on cluster)
fine_tuning_experiments/    training pipeline + Phase A scripts + Phase A-eval
external_evaluation/    benchmark loaders and inference helpers
knowledge_grounded_evidence_audit/   CIViC-anchored downstream KB audit
report/, reports/       legacy intermediate reports (kept locally; not uploaded)
```

The main currently-active code paths are:
- `fine_tuning_experiments/schema_exp/` — Phase A configs, sbatch, and eval pipeline.
- `fine_tuning_experiments/schema_exp/eval/` — three-pass inference (BioRED test, BC5CDR test, KB-surface) and aggregation.
- `schema_exploration/` — schema label functions and data package remapping.
- `oncology_projection/` — MeSH C04 keyword/MeSH projection of T2 oncology subset.
- `dataset_inventory/audit/` — raw data audits and leakage validation.
