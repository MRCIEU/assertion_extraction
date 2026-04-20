# Cancer Assertion Extraction — Code Repository

Code accompanying an in-progress study on **heterogeneous supervision and evaluation validity for cancer-focused biomedical relation extraction**.

This repository contains source code only. Generated artifacts (training data shards, model checkpoints, evaluation results, run logs) live on the cluster filesystem and are not version-controlled. The full design document (research questions, statistical plan, hypothesis registry) is maintained locally and is not part of this repository.

## Status

- **Phase A (exploratory pilot)** — completed. 120 runs spanning 4 encoders × 3 candidate schemas × 10 seeds.
- **Phase B (confirmatory factorial)** — design locked. Implementation pending: trainer rewrite, schema-aligned evaluation metrics, dual-schema (S_pair + S_flat) full factorial, mixed-effects analysis for RQ4.

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

## Known limitations (to be addressed in Phase B)

- The training stack under `fine_tuning_experiments/train/` previously depended on `.pyc`-only modules with no `.py` source. Those compiled artifacts were removed during cleanup. Phase B will re-implement the trainer in clean `.py` source with an integration test and a bridge-equivalence run against Phase A's saved checkpoints.
- Several modules in legacy subdirectories (`dataset_inventory/parsers/`, `dataset_inventory/downloaders/`, etc.) still rely on bytecode-only artifacts; they are not required for Phase B and will be re-derived as needed.

## Running

All training and evaluation are launched via `sbatch` scripts under each subproject. Environment variables required:

```
PROJECT_1_DATA_ROOT     cluster path to data + runs
PYTHONPATH              path to the repository root
HF_HOME                 HuggingFace model cache
```

## License

TBD
