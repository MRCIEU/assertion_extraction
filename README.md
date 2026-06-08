# Project 1: two-stage encoder study

Preparation steps 00–05 freeze data, protocol, the CIViC candidate pool, and marker-quality verification.

## Main study (rebuilt layout)

| Folder | Role |
|--------|------|
| `10_recipe_sweep_and_training/` | **Produces models** — step 1 recipe sweep, step 2 full matrix (72 runs) |
| `11_round1_analysis/` | **Consumes models** — Round 1 benchmark vs KB analysis |
| `20_round2_diagnostic/` | **Consumes models** — training-dynamics diagnostic and power check |
| `21_round2_experiment/` | Placeholder for Round 2 main experiment (after folder 20) |
| `shared/` | Unified training, benchmark F1, KB scoring, distance analysis |

Legacy folders `10_round1_benchmark_kb/` and `11_encoder_recipe_check/` are deprecated; see their `DEPRECATED.md` files.

## Environment

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate hf-hpc
export REPO=/home/b5ac/freddieyu.b5ac/project_1
export OUTPUT_ROOT=${REPO}/../projects/project_1
```

Each stage has its own standalone sbatch script. Run stages separately; nothing auto-chains.
