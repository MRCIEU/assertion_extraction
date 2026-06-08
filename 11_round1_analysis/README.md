# Round 1 analysis

Consumes folder-10 step-2 matrix checkpoints. **Does not train.**

## Stages

| Stage | Resource | Entry point | Output |
|-------|----------|-------------|--------|
| 1 — KB scoring | GPU | `step_score.sbatch` or `python run.py --score-only` | `data/11_round1_analysis/scores/` + per-run `scoring_complete.json` markers |
| 2 — Analysis | CPU | `step_analyze.sbatch` or `python run.py --analyze-only` | `outputs/`, `figures/`, `reports/` |

Stage 2 requires **72/72** scoring markers. It will not auto-score on the CPU path.

## Pre-flight (before submit)

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate hf-hpc
export REPO=/home/b5ac/freddieyu.b5ac/project_1
export OUTPUT_ROOT=${REPO}/../projects/project_1
cd ${REPO}/11_round1_analysis
python preflight.py
```

## Submit (parallel scoring + auto stage 2)

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate hf-hpc
export REPO=/home/b5ac/freddieyu.b5ac/project_1
export OUTPUT_ROOT=${REPO}/../projects/project_1
cd ${REPO}/11_round1_analysis
./submit_round1.sh
```

This submits **9 GPU scoring jobs** (one encoder each) and **1 CPU analysis job** with `--dependency=afterok:...` so stage 2 starts only after all scoring jobs succeed.

Manual single-job alternative: `sbatch step_score.sbatch` then `sbatch step_analyze.sbatch` after 72/72 markers.

## Outputs

- `data/11_round1_analysis/scores/{model_id}/seed_{seed}.jsonl` — per-candidate scores
- `outputs/11_round1_analysis/` — flat CSVs (encoder summary, variance components, pool-size robustness, RoBERTa table, etc.)
- `figures/11_round1_analysis/` — four publication PNGs
- `reports/11_round1_analysis/report.md` — full prose; `README.md` — key numbers (filled after analysis)

Shared pool-size table for folder 20: `outputs/11_round1_analysis/11_abstract_pool_sizes.csv`.
