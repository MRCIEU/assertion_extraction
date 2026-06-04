# Round 1 analysis (folder 11)

Consumes folder-10 step-2 matrix checkpoints. No training.

## Workflow

1. After step-2 training completes, score KB at best checkpoints (GPU):

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate hf-hpc
export REPO=/home/b5ac/freddieyu.b5ac/project_1
export OUTPUT_ROOT=${REPO}/../projects/project_1
cd ${REPO}/11_round1_analysis
python run.py --score-only
```

2. Run full analysis (CPU):

```bash
python run.py --analyze-only
```

Or combine rescoring and analysis: `python run.py --analyze-only --rescore`

Cluster: `sbatch step_analyze.sbatch`

## Outputs

Flat under `outputs/11_round1_analysis/`, `figures/11_round1_analysis/`, `reports/11_round1_analysis/report.md`.

Includes RoBERTa versus domain-specialised encoder analysis (`11_roberta_analysis.csv`).
