# Round 2 diagnostic (folder 20)

Reads per-epoch checkpoints from folder-10 step-2 matrix. Inference only; no training.

## Fast run (CPU, validation curves from logs only)

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate hf-hpc
export REPO=/home/b5ac/freddieyu.b5ac/project_1
export OUTPUT_ROOT=${REPO}/../projects/project_1
cd ${REPO}/20_round2_diagnostic
python run.py
```

## Full two-axis run (GPU, focus encoders only)

Computes per-epoch self-measured benchmark F1 and KB from saved epoch checkpoints for PubMedBERT, RoBERTa, and DistilBERT (three encoders x eight seeds x up to ten epochs).

```bash
python run.py --rescore-epochs
```

Cluster: `sbatch step_diagnostic.sbatch` (defaults to `--rescore-epochs`; use `STEP_ARGS=""` for fast CPU-only).

Requires folder-11 best-point scores for comparison (`11_per_run_scores.csv`).
