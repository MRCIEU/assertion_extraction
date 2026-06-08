# Round 2 diagnostic (folder 20)

Reads folder-10 per-epoch checkpoints and folder-11 Round 1 results. **No training.**

## Stages

| Stage | Resource | Entry | Output |
|-------|----------|-------|--------|
| 1 — Epoch scoring | GPU | `step_score_epochs.sbatch` | `data/20_round2_diagnostic/epoch_kb_trajectory.csv` |
| 2 — Analysis | CPU | `step_analyze.sbatch` | `outputs/`, `figures/`, `reports/` |

Focus encoders only: PubMedBERT, RoBERTa, DistilBERT (~187 epoch checkpoints).

## Pre-flight

```bash
conda activate hf-hpc
export REPO=/home/b5ac/freddieyu.b5ac/project_1
export OUTPUT_ROOT=${REPO}/../projects/project_1
cd ${REPO}/20_round2_diagnostic
python preflight.py
```

## Submit (after pre-flight passes)

```bash
chmod +x submit_diagnostic.sh
./submit_diagnostic.sh
```

Or manually:
```bash
sbatch step_score_epochs.sbatch
# after epoch_scoring_complete.json shows complete: true
sbatch step_analyze.sbatch
```

Check scoring completion:
```bash
cat ${OUTPUT_ROOT}/data/20_round2_diagnostic/epoch_scoring_complete.json
python -c "import pandas as pd; df=pd.read_csv('${OUTPUT_ROOT}/data/20_round2_diagnostic/epoch_kb_trajectory.csv'); print(df['kb_scored'].sum() if 'kb_scored' in df else df['kb_mrr_hard'].notna().sum())"
```
