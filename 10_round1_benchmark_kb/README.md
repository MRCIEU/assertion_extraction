# Round 1 — Benchmark rank vs KB ranking and calibration

First main-experiment round. Fixed BioRED + DrugProt presence training; only encoder and seed vary.

## Question

Does a model's self-measured BioRED benchmark score predict its ranking quality and calibration on the frozen CIViC candidate pool? Either alignment or divergence is a valid finding.

## Fixed training strategy (settled by sweep)

| Setting | Value |
|---------|--------|
| Learning rate | 2e-5 |
| Warmup | none |
| Checkpoint | best **validation F1** (not val_loss) |
| Early stopping | max 10 epochs, patience 3 |

## How to run

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate hf-hpc

export REPO=/path/to/project_1
export OUTPUT_ROOT=${REPO}/../projects/project_1
cd ${REPO}/10_round1_benchmark_kb

# Submit 9 parallel jobs (one encoder × 8 seeds) — train + per-model eval ONLY
./submit_round1_parallel.sh

# After ALL 72 round1_complete.json markers exist, finalise ONCE:
python run.py --analyze-only
```

**Concurrent GPUs assumed:** 9 (one job per encoder). To use fewer:

```bash
ENCODERS_OVERRIDE="pubmedbert_base biolinkbert_base" ./submit_round1_parallel.sh
```

Jobs skip runs that already have a completion marker with the current strategy tag (`val_f1_lr2e5_nowarmup`). Re-submitting resumes without retraining finished runs.

## Design

- **9 encoders:** PubMedBERT, BioMedBERT, BioLinkBERT, BioBERT, SciBERT, RoBERTa, BERT, DistilBERT, DeBERTa
- **8 seeds** per encoder (42–49) → 72 runs
- Blocking leak check: PMIDs 16434489, 18794803, 23430109 absent from training

## Outputs

| Location | Contents |
|----------|----------|
| `projects/project_1/data/10_round1_benchmark_kb/` | Checkpoints, scores, completion markers |
| `projects/project_1/outputs/10_round1_benchmark_kb/` | CSV tables |
| `projects/project_1/figures/10_round1_benchmark_kb/` | Figures |
| `projects/project_1/reports/10_round1_benchmark_kb/report.md` | Descriptive report |
| `projects/project_1/runs/10_round1_benchmark_kb/` | Sbatch logs |

## Key numbers

_Run not yet complete with the val_f1 checkpoint strategy — numbers appear here after the matrix finishes._

Prior partial runs used val_loss checkpoints and are **not** counted complete; re-training will overwrite them.

After completion, see `outputs/10_round1_benchmark_kb/10_encoder_summary.csv` and `report.md` for benchmark F1 spread (9 encoders), benchmark–KB correlations with CIs, and analyses A–D.
