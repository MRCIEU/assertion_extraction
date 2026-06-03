# Round 1 — Benchmark rank vs KB ranking and calibration

First main-experiment round. Fixed BioRED + DrugProt presence training; only encoder and seed vary.

## Question

Does a model's self-measured BioRED benchmark score predict its ranking quality and calibration on the frozen CIViC candidate pool?

## How to run (parallel training)

```bash
export REPO=/path/to/project_1
export OUTPUT_ROOT=${REPO}/../projects/project_1

# Submit 9 parallel jobs (one encoder each, 8 seeds) — train + per-model eval only
./submit_round1_parallel.sh

# After ALL jobs finish, run cross-model analysis once:
cd ${REPO}/10_round1_benchmark_kb && python run.py --analyze-only
```

To use fewer concurrent GPUs, submit a subset:

```bash
ENCODERS_OVERRIDE="pubmedbert_base biolinkbert_base" ./submit_round1_parallel.sh
```

Or split seeds manually:

```bash
export REPO=...
export OUTPUT_ROOT=...
STEP_ARGS="--train-eval-only --models pubmedbert_base --seeds 42 43" \
  sbatch --output="${OUTPUT_ROOT}/runs/10_round1_benchmark_kb/train_pubmedbert_s42-43_%j.out" \
         --error="${OUTPUT_ROOT}/runs/10_round1_benchmark_kb/train_pubmedbert_s42-43_%j.err" \
         step_train.sbatch
```

## How to run (legacy single job)

```bash
export REPO=/path/to/project_1
export OUTPUT_ROOT=${REPO}/../projects/project_1

# Full matrix in one job (may hit wall-time; prefer parallel submission above)
STEP_ARGS="--train-eval-only" sbatch step.sbatch
```

Requires `hf-hpc` conda env and GPU (Isambard sbatch).

## Design

- **9 encoders** across domain (PubMedBERT, BioMedBERT, BioLinkBERT, BioBERT, SciBERT) and general/lightweight (RoBERTa, BERT, DistilBERT, DeBERTa)
- **8 seeds** per encoder (42–49)
- Early stopping on BioRED+DrugProt validation loss
- Blocking leak check: PMIDs 16434489, 18794803, 23430109 must be absent from training

## Outputs

| Location | Contents |
|----------|----------|
| `projects/project_1/data/10_round1_benchmark_kb/` | Checkpoints, scores, completion markers |
| `projects/project_1/outputs/10_round1_benchmark_kb/` | CSV tables |
| `projects/project_1/figures/10_round1_benchmark_kb/` | Scatter plots, reliability diagrams |
| `projects/project_1/reports/10_round1_benchmark_kb/report.md` | Full descriptive report |
| `projects/project_1/runs/10_round1_benchmark_kb/` | Sbatch logs |

## Key numbers

_Run not yet complete — key numbers will appear here after the matrix finishes._

After a successful run, check `outputs/10_round1_benchmark_kb/10_encoder_summary.csv` and the report for benchmark F1 range, benchmark–KB correlations (with CIs), and easy/hard subset comparisons vs the distance ranker.
