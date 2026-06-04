# Encoder recipe check (folder 11)

Small diagnostic: is DeBERTa's low Round-1 benchmark F1 (mean **0.554**) a **recipe artifact** (no warmup) or a **genuine floor**? Uses the same training data and self-measured BioRED benchmark protocol as folder 10. **No CIViC / KB evaluation.**

## Step 0 (from Round 1)

Two degenerate runs in `outputs/10_round1_benchmark_kb/10_degenerate_runs.csv`:

| Encoder | Seed | Flags |
|---------|------|-------|
| DeBERTa-base | 45 | val_f1_zero, benchmark_f1_zero |
| DeBERTa-base | 49 | val_f1_zero, benchmark_f1_zero |

**Both are DeBERTa.** Not re-run here; only identified.

## Grid (DeBERTa only)

| Learning rate | Warmup | Seed |
|---------------|--------|------|
| 1e-5 | none | 42 |
| 2e-5 | none | 42 |
| 1e-5 | linear 10% steps | 42 |
| 2e-5 | linear 10% steps | 42 |

Bad-seed guard: if any primary run collapses (val F1 or benchmark F1 ~ 0), seeds **43** and **44** run automatically for that recipe only.

## Commands

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate hf-hpc
export REPO=/home/b5ac/freddieyu.b5ac/project_1
export OUTPUT_ROOT=${REPO}/../projects/project_1
cd ${REPO}/11_encoder_recipe_check

# 1) GPU grid (4 runs, ~1 sbatch job)
chmod +x submit_grid.sh submit_analyze.sh
./submit_grid.sh

# 2) After grid finishes, finalise once (CPU)
./submit_analyze.sh
# or locally: python run.py --analyze-only
```

**Optional fallback** (not submitted by default): lr=5e-6 + 10% warmup, seed 42:

```bash
python run.py --train-fallback-only
```

## Round-1 reference band (eight non-DeBERTa encoders)

Benchmark F1 means from folder 10: **0.725** (BERT-base) to **0.785** (BioLinkBERT-base). DeBERTa Round-1 mean: **0.554**.

_Key numbers from the grid appear in stdout and in `outputs/11_encoder_recipe_check/grid_results.csv` after training._

## Outputs

- `data/11_encoder_recipe_check/` checkpoints and per-epoch curves  
- `outputs/11_encoder_recipe_check/` CSV tables  
- `figures/11_encoder_recipe_check/` val curves and encoder strip  
- `reports/11_encoder_recipe_check/report.md` descriptive reading (no go/no-go)  
- `runs/11_encoder_recipe_check/` sbatch logs  

This step **does not** decide whether to re-run the full 9×8 matrix; it only reports numbers and curves.
