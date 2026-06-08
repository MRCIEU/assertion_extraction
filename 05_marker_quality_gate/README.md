# Step 05: marker and span quality gate

Verifies offset-based `[E1]` / `[E2]` marker insertion across training, benchmark, and CIViC evaluation. Rebuilds the canonical train/val example cache with the repaired shared code.

## Run (CPU)

```bash
export REPO=/path/to/project_1
export OUTPUT_ROOT=${REPO}/../projects/project_1
cd ${REPO}/05_marker_quality_gate
conda activate hf-hpc
PYTHONUNBUFFERED=1 python run.py
```

Or: `sbatch --chdir=${REPO}/05_marker_quality_gate step.sbatch`

## Outputs

- `outputs/05_marker_quality_gate/quality_gate_results.json`
- `outputs/05_marker_quality_gate/quality_gate_checks.csv`
- `data/05_marker_quality_gate/train_cache/` (local copy)
- `data/10_recipe_sweep_and_training/cache/` (canonical cache for folder 10, rebuilt)
- `reports/05_marker_quality_gate/report.md`

Does not retrain folder 10 or rescore folders 11/20.
