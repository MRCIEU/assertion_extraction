# Recipe sweep and full-matrix training (folder 10)

Produces models only. Does not score KB or run Round 1 analysis.

## Step 1: recipe sweep (GPU)

Four encoders (PubMedBERT, RoBERTa, DistilBERT, DeBERTa) x learning rates {5e-6, 1e-5, 2e-5, 3e-5} x warmup {none, 10pct}, seed 42. Bad-seed guard on collapse (seeds 43, 44). Benchmark F1 only; no KB.

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate hf-hpc
export REPO=/home/b5ac/freddieyu.b5ac/project_1
export OUTPUT_ROOT=${REPO}/../projects/project_1
cd ${REPO}/10_recipe_sweep_and_training
python run.py --sweep-only
python run.py --sweep-advisory-only
python run.py --decide-recipe
```

Outputs (under `outputs/.../sweep/`): advisory table, guard outcomes, `recipe_decision_table.csv`. Figure: `figures/.../sweep/recipe_spread_vs_deberta_health.png`. Report: `reports/.../sweep_report.md` (Recipe decision section).

Cluster: `sbatch step1_sweep.sbatch`

## Step 2: full matrix (GPU, after you set CHOSEN_RECIPE)

Set `CHOSEN_RECIPE = RecipeConfig(...)` in `config.py` (must not be `None`). Step 2 aborts if unset.

Nine encoders x eight seeds = 72 runs. Per-epoch checkpoints (fp16 by default) plus fp32 best checkpoint. Val metrics in `training_log.json`. Benchmark F1 only at the best checkpoint (72 evals total). Per-epoch benchmark F1 is computed in folder 20 on demand.

Storage knobs in `config.py`:
- `SAVE_EPOCH_CHECKPOINTS_FP16 = True` (default)
- `MAX_EPOCH_CHECKPOINTS_TO_KEEP = None` (default: keep all; set integer to cap)

```bash
python run.py --train-only
```

Cluster: `sbatch step2_train.sbatch`

## Checkpoint layout

```
matrix/checkpoints/{model_id}/seed_{seed}/
  best/                    # fp32 val_f1-best (folder 11)
  epochs/epoch_NN/         # fp16 weights (folder 20)
  training_log.json        # val_loss, val_f1 per epoch (no benchmark F1)
```
