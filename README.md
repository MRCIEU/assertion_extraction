# assertion_extraction

Code for a descriptive study: does public relation-extraction **benchmark** performance predict **CIViC** curation ranking utility?

Outputs (data, tables, figures, reports) live outside this repo under `../projects/project_1/` (`OUTPUT_ROOT`).

## Layout

| Path | Role |
|------|------|
| `00_`–`06_` | Preparation / feasibility |
| `10_recipe_sweep_and_training/` | Recipe sweep + 72-run training matrix |
| `11_round1_analysis/` | Benchmark vs KB analysis |
| `20_round2_diagnostic/` | Training-dynamics diagnostic |
| `shared/` | Models, metrics, training helpers |
| `manuscript_regenerate/` | Figure / table regeneration |

Each step folder has a short `README.md` and its own sbatch entry point.

## Setup

```bash
conda activate hf-hpc   # or: pip install -r requirements.txt
export REPO=/path/to/assertion_extraction
export OUTPUT_ROOT=$REPO/../projects/project_1
export PYTHONPATH=$REPO
```

Run stages separately via the step sbatch scripts; nothing auto-chains.
