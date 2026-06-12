# Round 1 analysis (folder 11)

Clean-data rerun on folder-10 matrix at **5e-6/none** (72 stable runs, offset markers). Consumes fine-tuned checkpoints; does not train.

## Stages

| Stage | Resource | Entry | Markers |
|-------|----------|-------|---------|
| 1 — KB scoring | GPU | `python run.py --score-only` | 72 `scoring_complete.json` |
| 1b — Untrained floor | GPU | `python run.py --score-untrained-only` | 9 `untrained_scoring_complete.json` |
| 2 — Analysis | CPU | `python run.py --analyze-only` | CSVs, figures, report |

Stage 2 requires **72/72 + 9/9** scoring markers. Use `--force-score` to overwrite on rerun.

## Pre-flight

```bash
conda activate hf-hpc
cd project_1/11_round1_analysis
python preflight.py
```

## Submit (after preflight passes)

```bash
./submit_round1.sh          # 9 parallel fine-tuned + 1 untrained + dependent analysis
# or manually:
sbatch step_score.sbatch
sbatch step_score_untrained.sbatch
# after markers complete:
sbatch step_analyze.sbatch
```

## New analyses (this rerun)

- **Benchmark saturation diagnostic:** same variance-components method on benchmark F1 and KB MRR (Figure 2).
- **Untrained-floor group:** pretrained encoder + random head, no fine-tuning; lift table vs fine-tuned (Figure 4).
- **Absolute KB levels:** fine-tuned MRR vs random (0.322) and distance ranker (0.489) from frozen pool.

## Outputs

- `data/11_round1_analysis/scores/` — per-run KB scores + untrained-floor scores
- `outputs/11_round1_analysis/` — metrics, variance, lift, associations
- `figures/11_round1_analysis/` — four PNGs (300 dpi)
- `reports/11_round1_analysis/report.md` — full prose (derived fresh, not old Round 1 narrative)
