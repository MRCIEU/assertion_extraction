# Data Inventory

Current report-level data derived from the post-B.24 Phase B aggregate:

| File | Rows | Description |
|---|---:|---|
| `phase_b_ft_seedlevel.csv` | 190 | One row per realised Phase B FT/RB run. Derived from `phase_b_eval_aggregate_LATEST.csv`. |
| `phase_b_ft_cells.csv` | 10 | Cell-level means/medians/SD/ranges/CI half-widths for 9 main FT cells + RB reference. |

These are convenience reporting tables, not the authoritative raw analysis
inputs.  The authoritative input remains:

```text
fine_tuning_experiments/phase_b/analysis/output/phase_b_eval_aggregate_20260430T145905Z.csv
```

SHA-256:

```text
84a7150dd916faed849c75050a284aae2b0bbe74bca391e3fd316f502545c117
```
