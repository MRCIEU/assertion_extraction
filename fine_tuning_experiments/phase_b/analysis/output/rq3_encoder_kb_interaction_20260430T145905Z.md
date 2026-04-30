# RQ3 Exploratory Encoder × KB-Metric Interaction

Input: `fine_tuning_experiments/phase_b/analysis/output/phase_b_eval_aggregate_LATEST.csv`

Rows: 180 Phase B FT main runs × 3 KB metrics = 540 long rows.

## Partial-SS Audit

| Term | df | partial SS share | F | descriptive p |
|---|---:|---:|---:|---:|
| encoder | 2 | 0.0178 | 13.6526 | 0.000002 |
| schedule_block | 2 | 0.4516 | 346.3786 | 0.000000 |
| kb_metric | 2 | 0.0433 | 33.2073 | 0.000000 |
| encoder_x_kb_metric | 4 | 0.0025 | 0.9559 | 0.431358 |

> Exploratory only: p-values are descriptive and not part of the confirmatory FDR tier.

## Encoder × KB Metric Means

| Encoder | KB_hit_A | KB_pmass_B | KB_auc_C | metric spread | ranking |
|---|---:|---:|---:|---:|---|
| PB | 0.4610 | 0.3814 | 0.6138 | 0.2324 | KB_auc_C > KB_hit_A > KB_pmass_B |
| BL | 0.5902 | 0.4553 | 0.7522 | 0.2969 | KB_auc_C > KB_hit_A > KB_pmass_B |
| PL | 0.5940 | 0.4575 | 0.7216 | 0.2641 | KB_auc_C > KB_hit_A > KB_pmass_B |

## Interpretation Guardrail

Use the encoder_x_kb_metric partial-SS share to assess whether audit formulation changes encoder-level KB conclusions. In the observed run, this interaction is weak; report the result as exploratory evidence against a strong encoder × audit-formulation interaction, not as a confirmatory test.
