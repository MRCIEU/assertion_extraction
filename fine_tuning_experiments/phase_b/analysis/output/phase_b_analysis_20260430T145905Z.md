# Phase B — primary-hypothesis analysis

**Coverage**: 190 runs loaded (180 main + 10 RB reference); expected 190.

**Factorial** (post-amendment 2026-04-16): 3 encoders × 1 update × 3 schedule = 9 cells × 20 seeds = 180 main + 10 RB reference.

## H1
**Verdict**: `partial_or_intermediate`

```json
{
  "hypothesis": "H1",
  "tests": [
    {
      "contrast": "PL_vs_PB",
      "mean_diff": -0.007638207147656005,
      "ci95": [
        -0.032093278384845846,
        0.013179647498514227
      ],
      "cohens_d": -0.14430292617895316,
      "paired_t_p": 0.5187053833635351,
      "wilcoxon_p": 0.681322326150025,
      "n_pairs": 20,
      "q_t": 0.5187053833635351,
      "q_w": 0.681322326150025
    },
    {
      "contrast": "PL_vs_BL",
      "mean_diff": -0.02105699119144892,
      "ci95": [
        -0.04308989984564961,
        -0.0014552681253228992
      ],
      "cohens_d": -0.43261528928979015,
      "paired_t_p": 0.05302535354905524,
      "wilcoxon_p": 0.07932167969676174,
      "n_pairs": 20,
      "q_t": 0.15907606064716573,
      "q_w": 0.2379650390902852
    },
    {
      "contrast": "PB_vs_BL",
      "mean_diff": -0.013418784043792913,
      "ci95": [
        -0.03181499420029805,
        0.004105455072494749
      ],
      "cohens_d": -0.3168221385233967,
      "paired_t_p": 0.1565204486446714,
      "wilcoxon_p": 0.2958775226696384,
      "n_pairs": 20,
      "q_t": 0.2347806729670071,
      "q_w": 0.4438162840044576
    }
  ],
  "verdict": "partial_or_intermediate",
  "anchor": {
    "update": "FT",
    "schedule": "T2"
  }
}
```

## H2
**Verdict**: `confirmed`

```json
{
  "hypothesis": "H2",
  "mean_diff": 0.13921302922873946,
  "ci95": [
    0.10823850909307761,
    0.16621348716303475
  ],
  "cohens_d": 2.0376754770758767,
  "paired_t_p": 0.0,
  "wilcoxon_p": 0.00010334649564658349,
  "n_pairs": 20,
  "verdict": "confirmed"
}
```

## H3
**Verdict**: `partial`

```json
{
  "hypothesis": "H3",
  "tests": [
    {
      "contrast": "PB_biored_ex_neg",
      "mean_diff": 0.05285644027200436,
      "ci95": [
        0.027085690768449185,
        0.07850024978696024
      ],
      "paired_t_p": 0.00011034735131487672,
      "wilcoxon_p": 0.0010188290449537618,
      "n_pairs": 20,
      "q_t": 0.0006620841078892603,
      "q_w": 0.006112974269722571
    },
    {
      "contrast": "PB_bc5cdr_dd",
      "mean_diff": 0.03025657131699011,
      "ci95": [
        0.011760676408712095,
        0.04972689763032963
      ],
      "paired_t_p": 0.0024104258361874464,
      "wilcoxon_p": 0.0099963884759795,
      "n_pairs": 20,
      "q_t": 0.007231277508562339,
      "q_w": 0.029989165427938502
    },
    {
      "contrast": "BL_biored_ex_neg",
      "mean_diff": 0.02283563454399085,
      "ci95": [
        0.006846005290474205,
        0.04001519742136062
      ],
      "paired_t_p": 0.007854386040100803,
      "wilcoxon_p": 0.020633435105949616,
      "n_pairs": 20,
      "q_t": 0.015708772080201605,
      "q_w": 0.04126687021189923
    },
    {
      "contrast": "BL_bc5cdr_dd",
      "mean_diff": -0.003470921649528952,
      "ci95": [
        -0.017655770286255333,
        0.011563590419507525
      ],
      "paired_t_p": 0.6519897980323459,
      "wilcoxon_p": 0.7938390455415016,
      "n_pairs": 20,
      "q_t": 0.6519897980323459,
      "q_w": 0.7938390455415018
    },
    {
      "contrast": "PL_biored_ex_neg",
      "mean_diff": 0.01057102143652967,
      "ci95": [
        -0.01757510593498325,
        0.03867938176018549
      ],
      "paired_t_p": 0.4674910602223501,
      "wilcoxon_p": 0.3317228918094637,
      "n_pairs": 20,
      "q_t": 0.5609892722668202,
      "q_w": 0.3980674701713564
    },
    {
      "contrast": "PL_bc5cdr_dd",
      "mean_diff": 0.02778190158040993,
      "ci95": [
        0.0032253511315627702,
        0.052579355881277164
      ],
      "paired_t_p": 0.03165122557945654,
      "wilcoxon_p": 0.03334022025101757,
      "n_pairs": 20,
      "q_t": 0.047476838369184815,
      "q_w": 0.05001033037652636
    }
  ],
  "n_confirmed": 3,
  "verdict": "partial"
}
```

## H4
**Verdict**: `empirically_undefined_lora_collapsed`

```json
{
  "hypothesis": "H4",
  "status": "methodological_null",
  "verdict": "empirically_undefined_lora_collapsed",
  "reason": "LoRA arm dropped per Appendix B.24 after three attempts (LR=2e-5/2048, LR=3e-4/2048, LR=2e-5/4096) all collapsed to bit-identical 100%-NEGATIVE dev predictions. A collapsed LoRA comparator is not a fair FT-vs-LoRA test.",
  "tests": []
}
```

## H5
**Verdict**: `deferred`

```json
{
  "hypothesis": "H5",
  "status": "deferred_to_future_work",
  "reason": "shared_multitask architecture dropped from Phase B factorial; Appendix B amendment 2026-04-16.",
  "verdict": "deferred"
}
```

## H7
**Verdict**: `null_no_asymmetry`

```json
{
  "hypothesis": "H7",
  "R_B": 0.2136030117018241,
  "lever_share_biored_ex_neg": 0.14210179675379075,
  "lever_share_kb_hit_a": 0.6652612040515403,
  "decomposition": {
    "biored_macro_f1_ex_neg": {
      "encoder": 0.028005635174123026,
      "schedule": 0.08229212437565471,
      "encoder_x_schedule": 0.03180403720401302
    },
    "kb_hit_A_setvalued": {
      "encoder": 0.04740940981887442,
      "schedule": 0.596228521372513,
      "encoder_x_schedule": 0.021623272860152865
    }
  },
  "threshold": 2.0,
  "threshold_justification": "first-principles: R = 1 = perfect benchmark-KB proxy; R >= 2 = >=2x disparity. Axis-count independent; retained unchanged after arch drop.",
  "verdict": "null_no_asymmetry"
}
```

## h7_R_B_bootstrap
**Verdict**: `None`

```json
{
  "point_estimate": 0.2136030117018241,
  "bootstrap_median": 0.22105572343770333,
  "ci_lower": 0.02752887421065751,
  "ci_upper": 0.9901540648994833,
  "n_resamples": 5000,
  "n_successful_resamples": 5000,
  "failed_resamples": 0,
  "n_cells_used": 9,
  "factors": [
    "encoder",
    "schedule"
  ],
  "metric_num": "biored_macro_f1_ex_neg",
  "metric_den": "kb_hit_A_setvalued",
  "seed": 20260417,
  "lever_shares_observed": {
    "biored_macro_f1_ex_neg": {
      "encoder": 0.028005635174123026,
      "schedule": 0.08229212437565471,
      "encoder_x_schedule": 0.03180403720401302
    },
    "kb_hit_A_setvalued": {
      "encoder": 0.04740940981887442,
      "schedule": 0.596228521372513,
      "encoder_x_schedule": 0.021623272860152865
    }
  }
}
```

## rq4_ordinal_instability
**Verdict**: `None`

```json
{
  "rho": 0.03,
  "n_cells_used": 9,
  "n_eligible_pairs": 18,
  "n_eligible_nonzero_pairs": 18,
  "median_delta_KB": 0.15956790123456788,
  "median_delta_KB_ci": [
    0.0,
    0.4694444444444445
  ],
  "rank_inversion_rate": 0.5,
  "rank_inversion_rate_ci": [
    0.14285714285714285,
    0.8333333333333334
  ],
  "delta_KB_distribution": [
    0.4611111111111112,
    0.15956790123456788,
    0.08117283950617282,
    0.06635802469135799,
    0.4030864197530865,
    0.02901234567901234,
    0.08672839506172836,
    0.6206790123456791,
    0.014814814814814836,
    0.48425925925925933,
    0.005555555555555536,
    0.6058641975308643,
    0.4694444444444445,
    0.037345679012345645,
    0.020370370370370372,
    0.43209876543209885,
    0.48981481481481487,
    0.05771604938271602
  ],
  "n_resamples": 5000,
  "failed_resamples": 0,
  "n_successful_resamples_median": 5000,
  "n_successful_resamples_inversion": 4997,
  "factors": [
    "encoder",
    "schedule"
  ],
  "metric_bench": "biored_macro_f1_ex_neg",
  "metric_kb": "kb_hit_A_setvalued",
  "exclude_RB": true,
  "pairs_csv_path": "/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/phase_b/analysis/output/ordinal_instability_pairs.csv",
  "seed": 20260418
}
```

## H6
H6 mechanism-stratified slopes are computed by fine_tuning_experiments.phase_b.analysis.h6_coupling_slopes; run that script separately with both Phase A and Phase B CSVs.
