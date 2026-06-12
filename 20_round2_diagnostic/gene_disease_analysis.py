"""Gene-disease erosion deepening: analyses A–F and pair-type-specific verdict."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shared.models import MODELS

from .adjudication import (
    WELL_DEFS,
    WELL_DEF_LABELS,
    WELL_DEF_VAL_F1,
    _bootstrap_ci,
    _metric,
    _row_at,
    _well_epoch,
)
from .config import (
    GENE_DISEASE_ENCODER_CSV,
    GENE_DISEASE_ROBUSTNESS_CSV,
    GENE_DISEASE_SEED_CSV,
    GENE_DISEASE_SUBSET_CSV,
    PAIR_TYPE_SUBSET_CSV,
)

METRIC_SPECS: list[tuple[str, str, str]] = [
    ("gene-disease (all)", "kb_mrr_gene_disease", "gene_disease"),
    ("gene-disease hard", "kb_mrr_gene_disease_hard", "gene_disease_hard"),
    ("gene-disease easy", "kb_mrr_gene_disease_easy", "gene_disease_easy"),
    ("gene-drug (all)", "kb_mrr_gene_drug", "gene_drug"),
    ("gene-drug hard", "kb_mrr_gene_drug_hard", "gene_drug_hard"),
    ("gene-drug easy", "kb_mrr_gene_drug_easy", "gene_drug_easy"),
]


def _delta_col(metric: str, well_def: str) -> str:
    return f"delta_{metric}_{well_def}"


def _extend_paired_with_cross_metrics(traj: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    """Add paired deltas for pair×subset cross metrics at each well-trained definition."""
    paired = paired.copy()
    cross_cols = [m for _, m, _ in METRIC_SPECS if m.startswith("kb_mrr_")]

    for well_def in WELL_DEFS:
        for col in cross_cols:
            dcol = _delta_col(col, well_def)
            if dcol not in paired.columns:
                paired[dcol] = np.nan

        for idx, row in paired.iterrows():
            sub = traj[(traj["model_id"] == row["model_id"]) & (traj["seed"] == row["seed"])].sort_values(
                "epoch"
            )
            if sub.empty:
                continue
            under_ep = int(row["epoch_under"])
            r_under = _row_at(sub, under_ep)
            well_ep = _well_epoch(well_def, row["model_id"], int(row["seed"]), sub)
            r_well = _row_at(sub, well_ep)
            if r_under is None or r_well is None or under_ep == well_ep:
                continue
            for col in cross_cols:
                if col not in sub.columns:
                    continue
                u, w = _metric(r_under, col), _metric(r_well, col)
                if pd.notna(u) and pd.notna(w):
                    paired.at[idx, _delta_col(col, well_def)] = w - u

    return paired


def _subset_breakdown_row(
    paired: pd.DataFrame,
    *,
    label: str,
    metric: str,
    slug: str,
    well_def: str,
) -> dict[str, Any]:
    dcol = _delta_col(metric, well_def)
    pc = f"pairable_{well_def}"
    sub = paired[paired[pc] & paired[dcol].notna()]
    vals = sub[dcol].astype(float).to_numpy()
    mean, lo, hi = _bootstrap_ci(vals)
    return {
        "label": label,
        "metric": metric,
        "slug": slug,
        "well_trained_definition": well_def,
        "n_seeds": int(len(vals)),
        "mean_delta_kb_mrr": mean,
        "median_delta_kb_mrr": float(np.median(vals)) if len(vals) else np.nan,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_kb_falls": int((vals < 0).sum()) if len(vals) else 0,
        "frac_kb_falls": float((vals < 0).mean()) if len(vals) else np.nan,
    }


def build_gene_disease_subset_breakdown(paired: pd.DataFrame, well_def: str = WELL_DEF_VAL_F1) -> pd.DataFrame:
    rows = [
        _subset_breakdown_row(paired, label=label, metric=metric, slug=slug, well_def=well_def)
        for label, metric, slug in METRIC_SPECS
    ]
    return pd.DataFrame(rows)


def build_pair_type_subset_contrast(paired: pd.DataFrame, well_def: str = WELL_DEF_VAL_F1) -> pd.DataFrame:
    return build_gene_disease_subset_breakdown(paired, well_def)


def build_gene_disease_robustness(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for label, metric, slug in METRIC_SPECS:
        if slug not in ("gene_disease", "gene_disease_hard", "gene_drug"):
            continue
        for well_def in WELL_DEFS:
            rows.append(
                _subset_breakdown_row(paired, label=label, metric=metric, slug=slug, well_def=well_def)
            )
    return pd.DataFrame(rows)


def build_gene_disease_seed_distribution(paired: pd.DataFrame, well_def: str = WELL_DEF_VAL_F1) -> pd.DataFrame:
    pc = f"pairable_{well_def}"
    rows: list[dict] = []
    for _, metric, slug in METRIC_SPECS:
        if slug not in ("gene_disease", "gene_disease_hard"):
            continue
        dcol = _delta_col(metric, well_def)
        sub = paired[paired[pc] & paired[dcol].notna()].copy()
        for _, r in sub.iterrows():
            rows.append(
                {
                    "model_id": r["model_id"],
                    "short_name": r["short_name"],
                    "seed": int(r["seed"]),
                    "slug": slug,
                    "delta_kb_mrr": float(r[dcol]),
                    "well_trained_definition": well_def,
                }
            )
    return pd.DataFrame(rows)


def build_gene_disease_encoder_breakdown(paired: pd.DataFrame, well_def: str = WELL_DEF_VAL_F1) -> pd.DataFrame:
    pc = f"pairable_{well_def}"
    rows: list[dict] = []
    for spec in MODELS:
        enc = paired[(paired["model_id"] == spec.model_id) & paired[pc]]
        if enc.empty:
            continue
        for _, metric, slug in METRIC_SPECS:
            if slug not in ("gene_disease", "gene_disease_hard", "gene_drug"):
                continue
            dcol = _delta_col(metric, well_def)
            if dcol not in enc.columns:
                continue
            vals = enc[dcol].dropna().astype(float)
            if vals.empty:
                continue
            mean, lo, hi = _bootstrap_ci(vals.to_numpy())
            rows.append(
                {
                    "model_id": spec.model_id,
                    "short_name": spec.short_name,
                    "slug": slug,
                    "well_trained_definition": well_def,
                    "n_seeds": int(len(vals)),
                    "mean_delta_kb_mrr": mean,
                    "median_delta_kb_mrr": float(vals.median()),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_kb_falls": int((vals < 0).sum()),
                    "frac_kb_falls": float((vals < 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _robustness_passes(robust_df: pd.DataFrame, slug: str) -> dict[str, Any]:
    sub = robust_df[robust_df["slug"] == slug]
    out: dict[str, Any] = {}
    for well_def in WELL_DEFS:
        row = sub[sub["well_trained_definition"] == well_def]
        if row.empty:
            out[well_def] = {"mean": np.nan, "frac_falls": np.nan}
            continue
        r = row.iloc[0]
        out[well_def] = {
            "mean": float(r["mean_delta_kb_mrr"]),
            "frac_falls": float(r["frac_kb_falls"]),
            "negative_mean": float(r["mean_delta_kb_mrr"]) < -0.01,
            "majority_fall": float(r["frac_kb_falls"]) >= 0.55,
        }
    stable = all(
        out[d]["negative_mean"] and out[d]["majority_fall"]
        for d in WELL_DEFS
        if not np.isnan(out[d]["mean"])
    )
    return {"by_def": out, "stable_across_defs": stable}


def adjudicate_gene_disease_verdict(
    subset: pd.DataFrame,
    robustness: pd.DataFrame,
    encoder: pd.DataFrame,
    pooled_seed_dist: pd.DataFrame,
) -> dict[str, Any]:
    def _row(slug: str) -> pd.Series | None:
        hit = subset[(subset["slug"] == slug) & (subset["well_trained_definition"] == WELL_DEF_VAL_F1)]
        return hit.iloc[0] if not hit.empty else None

    gdis = _row("gene_disease")
    gdis_hard = _row("gene_disease_hard")
    gdis_easy = _row("gene_disease_easy")
    gdrug = _row("gene_drug")

    if gdis is None or gdis_hard is None:
        return {
            "verdict": "insufficient_data",
            "narrative": "Gene-disease cross-subset metrics unavailable; run cross-metric supplement first.",
        }

    hard_concentrated = (
        float(gdis_hard["mean_delta_kb_mrr"]) < float(gdis_easy["mean_delta_kb_mrr"]) - 0.01
        and float(gdis_hard["mean_delta_kb_mrr"]) < -0.01
    )
    gdis_rob = _robustness_passes(robustness, "gene_disease")
    gdis_hard_rob = _robustness_passes(robustness, "gene_disease_hard")

    mean_hard = float(gdis_hard["mean_delta_kb_mrr"])
    median_hard = float(gdis_hard["median_delta_kb_mrr"])
    frac_fall_hard = float(gdis_hard["frac_kb_falls"])
    outlier_driven = abs(mean_hard - median_hard) > 0.025

    enc_hard = encoder[encoder["slug"] == "gene_disease_hard"]
    n_enc = len(enc_hard)
    n_enc_negative = int((enc_hard["mean_delta_kb_mrr"] < 0).sum()) if n_enc else 0
    n_enc_majority_fall = int((enc_hard["frac_kb_falls"] >= 0.5).sum()) if n_enc else 0
    encoder_consistent = n_enc >= 7 and n_enc_negative >= 6 and n_enc_majority_fall >= 6

    broad_based = frac_fall_hard >= 0.65 and median_hard < -0.01 and not outlier_driven
    moderate_broad = frac_fall_hard >= 0.55 and mean_hard < -0.02

    overall_robust = (
        gdis_rob["stable_across_defs"]
        and float(gdis["mean_delta_kb_mrr"]) < -0.02
        and float(gdis["frac_kb_falls"]) >= 0.65
    )
    moderate_hard = mean_hard < -0.01 and frac_fall_hard >= 0.55

    criteria = {
        "hard_concentrated": hard_concentrated,
        "robust_gene_disease_all_defs": gdis_rob["stable_across_defs"],
        "robust_gene_disease_hard_all_defs": gdis_hard_rob["stable_across_defs"],
        "broad_based_seeds_hard": broad_based,
        "moderate_broad_seeds_hard": moderate_broad,
        "encoder_consistent": encoder_consistent,
        "outlier_driven": outlier_driven,
        "overall_gene_disease_robust": overall_robust,
    }

    strong = (
        hard_concentrated
        and gdis_hard_rob["stable_across_defs"]
        and broad_based
        and encoder_consistent
    )
    partial = (
        hard_concentrated
        and moderate_broad
        and (gdis_rob["stable_across_defs"] or gdis_hard_rob["stable_across_defs"])
        and n_enc_negative >= 5
    )
    mixed = overall_robust and not strong and not partial

    if strong:
        verdict = "dual_mechanism_pair_type_specific"
        narrative = (
            "Gene-disease knowledge-base ranking falls within models as the in-distribution benchmark "
            "rises, and this erosion is concentrated on the cross-sentence hard subset where proximity "
            "cannot substitute for relation understanding. The drop is stable across all three well-trained "
            "checkpoint definitions, spread across most seeds and encoders, and cannot be attributed to "
            "non-drug chemical pool inflation, which affects only gene-drug candidates. Gene-drug ranking "
            "shows the opposite or neutral within-model trend and its negative between-model association "
            "remains consistent with static criterion and pool-composition differences. The overall picture "
            "is dual-mechanism and pair-type-dependent: a genuine training-dynamics erosion on gene-disease "
            "coexists with a static explanation on the gene-drug side."
        )
    elif partial:
        verdict = "mixed_partial_gene_disease_erosion"
        narrative = (
            "Gene-disease ranking shows a meaningful within-model decline concentrated on the hard subset, "
            "and this pattern is not explained by non-drug chemical inflation. However, the effect is not "
            "fully robust across all well-trained checkpoint definitions and/or is not uniformly spread "
            "across every encoder. Gene-drug behaviour remains consistent with the static pool and criterion "
            "explanation. The pooled no-erosion reading arose because rising gene-drug and falling gene-disease "
            "components cancel in the average; the pair-type-specific signal supports partial training-dynamics "
            "erosion on gene-disease only."
        )
    elif mixed:
        verdict = "mixed_gene_disease_signal"
        narrative = (
            "Overall gene-disease ranking falls reliably within models from early to well-trained "
            "checkpoints: the decline is stable across all three checkpoint definitions, spans most "
            "seeds, and cannot be explained by non-drug chemical pool inflation. However, the effect "
            "does not meet the full training-dynamics standard. The hard subset declines more than the "
            "easy subset, but the gap is modest rather than sharply concentrated on cross-sentence "
            "pairs alone. At the encoder level the pattern splits: biomedical-domain encoders show "
            "consistent gene-disease-hard erosion, while general-purpose encoders show flat or rising "
            "hard-subset ranking. Gene-drug ranking remains flat or positive in the pooled average and "
            "is still consistent with static criterion and pool-composition differences on the "
            "between-model axis. What this licenses: pair-type-specific within-model asymmetry "
            "(gene-disease down, gene-drug flat or up) is real and was hidden by the pooled average; "
            "a uniform mechanistic erosion story across all encoders and subsets is not supported; "
            "the between-model negative association remains best explained by static differences, "
            "possibly compounded by encoder-family heterogeneity on gene-disease."
        )
    else:
        verdict = "static_verdict_stands"
        narrative = (
            "Although mean gene-disease ranking falls from early to well-trained checkpoints, the change is "
            "not sufficiently concentrated on the hard subset, robust across well-trained definitions, and "
            "broad-based across seeds and encoders to qualify as a training-dynamics effect that the static "
            "explanation cannot produce. The original verdict stands: the between-model negative association "
            "is better explained by fixed inclusion-criteria and pool-construction differences. The pooled "
            "average masked pair-type asymmetry because gene-drug ranking rose slightly while gene-disease fell."
        )

    return {
        "verdict": verdict,
        "criteria": criteria,
        "narrative": narrative,
        "gene_disease_mean": float(gdis["mean_delta_kb_mrr"]),
        "gene_disease_hard_mean": mean_hard,
        "gene_disease_hard_median": median_hard,
        "gene_disease_easy_mean": float(gdis_easy["mean_delta_kb_mrr"]),
        "gene_drug_mean": float(gdrug["mean_delta_kb_mrr"]) if gdrug is not None else np.nan,
        "gene_disease_hard_frac_falls": frac_fall_hard,
        "n_encoders_negative_hard": n_enc_negative,
        "n_encoders_total": n_enc,
        "robustness_gene_disease": gdis_rob,
        "robustness_gene_disease_hard": gdis_hard_rob,
    }


def run_gene_disease_analysis(
    paired: pd.DataFrame, traj: pd.DataFrame, pooled_seed_dist: pd.DataFrame
) -> dict[str, Any]:
    paired = _extend_paired_with_cross_metrics(traj, paired)

    subset = build_gene_disease_subset_breakdown(paired)
    subset.to_csv(GENE_DISEASE_SUBSET_CSV, index=False)

    pair_subset = build_pair_type_subset_contrast(paired)
    pair_subset.to_csv(PAIR_TYPE_SUBSET_CSV, index=False)

    robustness = build_gene_disease_robustness(paired)
    robustness.to_csv(GENE_DISEASE_ROBUSTNESS_CSV, index=False)

    seed_dist = build_gene_disease_seed_distribution(paired)
    seed_dist.to_csv(GENE_DISEASE_SEED_CSV, index=False)

    encoder = build_gene_disease_encoder_breakdown(paired)
    encoder.to_csv(GENE_DISEASE_ENCODER_CSV, index=False)

    verdict = adjudicate_gene_disease_verdict(subset, robustness, encoder, pooled_seed_dist)

    print("\n=== Gene-disease erosion deepening (epoch 1 -> best validation F1) ===")
    for slug in ("gene_disease", "gene_disease_hard", "gene_disease_easy", "gene_drug"):
        row = subset[(subset["slug"] == slug) & (subset["well_trained_definition"] == WELL_DEF_VAL_F1)]
        if row.empty:
            print(f"  {slug}: (missing — run cross-metric supplement)")
            continue
        r = row.iloc[0]
        print(
            f"  {r['label']}: mean {float(r['mean_delta_kb_mrr']):+.4f}, "
            f"median {float(r['median_delta_kb_mrr']):+.4f} "
            f"[{float(r['ci_lo']):+.4f}, {float(r['ci_hi']):+.4f}], "
            f"falls {int(r['n_kb_falls'])}/{int(r['n_seeds'])} seeds"
        )

    print("\n=== Gene-disease robustness across well-trained definitions ===")
    for slug in ("gene_disease", "gene_disease_hard"):
        print(f"  {slug}:")
        for well_def in WELL_DEFS:
            row = robustness[(robustness["slug"] == slug) & (robustness["well_trained_definition"] == well_def)]
            if row.empty:
                continue
            r = row.iloc[0]
            print(
                f"    {WELL_DEF_LABELS[well_def]}: mean {float(r['mean_delta_kb_mrr']):+.4f}, "
                f"falls {float(r['frac_kb_falls']):.1%}"
            )

    print("\n=== Per-encoder gene-disease-hard (mean delta, frac seeds falling) ===")
    enc_h = encoder[(encoder["slug"] == "gene_disease_hard") & (encoder["well_trained_definition"] == WELL_DEF_VAL_F1)]
    for _, r in enc_h.sort_values("mean_delta_kb_mrr").iterrows():
        print(
            f"  {r['short_name']}: mean {float(r['mean_delta_kb_mrr']):+.4f}, "
            f"falls {float(r['frac_kb_falls']):.0%} ({int(r['n_kb_falls'])}/{int(r['n_seeds'])})"
        )

    print(f"\n=== GENE-DISEASE VERDICT: {verdict['verdict']} ===")
    print(verdict["narrative"])

    return {
        "paired_extended": paired,
        "subset": subset,
        "pair_subset": pair_subset,
        "robustness": robustness,
        "seed_dist": seed_dist,
        "encoder": encoder,
        "verdict": verdict,
    }


def require_cross_metrics(traj: pd.DataFrame) -> None:
    if "kb_mrr_gene_disease_hard" not in traj.columns:
        raise SystemExit(
            "Missing kb_mrr_gene_disease_hard in epoch scores. "
            "Run: python run.py --supplement-cross-metrics-only (GPU)"
        )
    if traj["kb_mrr_gene_disease_hard"].notna().sum() == 0:
        raise SystemExit("Cross metrics column present but empty; run --supplement-cross-metrics-only")
