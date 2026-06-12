"""Adjudicate Explanation 1 (training dynamics) vs Explanation 2 (static criterion/pool)."""

from __future__ import annotations

import json
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd

from shared.constants import TRAIN_SEEDS
from shared.models import MODEL_BY_ID, MODELS

from .config import (
    HARD_EASY_CSV,
    PAIRED_CHANGES_CSV,
    PAIR_TYPE_CSV,
    R11_VARIANCE_CSV,
    ROBUSTNESS_CSV,
    SEED_DISTRIBUTION_CSV,
    TRAJECTORY_CSV,
)
from .epoch_scoring import load_all_epoch_scores
from .matrix_io import load_training_meta

WELL_DEF_VAL_F1 = "val_f1_best"
WELL_DEF_LAST = "last_epoch"
WELL_DEF_FIXED5 = "fixed_epoch5"
WELL_DEFS = (WELL_DEF_VAL_F1, WELL_DEF_LAST, WELL_DEF_FIXED5)
WELL_DEF_LABELS = {
    WELL_DEF_VAL_F1: "best validation F1 checkpoint",
    WELL_DEF_LAST: "last saved epoch",
    WELL_DEF_FIXED5: "fixed epoch 5 (capped at last saved)",
}


def _bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        v = float(values[0])
        return v, v, v
    boots = [float(rng.choice(values, size=len(values), replace=True).mean()) for _ in range(n)]
    return float(np.mean(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _well_epoch(well_def: str, model_id: str, seed: int, sub: pd.DataFrame) -> int:
    if well_def == WELL_DEF_VAL_F1:
        meta = load_training_meta(model_id, seed)
        if meta and meta.get("best_epoch_val_f1"):
            return int(meta["best_epoch_val_f1"])
        return int(sub.loc[sub["val_f1"].idxmax(), "epoch"])
    if well_def == WELL_DEF_LAST:
        return int(sub["epoch"].max())
    if well_def == WELL_DEF_FIXED5:
        return min(5, int(sub["epoch"].max()))
    raise ValueError(well_def)


def _row_at(sub: pd.DataFrame, ep: int) -> pd.Series | None:
    hit = sub[sub["epoch"] == ep]
    return hit.iloc[0] if not hit.empty else None


def _metric(row: pd.Series | None, col: str) -> float:
    if row is None or col not in row.index:
        return float("nan")
    return float(row[col])


def build_trajectory_table(scores: pd.DataFrame | None = None) -> pd.DataFrame:
    if scores is None:
        scores = load_all_epoch_scores()
    if scores.empty:
        return scores
    scores = scores.sort_values(["model_id", "seed", "epoch"]).copy()
    scores["source"] = "matrix_per_epoch"
    return scores


def build_within_seed_paired_changes(traj: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in MODELS:
        for seed in TRAIN_SEEDS:
            sub = traj[(traj["model_id"] == spec.model_id) & (traj["seed"] == seed)].sort_values("epoch")
            if sub.empty or not (sub["epoch"] == 1).any():
                continue
            under_ep = 1
            end_ep = int(sub["epoch"].max())
            r_under = _row_at(sub, under_ep)
            if r_under is None:
                continue

            row: dict[str, Any] = {
                "model_id": spec.model_id,
                "short_name": spec.short_name,
                "seed": int(seed),
                "epoch_under": under_ep,
                "epoch_end": end_ep,
            }
            for label, ep in [("under", under_ep), ("end", end_ep)]:
                r = _row_at(sub, ep)
                for col in (
                    "benchmark_f1",
                    "kb_mrr_hard",
                    "kb_mrr_easy",
                    "kb_mrr_gene_drug",
                    "kb_mrr_gene_disease",
                ):
                    row[f"{col}_{label}"] = _metric(r, col)

            for well_def in WELL_DEFS:
                well_ep = _well_epoch(well_def, spec.model_id, int(seed), sub)
                r_well = _row_at(sub, well_ep)
                pairable = (
                    r_under is not None
                    and r_well is not None
                    and under_ep != well_ep
                    and pd.notna(_metric(r_under, "benchmark_f1"))
                    and pd.notna(_metric(r_well, "benchmark_f1"))
                )
                row[f"epoch_well_{well_def}"] = well_ep
                row[f"pairable_{well_def}"] = pairable
                if pairable:
                    for axis in (
                        "benchmark_f1",
                        "kb_mrr_hard",
                        "kb_mrr_easy",
                        "kb_mrr_gene_drug",
                        "kb_mrr_gene_disease",
                    ):
                        row[f"delta_{axis}_{well_def}"] = _metric(r_well, axis) - _metric(r_under, axis)
                    row[f"benchmark_rises_{well_def}"] = row[f"delta_benchmark_f1_{well_def}"] > 0
                    row[f"kb_hard_falls_{well_def}"] = row[f"delta_kb_mrr_hard_{well_def}"] < 0
                    row[f"kb_gdis_falls_{well_def}"] = row[f"delta_kb_mrr_gene_disease_{well_def}"] < 0
                    row[f"erosion_pattern_{well_def}"] = (
                        row[f"benchmark_rises_{well_def}"] and row[f"kb_hard_falls_{well_def}"]
                    )
                else:
                    for axis in (
                        "benchmark_f1",
                        "kb_mrr_hard",
                        "kb_mrr_easy",
                        "kb_mrr_gene_drug",
                        "kb_mrr_gene_disease",
                    ):
                        row[f"delta_{axis}_{well_def}"] = np.nan

            rows.append(row)
    return pd.DataFrame(rows)


def build_seed_erosion_distribution(paired: pd.DataFrame, well_def: str = WELL_DEF_VAL_F1) -> pd.DataFrame:
    rows: list[dict] = []
    pc = f"pairable_{well_def}"
    sub = paired[paired[pc]].copy()
    if sub.empty:
        return pd.DataFrame()

    for spec in MODELS:
        enc = sub[sub["model_id"] == spec.model_id]
        n = len(enc)
        if n == 0:
            continue
        erosion = enc[f"erosion_pattern_{well_def}"].sum()
        bench_up = enc[f"benchmark_rises_{well_def}"].sum()
        kb_down = enc[f"kb_hard_falls_{well_def}"].sum()
        gdis_down = enc[f"kb_gdis_falls_{well_def}"].sum()
        rows.append(
            {
                "model_id": spec.model_id,
                "short_name": spec.short_name,
                "well_trained_definition": well_def,
                "n_seeds_pairable": n,
                "n_benchmark_rises": int(bench_up),
                "n_kb_hard_falls": int(kb_down),
                "n_erosion_benchmark_up_kb_hard_down": int(erosion),
                "frac_erosion": float(erosion / n),
                "n_kb_gene_disease_falls": int(gdis_down),
                "mean_delta_benchmark": float(enc[f"delta_benchmark_f1_{well_def}"].mean()),
                "mean_delta_kb_hard": float(enc[f"delta_kb_mrr_hard_{well_def}"].mean()),
                "mean_delta_kb_gene_drug": float(enc[f"delta_kb_mrr_gene_drug_{well_def}"].mean()),
                "mean_delta_kb_gene_disease": float(enc[f"delta_kb_mrr_gene_disease_{well_def}"].mean()),
            }
        )

    all_p = sub
    n_all = len(all_p)
    erosion_all = int(all_p[f"erosion_pattern_{well_def}"].sum())
    rows.append(
        {
            "model_id": "ALL",
            "short_name": "All encoders pooled",
            "well_trained_definition": well_def,
            "n_seeds_pairable": n_all,
            "n_benchmark_rises": int(all_p[f"benchmark_rises_{well_def}"].sum()),
            "n_kb_hard_falls": int(all_p[f"kb_hard_falls_{well_def}"].sum()),
            "n_erosion_benchmark_up_kb_hard_down": erosion_all,
            "frac_erosion": float(erosion_all / n_all) if n_all else np.nan,
            "n_kb_gene_disease_falls": int(all_p[f"kb_gdis_falls_{well_def}"].sum()),
            "mean_delta_benchmark": float(all_p[f"delta_benchmark_f1_{well_def}"].mean()),
            "mean_delta_kb_hard": float(all_p[f"delta_kb_mrr_hard_{well_def}"].mean()),
            "mean_delta_kb_gene_drug": float(all_p[f"delta_kb_mrr_gene_drug_{well_def}"].mean()),
            "mean_delta_kb_gene_disease": float(all_p[f"delta_kb_mrr_gene_disease_{well_def}"].mean()),
        }
    )
    return pd.DataFrame(rows)


def build_hard_easy_breakdown(paired: pd.DataFrame, well_def: str = WELL_DEF_VAL_F1) -> pd.DataFrame:
    pc = f"pairable_{well_def}"
    sub = paired[paired[pc]]
    rows: list[dict] = []
    for subset, col in [("hard_cross_sentence", "kb_mrr_hard"), ("easy_co_sentence", "kb_mrr_easy")]:
        dcol = f"delta_{col}_{well_def}"
        if dcol not in sub.columns:
            continue
        vals = sub[dcol].astype(float)
        mean, lo, hi = _bootstrap_ci(vals.to_numpy())
        rows.append(
            {
                "subset": subset,
                "well_trained_definition": well_def,
                "n_seeds": int(len(vals)),
                "mean_delta_kb_mrr": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_kb_falls": int((vals < 0).sum()),
                "frac_kb_falls": float((vals < 0).mean()) if len(vals) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_pair_type_breakdown(paired: pd.DataFrame, well_def: str = WELL_DEF_VAL_F1) -> pd.DataFrame:
    pc = f"pairable_{well_def}"
    sub = paired[paired[pc]]
    rows: list[dict] = []
    for pt, col in [("gene-drug", "kb_mrr_gene_drug"), ("gene-disease", "kb_mrr_gene_disease")]:
        dcol = f"delta_{col}_{well_def}"
        vals = sub[dcol].astype(float)
        mean, lo, hi = _bootstrap_ci(vals.to_numpy())
        rows.append(
            {
                "pair_type": pt,
                "well_trained_definition": well_def,
                "n_seeds": int(len(vals)),
                "mean_delta_kb_mrr": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_kb_falls": int((vals < 0).sum()),
                "frac_kb_falls": float((vals < 0).mean()) if len(vals) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_robustness_table(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in MODELS:
        row: dict[str, Any] = {"model_id": spec.model_id, "short_name": spec.short_name}
        for well_def in WELL_DEFS:
            pc = f"pairable_{well_def}"
            enc = paired[(paired["model_id"] == spec.model_id) & (paired[pc])]
            n = len(enc)
            if n == 0:
                row[f"n_pairable_{well_def}"] = 0
                row[f"frac_erosion_{well_def}"] = np.nan
                continue
            row[f"n_pairable_{well_def}"] = n
            row[f"frac_erosion_{well_def}"] = float(enc[f"erosion_pattern_{well_def}"].mean())
            row[f"mean_delta_benchmark_{well_def}"] = float(
                enc[f"delta_benchmark_f1_{well_def}"].mean()
            )
            row[f"mean_delta_kb_hard_{well_def}"] = float(enc[f"delta_kb_mrr_hard_{well_def}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _r1_seed_noise_note() -> str:
    if not R11_VARIANCE_CSV.exists():
        return "Round 1 variance components unavailable."
    vc = pd.read_csv(R11_VARIANCE_CSV)
    gd = vc[vc["metric"] == "kb_mrr_gene_drug"]
    if gd.empty:
        return ""
    seed_share = float(gd.iloc[0]["seed_variance_share"])
    enc_share = float(gd.iloc[0]["encoder_variance_share"])
    return (
        f"Round 1 found KB gene-drug variance was about {seed_share:.0%} within-encoder seed noise "
        f"and {enc_share:.0%} between encoders, so between-model comparisons had limited power. "
        "Within-seed paired changes across training remove between-seed noise and are the "
        "appropriate design for detecting a training-dynamics effect."
    )


def adjudicate_verdict(
    seed_dist: pd.DataFrame,
    hard_easy: pd.DataFrame,
    pair_type: pd.DataFrame,
    robustness: pd.DataFrame,
    well_def: str = WELL_DEF_VAL_F1,
) -> dict[str, Any]:
    """Explanation 1 vs Explanation 2 overall verdict."""
    pooled = seed_dist[seed_dist["model_id"] == "ALL"]
    if pooled.empty:
        return {"verdict": "insufficient_data", "explanation": "none"}

    frac_erosion = float(pooled.iloc[0]["frac_erosion"])
    n_pairable = int(pooled.iloc[0]["n_seeds_pairable"])
    n_erosion = int(pooled.iloc[0]["n_erosion_benchmark_up_kb_hard_down"])
    mean_d_bench = float(pooled.iloc[0]["mean_delta_benchmark"])
    mean_d_kb = float(pooled.iloc[0]["mean_delta_kb_hard"])
    mean_d_gdis = float(pooled.iloc[0]["mean_delta_kb_gene_disease"])

    hard_row = hard_easy[(hard_easy["subset"] == "hard_cross_sentence") & (hard_easy["well_trained_definition"] == well_def)]
    easy_row = hard_easy[(hard_easy["subset"] == "easy_co_sentence") & (hard_easy["well_trained_definition"] == well_def)]
    gdis_row = pair_type[(pair_type["pair_type"] == "gene-disease") & (pair_type["well_trained_definition"] == well_def)]
    gd_row = pair_type[(pair_type["pair_type"] == "gene-drug") & (pair_type["well_trained_definition"] == well_def)]

    hard_falls = float(hard_row.iloc[0]["frac_kb_falls"]) if not hard_row.empty else np.nan
    gdis_mean = float(gdis_row.iloc[0]["mean_delta_kb_mrr"]) if not gdis_row.empty else np.nan
    gdis_falls = float(gdis_row.iloc[0]["frac_kb_falls"]) if not gdis_row.empty else np.nan

    robust_cols = [f"frac_erosion_{d}" for d in WELL_DEFS if f"frac_erosion_{d}" in robustness.columns]
    robust_erosion = [
        float(robustness[c].mean()) for c in robust_cols if robustness[c].notna().any()
    ]
    robust_across_defs = len(robust_erosion) >= 2 and all(x > 0.4 for x in robust_erosion)

    # Decision tree
    if n_pairable < 20:
        explanation = "insufficient_data"
        narrative = "Too few pairable seed trajectories for a firm verdict."
    elif frac_erosion >= 0.55 and mean_d_kb < -0.005 and mean_d_bench > 0.01:
        if gdis_mean < 0 and gdis_falls >= 0.45:
            explanation = "explanation_1_mechanistic"
            narrative = (
                "Within-model trajectories show benchmark rising while knowledge-base ranking "
                "falls for a majority of seeds, including on the hard cross-sentence subset and "
                "on gene-disease pairs where non-drug chemical pool inflation cannot explain the "
                "drop. This pattern is evidence for a genuine training-dynamics erosion effect "
                "over and above static criterion and pool-composition differences."
            )
        elif hard_falls >= 0.5:
            explanation = "explanation_1_partial"
            narrative = (
                "Within-model benchmark-up / KB-hard-down erosion is common across seeds, "
                "concentrated on the hard subset, but gene-disease erosion is weaker. The data "
                "support a partial training-dynamics effect; pool composition may still contribute "
                "on the gene-drug side."
            )
        else:
            explanation = "mixed"
            narrative = (
                "Some seeds show benchmark-up / KB-down paired changes, but hard-subset erosion "
                "is not dominant. The picture is mixed and does not cleanly support full "
                "generalisation erosion."
            )
    elif mean_d_kb >= -0.002 or frac_erosion < 0.35:
        explanation = "explanation_2_static"
        narrative = (
            "Knowledge-base ranking does not systematically fall within models as the "
            "in-distribution benchmark rises. The negative benchmark–KB association observed "
            "between models is better explained by static differences in inclusion criteria "
            "and evaluation pool construction (including PubTator Chemical breadth), not by "
            "progressive overfitting during training."
        )
    else:
        explanation = "mixed"
        narrative = (
            "Within-model paired changes are heterogeneous across seeds. Neither a robust "
            "training-dynamics erosion story nor a pure static-mismatch story is fully supported."
        )

    if explanation == "explanation_1_mechanistic" and not robust_across_defs:
        narrative += (
            " Note: the erosion pattern is not equally strong under all three well-trained "
            "checkpoint definitions; interpret robustness table before strong causal claims."
        )

    return {
        "verdict": explanation,
        "n_pairable_seeds": n_pairable,
        "n_erosion_seeds": n_erosion,
        "frac_erosion": frac_erosion,
        "mean_delta_benchmark": mean_d_bench,
        "mean_delta_kb_hard": mean_d_kb,
        "mean_delta_kb_gene_disease": mean_d_gdis,
        "hard_subset_frac_kb_falls": hard_falls,
        "gene_disease_frac_kb_falls": gdis_falls,
        "robust_across_well_trained_defs": robust_across_defs,
        "narrative": narrative,
        "power_note": _r1_seed_noise_note(),
    }


def run_adjudication_analysis() -> dict[str, Any]:
    traj = build_trajectory_table()
    traj.to_csv(TRAJECTORY_CSV, index=False)

    paired = build_within_seed_paired_changes(traj)
    paired.to_csv(PAIRED_CHANGES_CSV, index=False)

    seed_dist = build_seed_erosion_distribution(paired, WELL_DEF_VAL_F1)
    seed_dist.to_csv(SEED_DISTRIBUTION_CSV, index=False)

    hard_easy = build_hard_easy_breakdown(paired, WELL_DEF_VAL_F1)
    hard_easy.to_csv(HARD_EASY_CSV, index=False)

    pair_type = build_pair_type_breakdown(paired, WELL_DEF_VAL_F1)
    pair_type.to_csv(PAIR_TYPE_CSV, index=False)

    robustness = build_robustness_table(paired)
    robustness.to_csv(ROBUSTNESS_CSV, index=False)

    verdict = adjudicate_verdict(seed_dist, hard_easy, pair_type, robustness)

    print("\n=== Within-seed paired change summary (epoch 1 -> best val F1) ===")
    pooled = seed_dist[seed_dist["model_id"] == "ALL"]
    if not pooled.empty:
        p = pooled.iloc[0]
        print(
            f"  Pairable seeds: {int(p['n_seeds_pairable'])} | "
            f"erosion (bench up, KB hard down): {int(p['n_erosion_benchmark_up_kb_hard_down'])} "
            f"({float(p['frac_erosion']):.1%})"
        )
        print(
            f"  Mean delta benchmark: {float(p['mean_delta_benchmark']):+.4f} | "
            f"mean delta KB hard: {float(p['mean_delta_kb_hard']):+.4f} | "
            f"mean delta KB gene-disease: {float(p['mean_delta_kb_gene_disease']):+.4f}"
        )

    print("\n=== Hard vs easy (KB MRR change) ===")
    for _, r in hard_easy.iterrows():
        print(
            f"  {r['subset']}: mean delta {r['mean_delta_kb_mrr']:+.4f} "
            f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], "
            f"KB falls in {int(r['n_kb_falls'])}/{int(r['n_seeds'])} seeds"
        )

    print("\n=== Pair type (KB MRR change) ===")
    for _, r in pair_type.iterrows():
        print(
            f"  {r['pair_type']}: mean delta {r['mean_delta_kb_mrr']:+.4f} "
            f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], "
            f"KB falls in {int(r['n_kb_falls'])}/{int(r['n_seeds'])} seeds"
        )

    print("\n=== Robustness across well-trained definitions (mean frac erosion per encoder) ===")
    for well_def in WELL_DEFS:
        col = f"frac_erosion_{well_def}"
        if col in robustness.columns:
            print(f"  {WELL_DEF_LABELS[well_def]}: {robustness[col].mean():.1%}")

    print(f"\n=== ADJUDICATION VERDICT (pooled hard subset): {verdict['verdict']} ===")
    print(verdict["narrative"])
    print(verdict.get("power_note", ""))

    gd_mod = import_module("20_round2_diagnostic.gene_disease_analysis")
    gd_mod.require_cross_metrics(traj)
    gd_results = gd_mod.run_gene_disease_analysis(paired, traj, seed_dist)
    paired = gd_results["paired_extended"]
    paired.to_csv(PAIRED_CHANGES_CSV, index=False)

    return {
        "trajectory": traj,
        "paired": paired,
        "seed_dist": seed_dist,
        "hard_easy": hard_easy,
        "pair_type": pair_type,
        "robustness": robustness,
        "verdict": verdict,
        "gene_disease": gd_results,
    }
