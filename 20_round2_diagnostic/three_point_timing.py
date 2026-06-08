"""Three-point paired timing analysis (CPU post-hoc on epoch_kb_trajectory.csv)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    EPOCH_KB_CACHE,
    FOCUS_MODEL_IDS,
    MODEL_BY_ID,
    OUTPUT_DIR,
    R11_EASY_HARD_CSV,
    R11_PER_RUN_CSV,
    TRAIN_SEEDS,
)
from .matrix_io import load_training_meta

THREE_POINT_CSV = OUTPUT_DIR / "three_point_timing.csv"
THREE_POINT_SUMMARY_CSV = OUTPUT_DIR / "three_point_timing_summary.csv"

# Well-trained definitions for decoupling (only val_f1_best uses validation F1).
WELL_DEF_VAL_F1 = "val_f1_best"
WELL_DEF_LAST = "last_epoch"
WELL_DEF_FIXED5 = "fixed_epoch5"
WELL_DEFS = (WELL_DEF_VAL_F1, WELL_DEF_LAST, WELL_DEF_FIXED5)
WELL_DEF_LABELS = {
    WELL_DEF_VAL_F1: "val_f1-best (benchmark-side criterion)",
    WELL_DEF_LAST: "last saved epoch (end of training)",
    WELL_DEF_FIXED5: "fixed epoch 5 capped at last saved epoch (training-amount criterion)",
}
NON_BENCHMARK_WELL_DEFS = (WELL_DEF_LAST, WELL_DEF_FIXED5)


def _bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        v = float(values[0])
        return v, v, v
    means = [float(rng.choice(values, size=len(values), replace=True).mean()) for _ in range(n)]
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _focus_r1_noise() -> pd.DataFrame:
    """Round 1 seed-level SD for focus encoders (matched scope per axis)."""
    rows: list[dict] = []
    per_run = pd.read_csv(R11_PER_RUN_CSV) if R11_PER_RUN_CSV.exists() else pd.DataFrame()
    eh = pd.read_csv(R11_EASY_HARD_CSV) if R11_EASY_HARD_CSV.exists() else pd.DataFrame()
    for mid in FOCUS_MODEL_IDS:
        bench_sd = np.nan
        kb_gd_sd = np.nan
        kb_gdis_sd = np.nan
        if not per_run.empty:
            sub = per_run[per_run["model_id"] == mid]
            if len(sub) > 1:
                bench_sd = float(sub["benchmark_f1"].std(ddof=1))
                if "kb_mrr_gene_drug" in sub.columns:
                    kb_gd_sd = float(sub["kb_mrr_gene_drug"].std(ddof=1))
                if "kb_mrr_gene_disease" in sub.columns:
                    kb_gdis_sd = float(sub["kb_mrr_gene_disease"].std(ddof=1))
        kb_hard_sd = np.nan
        if not eh.empty:
            hard = eh[(eh["model_id"] == mid) & (eh["subset"] == "hard_cross_sentence")]
            if len(hard) > 1:
                kb_hard_sd = float(hard["mrr"].std(ddof=1))
        rows.append(
            {
                "model_id": mid,
                "short_name": MODEL_BY_ID[mid].short_name,
                "r1_benchmark_f1_sd": bench_sd,
                "r1_kb_hard_mrr_sd": kb_hard_sd,
                "r1_kb_gene_drug_sd": kb_gd_sd,
                "r1_kb_gene_disease_sd": kb_gdis_sd,
            }
        )
    return pd.DataFrame(rows)


def _is_scored_row(row: pd.Series) -> bool:
    if bool(row.get("kb_scored")) and bool(row.get("benchmark_f1_scored")):
        return True
    return pd.notna(row.get("benchmark_f1")) and pd.notna(row.get("kb_mrr_hard"))


def _well_trained_epoch_val_f1(model_id: str, seed: int, sub: pd.DataFrame) -> int:
    meta = load_training_meta(model_id, seed)
    if meta and meta.get("best_epoch_val_f1"):
        return int(meta["best_epoch_val_f1"])
    return int(sub.loc[sub["val_f1"].idxmax(), "epoch"])


def _well_trained_epoch_last(sub: pd.DataFrame) -> int:
    return int(sub["epoch"].max())


def _well_trained_epoch_fixed5(sub: pd.DataFrame) -> int:
    """Fixed training-amount milestone: epoch 5 when saved, else last saved epoch."""
    last = int(sub["epoch"].max())
    return min(5, last)


def _resolve_well_epoch(well_def: str, model_id: str, seed: int, sub: pd.DataFrame) -> int:
    if well_def == WELL_DEF_VAL_F1:
        return _well_trained_epoch_val_f1(model_id, seed, sub)
    if well_def == WELL_DEF_LAST:
        return _well_trained_epoch_last(sub)
    if well_def == WELL_DEF_FIXED5:
        return _well_trained_epoch_fixed5(sub)
    raise ValueError(f"Unknown well-trained definition: {well_def}")


def _milestone_flags(under_ep: int | None, well_ep: int | None, end_ep: int | None) -> str:
    if under_ep is None:
        return "missing_epoch1"
    if well_ep is None:
        return "missing_well_trained"
    if under_ep == well_ep:
        return "collapsed_under_well"
    if end_ep is None:
        return "missing_end"
    return "complete_pairable"


def _row_at(sub: pd.DataFrame, ep: int | None) -> pd.Series | None:
    if ep is None:
        return None
    hit = sub[sub["epoch"] == ep]
    return hit.iloc[0] if not hit.empty else None


def _metric(row: pd.Series | None, col: str) -> float:
    if row is None:
        return np.nan
    return float(row.get(col, np.nan))


def _trajectory_shape_for_seed(
    sub: pd.DataFrame,
    under_ep: int,
    well_ep: int,
    end_ep: int,
) -> dict[str, Any]:
    """Characterise full kb_mrr_hard trajectory independent of milestone choice."""
    sub = sub.sort_values("epoch")
    r_under = _row_at(sub, under_ep)
    if r_under is None or not _is_scored_row(r_under):
        return {"trajectory_scored": False}

    kb_e1 = _metric(r_under, "kb_mrr_hard")
    post = sub[sub["epoch"] > under_ep]
    if post.empty or post["kb_mrr_hard"].isna().all():
        return {"trajectory_scored": False}

    kb_post = post["kb_mrr_hard"].astype(float)
    frac_below = float((kb_post < kb_e1).mean())
    median_post = float(kb_post.median())
    median_post_delta = median_post - kb_e1
    last_kb = _metric(_row_at(sub, end_ep), "kb_mrr_hard")
    well_kb = _metric(_row_at(sub, well_ep), "kb_mrr_hard")
    min_kb = float(kb_post.min())
    min_epochs = post.loc[kb_post == min_kb, "epoch"].tolist()
    val_f1_is_unique_min = well_ep in min_epochs and len(min_epochs) == 1
    val_f1_is_min = well_ep in min_epochs
    rebound_after_val_f1 = (
        not np.isnan(well_kb) and not np.isnan(last_kb) and well_ep != end_ep and last_kb > well_kb
    )
    broadly_below_e1 = median_post_delta < 0 and frac_below >= 0.5
    monotone_ish = last_kb <= kb_e1 and frac_below >= 0.5

    return {
        "trajectory_scored": True,
        "kb_hard_epoch1": kb_e1,
        "frac_post_epoch1_kb_below_e1": frac_below,
        "median_post_epoch1_kb_delta": median_post_delta,
        "kb_hard_at_val_f1": well_kb,
        "kb_hard_at_end": last_kb,
        "val_f1_epoch_is_kb_min": val_f1_is_min,
        "val_f1_epoch_is_unique_kb_min": val_f1_is_unique_min,
        "kb_rebound_after_val_f1": rebound_after_val_f1,
        "trajectory_broadly_below_epoch1": broadly_below_e1,
        "trajectory_monotone_ish": monotone_ish,
    }


def build_three_point_table(traj: pd.DataFrame | None = None) -> pd.DataFrame:
    if traj is None:
        if not EPOCH_KB_CACHE.exists():
            raise FileNotFoundError(f"Missing trajectory cache: {EPOCH_KB_CACHE}")
        traj = pd.read_csv(EPOCH_KB_CACHE)
    pe = traj[traj["source"] == "matrix_per_epoch"].copy()

    rows: list[dict] = []
    for model_id in FOCUS_MODEL_IDS:
        for seed in TRAIN_SEEDS:
            sub = pe[(pe["model_id"] == model_id) & (pe["seed"] == seed)].copy()
            if sub.empty:
                rows.append(
                    {
                        "model_id": model_id,
                        "seed": seed,
                        "milestone_status": "no_trajectory",
                        "pairable": False,
                    }
                )
                continue

            end_ep = int(sub["epoch"].max())
            under_ep = 1 if (sub["epoch"] == 1).any() else None
            well_val_f1 = _well_trained_epoch_val_f1(model_id, seed, sub)

            row: dict[str, Any] = {
                "model_id": model_id,
                "short_name": MODEL_BY_ID[model_id].short_name,
                "seed": seed,
                "epoch_under": under_ep,
                "epoch_end": end_ep,
                "epoch_well_val_f1": well_val_f1,
            }

            # Milestone values at under / val_f1 / end (three-point KB path).
            for label, ep in [("under", under_ep), ("well_val_f1", well_val_f1), ("end", end_ep)]:
                r = _row_at(sub, ep)
                for metric in (
                    "benchmark_f1",
                    "kb_mrr_hard",
                    "kb_mrr_gene_drug",
                    "kb_mrr_gene_disease",
                ):
                    row[f"{metric}_{label}"] = _metric(r, metric)

            if (
                pd.notna(row.get("kb_mrr_hard_under"))
                and pd.notna(row.get("kb_mrr_hard_well_val_f1"))
                and pd.notna(row.get("kb_mrr_hard_end"))
            ):
                row["delta_kb_hard_under_to_well"] = (
                    row["kb_mrr_hard_well_val_f1"] - row["kb_mrr_hard_under"]
                )
                row["delta_kb_hard_well_to_end"] = (
                    row["kb_mrr_hard_end"] - row["kb_mrr_hard_well_val_f1"]
                )
            else:
                row["delta_kb_hard_under_to_well"] = np.nan
                row["delta_kb_hard_well_to_end"] = np.nan

            # Per well-trained definition: paired deltas and pairability.
            for well_def in WELL_DEFS:
                well_ep = _resolve_well_epoch(well_def, model_id, seed, sub)
                r_under = _row_at(sub, under_ep)
                r_well = _row_at(sub, well_ep)
                status = _milestone_flags(under_ep, well_ep, end_ep)
                pairable = status == "complete_pairable"
                if pairable and r_under is not None and r_well is not None:
                    if not (_is_scored_row(r_under) and _is_scored_row(r_well)):
                        status = "unscored_milestones"
                        pairable = False

                suffix = well_def
                row[f"epoch_well_{suffix}"] = well_ep
                row[f"milestone_status_{suffix}"] = status
                row[f"pairable_{suffix}"] = pairable

                if pairable:
                    row[f"delta_benchmark_{suffix}"] = _metric(r_well, "benchmark_f1") - _metric(
                        r_under, "benchmark_f1"
                    )
                    row[f"delta_kb_hard_{suffix}"] = _metric(r_well, "kb_mrr_hard") - _metric(
                        r_under, "kb_mrr_hard"
                    )
                    row[f"delta_kb_gene_drug_{suffix}"] = _metric(r_well, "kb_mrr_gene_drug") - _metric(
                        r_under, "kb_mrr_gene_drug"
                    )
                    row[f"delta_kb_gene_disease_{suffix}"] = _metric(
                        r_well, "kb_mrr_gene_disease"
                    ) - _metric(r_under, "kb_mrr_gene_disease")
                else:
                    for col in (
                        f"delta_benchmark_{suffix}",
                        f"delta_kb_hard_{suffix}",
                        f"delta_kb_gene_drug_{suffix}",
                        f"delta_kb_gene_disease_{suffix}",
                    ):
                        row[col] = np.nan

            # Legacy columns (val_f1-best primary contrast).
            row["epoch_well"] = row["epoch_well_val_f1_best"]
            row["milestone_status"] = row["milestone_status_val_f1_best"]
            row["pairable"] = row["pairable_val_f1_best"]
            row["benchmark_f1_under"] = row.get("benchmark_f1_under", np.nan)
            row["benchmark_f1_well"] = row.get("benchmark_f1_well_val_f1", np.nan)
            row["benchmark_f1_end"] = row.get("benchmark_f1_end", np.nan)
            row["kb_mrr_hard_under"] = row.get("kb_mrr_hard_under", np.nan)
            row["kb_mrr_hard_well"] = row.get("kb_mrr_hard_well_val_f1", np.nan)
            row["kb_mrr_hard_end"] = row.get("kb_mrr_hard_end", np.nan)
            row["delta_benchmark"] = row.get("delta_benchmark_val_f1_best", np.nan)
            row["delta_kb_hard"] = row.get("delta_kb_hard_val_f1_best", np.nan)
            if pd.notna(row.get("benchmark_f1_end")) and pd.notna(row.get("benchmark_f1_well")):
                row["delta_benchmark_well_to_end"] = row["benchmark_f1_end"] - row["benchmark_f1_well"]
            else:
                row["delta_benchmark_well_to_end"] = np.nan
            row["delta_kb_hard_well_to_end"] = row.get("delta_kb_hard_well_to_end", np.nan)

            shape = _trajectory_shape_for_seed(sub, under_ep or 1, well_val_f1, end_ep)
            row.update(shape)

            rows.append(row)

    return pd.DataFrame(rows)


def _axis_clears_noise(mean: float, lo: float, hi: float, noise_sd: float) -> bool:
    if np.isnan(noise_sd) or np.isnan(mean):
        return False
    band = float(noise_sd)
    if band <= 0:
        return abs(mean) > 0
    return not (lo >= -band and hi <= band)


def _axes_diverge(
    mean_b: float,
    mean_k: float,
    lo_b: float,
    hi_b: float,
    lo_k: float,
    hi_k: float,
    bench_sd: float,
    kb_sd: float,
) -> bool:
    if np.isnan(mean_b) or np.isnan(mean_k):
        return False
    opposite_sign = mean_b * mean_k < 0
    diff = abs(mean_b - mean_k)
    threshold = 0.5 * (float(bench_sd if not np.isnan(bench_sd) else 0) + float(kb_sd if not np.isnan(kb_sd) else 0))
    clearly_different = diff > threshold if threshold > 0 else diff > 0.01
    nonoverlap = (hi_b < lo_k) or (hi_k < lo_b)
    return opposite_sign or clearly_different or nonoverlap


def _divergence_verdict(
    mean_b: float,
    mean_k: float,
    lo_b: float,
    hi_b: float,
    lo_k: float,
    hi_k: float,
    bench_sd: float,
    kb_sd: float,
) -> tuple[bool, bool, bool, str]:
    bench_clears = _axis_clears_noise(mean_b, lo_b, hi_b, bench_sd)
    kb_clears = _axis_clears_noise(mean_k, lo_k, hi_k, kb_sd)
    diverge = _axes_diverge(mean_b, mean_k, lo_b, hi_b, lo_k, hi_k, bench_sd, kb_sd)
    benchmark_rises = mean_b > 0
    kb_hard_falls = mean_k < 0
    pattern = benchmark_rises and kb_hard_falls
    verdict = (
        "two_axes_diverge_along_training"
        if pattern and (bench_clears or kb_clears) and diverge
        else "training_effect_within_seed_noise_or_axes_aligned"
    )
    return bench_clears, kb_clears, diverge, verdict


def _summarise_definition(
    three_pt: pd.DataFrame,
    mid: str,
    well_def: str,
    n_row: pd.Series,
) -> dict[str, Any]:
    pair_col = f"pairable_{well_def}"
    sub_all = three_pt[three_pt["model_id"] == mid]
    pair = sub_all[sub_all[pair_col]].copy()
    n_excluded = int(len(sub_all) - len(pair))
    excluded_reasons = (
        sub_all[~sub_all[pair_col]][f"milestone_status_{well_def}"].value_counts().to_dict()
        if n_excluded
        else {}
    )

    bench_d = pair[f"delta_benchmark_{well_def}"].to_numpy(dtype=float)
    kb_d = pair[f"delta_kb_hard_{well_def}"].to_numpy(dtype=float)
    gd_d = pair[f"delta_kb_gene_drug_{well_def}"].to_numpy(dtype=float)
    gdis_d = pair[f"delta_kb_gene_disease_{well_def}"].to_numpy(dtype=float)

    mean_b, lo_b, hi_b = _bootstrap_ci(bench_d)
    mean_k, lo_k, hi_k = _bootstrap_ci(kb_d)
    mean_gd, lo_gd, hi_gd = _bootstrap_ci(gd_d)
    mean_gdis, lo_gdis, hi_gdis = _bootstrap_ci(gdis_d)

    bench_sd = float(n_row["r1_benchmark_f1_sd"])
    kb_sd = float(n_row["r1_kb_hard_mrr_sd"])
    gd_sd = float(n_row.get("r1_kb_gene_drug_sd", np.nan))
    gdis_sd = float(n_row.get("r1_kb_gene_disease_sd", np.nan))

    bench_clears, kb_clears, diverge, verdict = _divergence_verdict(
        mean_b, mean_k, lo_b, hi_b, lo_k, hi_k, bench_sd, kb_sd
    )

    return {
        "well_trained_definition": well_def,
        "n_seeds_pairable": len(pair),
        "n_seeds_excluded": n_excluded,
        "excluded_reasons": json.dumps(excluded_reasons),
        f"mean_delta_benchmark_{well_def}": mean_b,
        f"delta_benchmark_ci_lo_{well_def}": lo_b,
        f"delta_benchmark_ci_hi_{well_def}": hi_b,
        f"mean_delta_kb_hard_{well_def}": mean_k,
        f"delta_kb_hard_ci_lo_{well_def}": lo_k,
        f"delta_kb_hard_ci_hi_{well_def}": hi_k,
        f"mean_delta_kb_gene_drug_{well_def}": mean_gd,
        f"delta_kb_gene_drug_ci_lo_{well_def}": lo_gd,
        f"delta_kb_gene_drug_ci_hi_{well_def}": hi_gd,
        f"mean_delta_kb_gene_disease_{well_def}": mean_gdis,
        f"delta_kb_gene_disease_ci_lo_{well_def}": lo_gdis,
        f"delta_kb_gene_disease_ci_hi_{well_def}": hi_gdis,
        f"benchmark_clears_noise_{well_def}": bench_clears,
        f"kb_clears_noise_{well_def}": kb_clears,
        f"axes_diverge_{well_def}": diverge,
        f"benchmark_rises_{well_def}": mean_b > 0 if not np.isnan(mean_b) else False,
        f"kb_hard_falls_{well_def}": mean_k < 0 if not np.isnan(mean_k) else False,
        f"verdict_{well_def}": verdict,
        f"kb_gene_drug_clears_noise_{well_def}": _axis_clears_noise(mean_gd, lo_gd, hi_gd, gd_sd),
        f"kb_gene_disease_clears_noise_{well_def}": _axis_clears_noise(
            mean_gdis, lo_gdis, hi_gdis, gdis_sd
        ),
    }


def _trajectory_shape_summary(three_pt: pd.DataFrame, mid: str) -> dict[str, Any]:
    sub = three_pt[
        (three_pt["model_id"] == mid)
        & (three_pt["trajectory_scored"].fillna(False).astype(bool))
    ]
    n = len(sub)
    if n == 0:
        return {
            "trajectory_n_scored": 0,
            "trajectory_frac_seeds_broadly_below_e1": np.nan,
            "trajectory_frac_seeds_monotone_ish": np.nan,
            "trajectory_frac_val_f1_unique_kb_min": np.nan,
            "trajectory_frac_kb_rebound_after_val_f1": np.nan,
            "trajectory_shape_reading": "no scored trajectories",
        }
    broadly = float(sub["trajectory_broadly_below_epoch1"].mean())
    monotone = float(sub["trajectory_monotone_ish"].mean())
    unique_min = float(sub["val_f1_epoch_is_unique_kb_min"].mean())
    rebound = float(sub["kb_rebound_after_val_f1"].mean())
    mean_frac_below = float(sub["frac_post_epoch1_kb_below_e1"].mean())

    if broadly >= 0.5 and mean_frac_below >= 0.5:
        reading = (
            "KB hard is generally below epoch-1 across post-epoch-1 checkpoints, "
            "not only at val_f1-best"
        )
    elif unique_min >= 0.5:
        reading = "KB hard is lowest specifically at val_f1-best for most seeds"
    elif rebound >= 0.5:
        reading = "KB hard often rebounds after val_f1-best toward last epoch"
    else:
        reading = "mixed trajectory shape across seeds"

    return {
        "trajectory_n_scored": n,
        "trajectory_frac_seeds_broadly_below_e1": broadly,
        "trajectory_frac_seeds_monotone_ish": monotone,
        "trajectory_frac_val_f1_unique_kb_min": unique_min,
        "trajectory_frac_kb_rebound_after_val_f1": rebound,
        "trajectory_mean_frac_post_epochs_below_e1": mean_frac_below,
        "trajectory_shape_reading": reading,
    }


def _three_point_kb_reading(three_pt: pd.DataFrame, mid: str) -> dict[str, Any]:
    pair = three_pt[(three_pt["model_id"] == mid) & (three_pt["pairable_val_f1_best"])]
    if pair.empty:
        return {
            "mean_delta_kb_under_to_well": np.nan,
            "mean_delta_kb_well_to_end": np.nan,
            "kb_keeps_falling_after_val_f1": False,
            "three_point_kb_reading": "insufficient pairable seeds",
        }
    d1 = pair["delta_kb_hard_under_to_well"].astype(float)
    d2 = pair["delta_kb_hard_well_to_end"].astype(float)
    m1, _, _ = _bootstrap_ci(d1.to_numpy())
    m2, _, _ = _bootstrap_ci(d2.to_numpy())
    keeps_falling = m2 < 0 and not np.isnan(m2)
    if m1 < 0 and m2 < 0:
        reading = "KB hard falls from epoch 1 to val_f1-best and continues falling to last epoch"
    elif m1 < 0 and m2 > 0:
        reading = "KB hard falls to val_f1-best then rebounds toward last epoch"
    elif m1 > 0:
        reading = "KB hard does not fall from epoch 1 to val_f1-best"
    else:
        reading = "ambiguous three-point KB movement"
    return {
        "mean_delta_kb_under_to_well": m1,
        "mean_delta_kb_well_to_end": m2,
        "kb_keeps_falling_after_val_f1": keeps_falling,
        "three_point_kb_reading": reading,
    }


def build_summary(three_pt: pd.DataFrame) -> pd.DataFrame:
    noise = _focus_r1_noise()
    rows: list[dict] = []

    for mid in FOCUS_MODEL_IDS:
        sub_all = three_pt[three_pt["model_id"] == mid]
        n_row = noise[noise["model_id"] == mid].iloc[0]

        row: dict[str, Any] = {
            "model_id": mid,
            "short_name": n_row["short_name"],
            "n_seeds_total": len(sub_all),
            "r1_benchmark_f1_sd": float(n_row["r1_benchmark_f1_sd"]),
            "r1_kb_hard_mrr_sd": float(n_row["r1_kb_hard_mrr_sd"]),
            "r1_kb_gene_drug_sd": float(n_row.get("r1_kb_gene_drug_sd", np.nan)),
            "r1_kb_gene_disease_sd": float(n_row.get("r1_kb_gene_disease_sd", np.nan)),
        }

        defs_summary: dict[str, dict] = {}
        for well_def in WELL_DEFS:
            defs_summary[well_def] = _summarise_definition(three_pt, mid, well_def, n_row)
            row.update(defs_summary[well_def])

        vf = defs_summary[WELL_DEF_VAL_F1]
        row.update(
            {
                "n_seeds_pairable": vf["n_seeds_pairable"],
                "n_seeds_excluded": vf["n_seeds_excluded"],
                "excluded_reasons": vf["excluded_reasons"],
                "mean_delta_benchmark": vf[f"mean_delta_benchmark_{WELL_DEF_VAL_F1}"],
                "delta_benchmark_ci_lo": vf[f"delta_benchmark_ci_lo_{WELL_DEF_VAL_F1}"],
                "delta_benchmark_ci_hi": vf[f"delta_benchmark_ci_hi_{WELL_DEF_VAL_F1}"],
                "mean_delta_kb_hard": vf[f"mean_delta_kb_hard_{WELL_DEF_VAL_F1}"],
                "delta_kb_hard_ci_lo": vf[f"delta_kb_hard_ci_lo_{WELL_DEF_VAL_F1}"],
                "delta_kb_hard_ci_hi": vf[f"delta_kb_hard_ci_hi_{WELL_DEF_VAL_F1}"],
                "benchmark_clears_noise": vf[f"benchmark_clears_noise_{WELL_DEF_VAL_F1}"],
                "kb_clears_noise": vf[f"kb_clears_noise_{WELL_DEF_VAL_F1}"],
                "axes_diverge": vf[f"axes_diverge_{WELL_DEF_VAL_F1}"],
                "verdict": vf[f"verdict_{WELL_DEF_VAL_F1}"],
            }
        )

        row.update(_trajectory_shape_summary(three_pt, mid))
        row.update(_three_point_kb_reading(three_pt, mid))
        row.update(_robustness_fields(row))
        rows.append(row)

    return pd.DataFrame(rows)


def _robustness_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Encoder-level robustness to selection criterion."""
    non_bench_diverge = all(
        row.get(f"verdict_{d}") == "two_axes_diverge_along_training" for d in NON_BENCHMARK_WELL_DEFS
    )
    val_f1_diverge = row.get(f"verdict_{WELL_DEF_VAL_F1}") == "two_axes_diverge_along_training"
    traj_supports = (
        float(row.get("trajectory_frac_seeds_broadly_below_e1", 0) or 0) >= 0.5
        and float(row.get("trajectory_mean_frac_post_epochs_below_e1", 0) or 0) >= 0.5
    )
    selection_artefact = (
        val_f1_diverge
        and not non_bench_diverge
        and float(row.get("trajectory_frac_val_f1_unique_kb_min", 0) or 0) >= 0.5
    )
    robust = val_f1_diverge and non_bench_diverge and traj_supports

    if robust:
        verdict = "divergence_robust_to_selection_criterion"
    elif selection_artefact or (val_f1_diverge and not non_bench_diverge):
        verdict = "divergence_only_under_val_f1_selection"
    elif val_f1_diverge and not traj_supports:
        verdict = "divergence_val_f1_only_trajectory_mixed"
    else:
        verdict = "no_robust_divergence"

    return {
        "non_benchmark_defs_diverge": non_bench_diverge,
        "trajectory_supports_training_dynamic": traj_supports,
        "selection_artefact_likely": selection_artefact,
        "robustness_verdict": verdict,
    }


def overall_verdict(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "insufficient_data"

    robust_count = int((summary["robustness_verdict"] == "divergence_robust_to_selection_criterion").sum())
    val_f1_only = int(
        summary["robustness_verdict"].isin(
            ("divergence_only_under_val_f1_selection", "divergence_val_f1_only_trajectory_mixed")
        ).sum()
    )
    any_val_f1 = int((summary[f"verdict_{WELL_DEF_VAL_F1}"] == "two_axes_diverge_along_training").sum())

    if robust_count >= 2:
        return "round2_on_training_amount_may_be_informative_robust"
    if any_val_f1 >= 2 and val_f1_only >= 2:
        return "round2_on_training_amount_softened_selection_confound"
    if any_val_f1 >= 1:
        return "round2_on_training_amount_may_be_informative"
    return "round2_on_training_amount_likely_null"


def run_three_point_timing() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    three_pt = build_three_point_table()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    three_pt.to_csv(THREE_POINT_CSV, index=False)

    summary = build_summary(three_pt)
    summary.to_csv(THREE_POINT_SUMMARY_CSV, index=False)

    overall = overall_verdict(summary)

    print("\n=== Three-point paired timing (selection-criterion decoupling) ===")
    print(f"  Trajectory rows read: {len(pd.read_csv(EPOCH_KB_CACHE))}")
    print(f"  Fixed-epoch rule: epoch 5 when saved, else last saved epoch (no val_f1)")
    print(f"  Well-trained definitions: {', '.join(WELL_DEFS)}")

    for well_def in WELL_DEFS:
        pc = f"pairable_{well_def}"
        print(f"\n  Pairable under {WELL_DEF_LABELS[well_def]}: {int(three_pt[pc].sum())} / {len(three_pt)}")

    for _, r in summary.iterrows():
        print(f"\n  === {r['short_name']} ===")
        for well_def in WELL_DEFS:
            n_p = int(r["n_seeds_pairable"]) if well_def == WELL_DEF_VAL_F1 else int(
                three_pt[(three_pt["model_id"] == r["model_id"]) & (three_pt[f"pairable_{well_def}"])].shape[0]
            )
            print(f"\n  [{WELL_DEF_LABELS[well_def]}] pairable={n_p}")
            print(
                f"    delta_benchmark {r[f'mean_delta_benchmark_{well_def}']:.4f} "
                f"[{r[f'delta_benchmark_ci_lo_{well_def}']:.4f}, {r[f'delta_benchmark_ci_hi_{well_def}']:.4f}] "
                f"vs R1 SD {r['r1_benchmark_f1_sd']:.4f} "
                f"clears={r[f'benchmark_clears_noise_{well_def}']} rises={r[f'benchmark_rises_{well_def}']}"
            )
            print(
                f"    delta_kb_hard   {r[f'mean_delta_kb_hard_{well_def}']:.4f} "
                f"[{r[f'delta_kb_hard_ci_lo_{well_def}']:.4f}, {r[f'delta_kb_hard_ci_hi_{well_def}']:.4f}] "
                f"vs R1 SD {r['r1_kb_hard_mrr_sd']:.4f} "
                f"clears={r[f'kb_clears_noise_{well_def}']} falls={r[f'kb_hard_falls_{well_def}']}"
            )
            print(f"    axes_diverge={r[f'axes_diverge_{well_def}']} verdict={r[f'verdict_{well_def}']}")

        print(
            f"\n  Pool-level (val_f1-best minus epoch 1): "
            f"gene-drug delta {r[f'mean_delta_kb_gene_drug_{WELL_DEF_VAL_F1}']:.4f}, "
            f"gene-disease delta {r[f'mean_delta_kb_gene_disease_{WELL_DEF_VAL_F1}']:.4f}"
        )
        print(
            f"  Three-point KB hard (epoch 1 -> val_f1-best -> last): "
            f"under-to-well {r['mean_delta_kb_under_to_well']:.4f}, "
            f"well-to-end {r['mean_delta_kb_well_to_end']:.4f}; "
            f"{r['three_point_kb_reading']}"
        )
        print(
            f"  Trajectory shape: {r['trajectory_shape_reading']} "
            f"(broadly below e1: {r['trajectory_frac_seeds_broadly_below_e1']:.0%} seeds, "
            f"val_f1 unique min: {r['trajectory_frac_val_f1_unique_kb_min']:.0%}, "
            f"rebound after val_f1: {r['trajectory_frac_kb_rebound_after_val_f1']:.0%})"
        )
        print(f"  Robustness verdict: {r['robustness_verdict']}")

    print(f"\n  Overall Round 2 recommendation: {overall}")
    return three_pt, summary, overall


def dry_trace() -> bool:
    """Read existing CSV, extract milestones, compute deltas; tolerate unscored rows."""
    try:
        three_pt = build_three_point_table()
        summary = build_summary(three_pt)
        ok = len(three_pt) == len(FOCUS_MODEL_IDS) * len(TRAIN_SEEDS)
        for well_def in WELL_DEFS:
            n = int(three_pt[f"pairable_{well_def}"].sum())
            print(f"[dry trace] {well_def}: pairable={n}/{len(three_pt)}")
        print(
            f"[dry trace] rows={len(three_pt)} summary={len(summary)} "
            f"overall={overall_verdict(summary)} ok={ok}"
        )
        return ok
    except Exception as exc:
        print(f"[dry trace] FAIL: {exc}")
        import traceback

        traceback.print_exc()
        return False
