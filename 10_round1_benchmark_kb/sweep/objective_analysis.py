"""Objective checkpoint-criterion and learning-rate analysis (no retraining, no KB)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .config import OUTPUT_DIR, SWEEP_RESULTS_DIR, SweepRun, all_runs

SPREAD_THRESHOLD = 0.05
REFERENCE_LR = 5e-6
STRONG_MODEL = "pubmedbert_base"
WEAK_MODEL = "distilbert_base"
MID_MODEL = "roberta_base"


def _artifact_note() -> str:
    return (
        "Per-epoch **validation** metrics (val_loss, val_f1) are logged for every epoch. "
        "Only the val_loss-best and val_f1-best **checkpoints** were saved (not every epoch). "
        "BioRED test benchmark F1 is therefore available for those two checkpoints only — "
        "not for intermediate epochs."
    )


def load_sweep_results() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load completed runs; return dataframe and list of missing run_ids."""
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    limitations: list[str] = []

    for run in all_runs():
        path = run.result_path()
        if not path.exists():
            missing.append(run.run_id)
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        loss_ckpt = Path(d["checkpoint_val_loss"])
        f1_ckpt = Path(d["checkpoint_val_f1"])
        if not loss_ckpt.exists():
            limitations.append(f"{run.run_id}: missing val_loss checkpoint")
        if not f1_ckpt.exists():
            limitations.append(f"{run.run_id}: missing val_f1 checkpoint")

        bf1_loss = d.get("benchmark_f1_val_loss_ckpt")
        bf1_f1 = d.get("benchmark_f1_val_f1_ckpt")
        if bf1_loss is None or bf1_f1 is None:
            limitations.append(f"{run.run_id}: missing benchmark F1 for one or both checkpoints")

        rows.append(
            {
                "run_id": d["run_id"],
                "model_id": d["model_id"],
                "short_name": d["short_name"],
                "lr": float(d["lr"]),
                "warmup_label": d["warmup_label"],
                "best_epoch_val_loss": int(d["best_epoch_by_val_loss"]),
                "best_epoch_val_f1": int(d["best_epoch_by_val_f1"]),
                "best_val_f1": float(d["best_val_f1"]),
                "benchmark_f1_val_loss_ckpt": float(bf1_loss) if bf1_loss is not None else np.nan,
                "benchmark_f1_val_f1_ckpt": float(bf1_f1) if bf1_f1 is not None else np.nan,
                "benchmark_f1_delta_f1_minus_loss": (
                    float(bf1_f1) - float(bf1_loss) if bf1_loss is not None and bf1_f1 is not None else np.nan
                ),
                "selection_disagrees": bool(d["selection_disagrees"]),
                "same_checkpoint_paths": str(loss_ckpt) == str(f1_ckpt),
                "checkpoint_limitation": "; ".join(
                    x for x in [
                        None if loss_ckpt.exists() else "no val_loss ckpt",
                        None if f1_ckpt.exists() else "no val_f1 ckpt",
                        None if bf1_loss is not None else "no bench F1 (loss)",
                        None if bf1_f1 is not None else "no bench F1 (f1)",
                    ] if x
                ) or "",
            }
        )

    df = pd.DataFrame(rows)
    return df, missing, limitations


def analysis1_criterion_comparison(df: pd.DataFrame) -> dict[str, Any]:
    """Paired comparison of benchmark F1 under val_loss vs val_f1 checkpoint selection."""
    valid = df.dropna(subset=["benchmark_f1_val_loss_ckpt", "benchmark_f1_val_f1_ckpt"])
    loss = valid["benchmark_f1_val_loss_ckpt"].values
    f1 = valid["benchmark_f1_val_f1_ckpt"].values
    delta = valid["benchmark_f1_delta_f1_minus_loss"].values

    wilcoxon_stat, wilcoxon_p = wilcoxon(f1, loss) if len(valid) >= 1 else (np.nan, np.nan)

    by_model = []
    for model_id, sub in valid.groupby("model_id"):
        wins_f1 = int((sub["benchmark_f1_delta_f1_minus_loss"] > 0).sum())
        wins_loss = int((sub["benchmark_f1_delta_f1_minus_loss"] < 0).sum())
        ties = int((sub["benchmark_f1_delta_f1_minus_loss"] == 0).sum())
        by_model.append(
            {
                "model_id": model_id,
                "short_name": sub["short_name"].iloc[0],
                "n_runs": len(sub),
                "val_f1_ckpt_higher_count": wins_f1,
                "val_loss_ckpt_higher_count": wins_loss,
                "ties": ties,
                "mean_benchmark_f1_val_loss": float(sub["benchmark_f1_val_loss_ckpt"].mean()),
                "mean_benchmark_f1_val_f1": float(sub["benchmark_f1_val_f1_ckpt"].mean()),
                "mean_delta_f1_minus_loss": float(sub["benchmark_f1_delta_f1_minus_loss"].mean()),
                "median_delta_f1_minus_loss": float(sub["benchmark_f1_delta_f1_minus_loss"].median()),
            }
        )

    per_run = valid[
        [
            "run_id",
            "short_name",
            "lr",
            "warmup_label",
            "best_epoch_val_loss",
            "best_epoch_val_f1",
            "benchmark_f1_val_loss_ckpt",
            "benchmark_f1_val_f1_ckpt",
            "benchmark_f1_delta_f1_minus_loss",
            "checkpoint_limitation",
        ]
    ].copy()

    return {
        "n_runs": len(valid),
        "mean_benchmark_f1_val_loss": float(np.mean(loss)),
        "median_benchmark_f1_val_loss": float(np.median(loss)),
        "mean_benchmark_f1_val_f1": float(np.mean(f1)),
        "median_benchmark_f1_val_f1": float(np.median(f1)),
        "mean_delta_f1_minus_loss": float(np.mean(delta)),
        "median_delta_f1_minus_loss": float(np.median(delta)),
        "val_f1_ckpt_higher_count": int((delta > 0).sum()),
        "val_loss_ckpt_higher_count": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
        "wilcoxon_statistic": float(wilcoxon_stat) if not np.isnan(wilcoxon_stat) else None,
        "wilcoxon_pvalue": float(wilcoxon_p) if not np.isnan(wilcoxon_p) else None,
        "by_model": pd.DataFrame(by_model),
        "per_run": per_run,
    }


def _spread_table(df: pd.DataFrame, f1_col: str, epoch_col: str, criterion: str) -> pd.DataFrame:
    rows = []
    for (lr, warmup_label), sub in df.groupby(["lr", "warmup_label"]):
        v = sub[f1_col].astype(float)
        rows.append(
            {
                "criterion": criterion,
                "lr": lr,
                "warmup_label": warmup_label,
                "benchmark_f1_min": float(v.min()),
                "benchmark_f1_max": float(v.max()),
                "benchmark_f1_spread": float(v.max() - v.min()),
                "benchmark_f1_mean": float(v.mean()),
                "benchmark_f1_median": float(v.median()),
                "mean_best_epoch": float(sub[epoch_col].mean()),
                "median_best_epoch": float(sub[epoch_col].median()),
                "n_runs": len(sub),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["criterion", "benchmark_f1_spread"], ascending=[True, False])


def analysis2_lr_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-architecture spread per lr×warmup under each checkpoint criterion."""
    loss_tbl = _spread_table(
        df, "benchmark_f1_val_loss_ckpt", "best_epoch_val_loss", "val_loss"
    )
    f1_tbl = _spread_table(
        df, "benchmark_f1_val_f1_ckpt", "best_epoch_val_f1", "val_f1"
    )
    return pd.concat([loss_tbl, f1_tbl], ignore_index=True)


def analysis3_gradient_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose spread changes vs lr=5e-6 reference: strong-model lift vs weak-model drop.
    Computed separately under each checkpoint criterion.
    """
    rows: list[dict[str, Any]] = []
    for criterion, col in [("val_loss", "benchmark_f1_val_loss_ckpt"), ("val_f1", "benchmark_f1_val_f1_ckpt")]:
        ref = df[df["lr"] == REFERENCE_LR].set_index(["model_id", "warmup_label"])[col]
        for (lr, warmup_label), sub in df.groupby(["lr", "warmup_label"]):
            spread = float(sub[col].max() - sub[col].min())
            ref_rows = []
            ref_spread = np.nan
            try:
                pub_ref = float(ref.loc[(STRONG_MODEL, warmup_label)])
                dist_ref = float(ref.loc[(WEAK_MODEL, warmup_label)])
                rob_ref = float(ref.loc[(MID_MODEL, warmup_label)])
                ref_spread = float(
                    max(pub_ref, rob_ref, dist_ref) - min(pub_ref, rob_ref, dist_ref)
                )
            except KeyError:
                pub_ref = dist_ref = rob_ref = np.nan

            pub = float(sub.loc[sub["model_id"] == STRONG_MODEL, col].iloc[0])
            dist = float(sub.loc[sub["model_id"] == WEAK_MODEL, col].iloc[0])
            rob = float(sub.loc[sub["model_id"] == MID_MODEL, col].iloc[0])

            pub_delta = pub - pub_ref if not np.isnan(pub_ref) else np.nan
            dist_delta = dist - dist_ref if not np.isnan(dist_ref) else np.nan
            rob_delta = rob - rob_ref if not np.isnan(rob_ref) else np.nan
            spread_delta = spread - ref_spread if not np.isnan(ref_spread) else np.nan

            # Positive weak_drop = weak model scored lower than at 5e-6
            weak_drop = -dist_delta if not np.isnan(dist_delta) else np.nan
            strong_lift = pub_delta if not np.isnan(pub_delta) else np.nan

            if not np.isnan(spread_delta) and spread_delta > 0:
                if not np.isnan(weak_drop) and not np.isnan(strong_lift):
                    if weak_drop > strong_lift:
                        driver = "weak_model_degradation"
                    elif strong_lift > weak_drop:
                        driver = "strong_model_improvement"
                    else:
                        driver = "mixed_equal"
                else:
                    driver = "unknown"
            elif not np.isnan(spread_delta) and spread_delta <= 0:
                driver = "spread_not_widening"
            else:
                driver = "no_reference"

            rows.append(
                {
                    "criterion": criterion,
                    "lr": lr,
                    "warmup_label": warmup_label,
                    "benchmark_f1_spread": spread,
                    "spread_vs_5e6_reference": spread_delta,
                    f"{STRONG_MODEL}_f1": pub,
                    f"{WEAK_MODEL}_f1": dist,
                    f"{MID_MODEL}_f1": rob,
                    f"{STRONG_MODEL}_delta_vs_5e6": pub_delta,
                    f"{WEAK_MODEL}_delta_vs_5e6": dist_delta,
                    f"{MID_MODEL}_delta_vs_5e6": rob_delta,
                    "strong_lift": strong_lift,
                    "weak_drop": weak_drop,
                    "spread_driver": driver,
                }
            )

    return pd.DataFrame(rows)


def analysis4_recommendation(
    df: pd.DataFrame,
    a1: dict[str, Any],
    a2: pd.DataFrame,
    a3: pd.DataFrame,
) -> dict[str, Any]:
    """Data-driven recommendation from analyses 1–3."""
    # --- Criterion ---
    f1_better_count = a1["val_f1_ckpt_higher_count"]
    loss_better_count = a1["val_loss_ckpt_higher_count"]
    criterion_rec = "val_f1" if a1["mean_benchmark_f1_val_f1"] >= a1["mean_benchmark_f1_val_loss"] else "val_loss"
    if a1["wilcoxon_pvalue"] is not None and a1["wilcoxon_pvalue"] < 0.05:
        criterion_rec = "val_f1" if a1["mean_benchmark_f1_val_f1"] > a1["mean_benchmark_f1_val_loss"] else "val_loss"

    by_model = a1["by_model"]
    f1_helps_all = all(row["mean_delta_f1_minus_loss"] >= 0 for _, row in by_model.iterrows())
    f1_hurts_strong = float(
        by_model.loc[by_model["model_id"] == STRONG_MODEL, "mean_delta_f1_minus_loss"].iloc[0]
    ) < 0

    # --- LR under each criterion ---
    lr_candidates: list[dict[str, Any]] = []
    for criterion in ("val_loss", "val_f1"):
        sub = a2[a2["criterion"] == criterion].copy()
        sub = sub[sub["benchmark_f1_spread"] >= SPREAD_THRESHOLD]
        for _, row in sub.iterrows():
            a3_row = a3[
                (a3["criterion"] == criterion)
                & (a3["lr"] == row["lr"])
                & (a3["warmup_label"] == row["warmup_label"])
            ]
            driver = a3_row["spread_driver"].iloc[0] if len(a3_row) else "unknown"
            lr_candidates.append(
                {
                    "criterion": criterion,
                    "lr": row["lr"],
                    "warmup_label": row["warmup_label"],
                    "spread": row["benchmark_f1_spread"],
                    "mean_f1": row["benchmark_f1_mean"],
                    "median_f1": row["benchmark_f1_median"],
                    "mean_best_epoch": row["mean_best_epoch"],
                    "spread_driver": driver,
                    "strong_lift": float(a3_row["strong_lift"].iloc[0]) if len(a3_row) else np.nan,
                    "weak_drop": float(a3_row["weak_drop"].iloc[0]) if len(a3_row) else np.nan,
                }
            )

    cand_df = pd.DataFrame(lr_candidates)
    rec: dict[str, Any] = {
        "spread_threshold": SPREAD_THRESHOLD,
        "recommended_criterion": criterion_rec,
        "criterion_rationale": {
            "mean_f1_val_loss": a1["mean_benchmark_f1_val_loss"],
            "mean_f1_val_f1": a1["mean_benchmark_f1_val_f1"],
            "val_f1_wins": f1_better_count,
            "val_loss_wins": loss_better_count,
            "wilcoxon_p": a1["wilcoxon_pvalue"],
            "f1_helps_all_architectures": f1_helps_all,
            "f1_hurts_pubmedbert_on_average": f1_hurts_strong,
        },
    }

    if cand_df.empty:
        rec["recommended_lr"] = None
        rec["recommended_warmup"] = None
        rec["note"] = "No lr×warmup setting meets spread threshold under either criterion."
        return rec

    # Prefer recommended criterion; rank by spread then mean F1; penalize weak_model_degradation driver
    primary = cand_df[cand_df["criterion"] == criterion_rec].copy()
    if primary.empty:
        primary = cand_df.copy()

    primary["degradation_penalty"] = (primary["spread_driver"] == "weak_model_degradation").astype(int)
    primary = primary.sort_values(
        ["degradation_penalty", "spread", "mean_f1"],
        ascending=[True, False, False],
    )
    primary = primary.drop(columns=["degradation_penalty"])
    best = primary.iloc[0]
    second = primary.iloc[1] if len(primary) > 1 else None

    rec["recommended_lr"] = float(best["lr"])
    rec["recommended_warmup"] = str(best["warmup_label"])
    rec["recommended_under_criterion"] = str(best["criterion"])
    rec["recommended_spread"] = float(best["spread"])
    rec["recommended_mean_f1"] = float(best["mean_f1"])
    rec["recommended_mean_best_epoch"] = float(best["mean_best_epoch"])
    rec["recommended_spread_driver"] = str(best["spread_driver"])
    rec["recommended_strong_lift"] = float(best["strong_lift"]) if not np.isnan(best["strong_lift"]) else None
    rec["recommended_weak_drop"] = float(best["weak_drop"]) if not np.isnan(best["weak_drop"]) else None

    if second is not None:
        rec["runner_up"] = {
            "lr": float(second["lr"]),
            "warmup_label": str(second["warmup_label"]),
            "criterion": str(second["criterion"]),
            "spread": float(second["spread"]),
            "mean_f1": float(second["mean_f1"]),
            "spread_driver": str(second["spread_driver"]),
            "spread_gap_vs_best": float(best["spread"] - second["spread"]),
            "mean_f1_gap_vs_best": float(best["mean_f1"] - second["mean_f1"]),
        }

    # Stability under val_f1 selection
    rec["stability"] = {
        "mean_best_epoch_val_loss": float(df["best_epoch_val_loss"].mean()),
        "mean_best_epoch_val_f1": float(df["best_epoch_val_f1"].mean()),
        "median_best_epoch_val_f1": float(df["best_epoch_val_f1"].median()),
        "pct_runs_best_f1_epoch_gt1": float((df["best_epoch_val_f1"] > 1).mean()),
        "pct_runs_best_loss_epoch_eq1": float((df["best_epoch_val_loss"] == 1).mean()),
    }

    return rec


def run_objective_analysis() -> dict[str, Any]:
    """Run all analyses, write CSVs, return results bundle."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, missing, limitations = load_sweep_results()

    if missing:
        print(f"WARNING: missing sweep runs: {missing}")

    a1 = analysis1_criterion_comparison(df)
    a1["per_run"].to_csv(OUTPUT_DIR / "criterion_comparison_per_run.csv", index=False)
    a1["by_model"].to_csv(OUTPUT_DIR / "criterion_comparison_by_model.csv", index=False)

    a2 = analysis2_lr_comparison(df)
    a2.to_csv(OUTPUT_DIR / "lr_spread_by_criterion.csv", index=False)

    a3 = analysis3_gradient_decomposition(df)
    a3.to_csv(OUTPUT_DIR / "gradient_decomposition_by_lr.csv", index=False)

    # Per-architecture lr trend under each criterion
    trend_rows = []
    for criterion, col in [("val_loss", "benchmark_f1_val_loss_ckpt"), ("val_f1", "benchmark_f1_val_f1_ckpt")]:
        for model_id, sub in df.groupby("model_id"):
            ref_vals = sub[sub["lr"] == REFERENCE_LR].set_index("warmup_label")[col]
            for _, row in sub.iterrows():
                ref = ref_vals.get(row["warmup_label"], np.nan)
                trend_rows.append(
                    {
                        "criterion": criterion,
                        "model_id": model_id,
                        "short_name": row["short_name"],
                        "lr": row["lr"],
                        "warmup_label": row["warmup_label"],
                        "benchmark_f1": float(row[col]),
                        "benchmark_f1_delta_vs_5e6": float(row[col] - ref) if not np.isnan(ref) else np.nan,
                    }
                )
    trend = pd.DataFrame(trend_rows)
    trend.to_csv(OUTPUT_DIR / "architecture_lr_trend.csv", index=False)

    a4 = analysis4_recommendation(df, a1, a2, a3)

    print("\n=== Objective sweep analysis ===")
    print(f"Runs analysed: {a1['n_runs']}/24")
    print(
        f"Benchmark F1 mean: val_loss ckpt={a1['mean_benchmark_f1_val_loss']:.3f} "
        f"val_f1 ckpt={a1['mean_benchmark_f1_val_f1']:.3f} "
        f"(delta={a1['mean_delta_f1_minus_loss']:+.3f})"
    )
    print(
        f"val_f1 ckpt higher: {a1['val_f1_ckpt_higher_count']}/{a1['n_runs']} "
        f"Wilcoxon p={a1['wilcoxon_pvalue']:.4f}" if a1["wilcoxon_pvalue"] is not None else ""
    )
    print(
        f"Recommended: criterion={a4.get('recommended_criterion')} "
        f"lr={a4.get('recommended_lr')} warmup={a4.get('recommended_warmup')} "
        f"spread={a4.get('recommended_spread', 0):.3f}"
    )

    return {
        "df": df,
        "missing_runs": missing,
        "limitations": limitations,
        "artifact_note": _artifact_note(),
        "analysis1": a1,
        "analysis2": a2,
        "analysis3": a3,
        "analysis4": a4,
        "architecture_lr_trend": trend,
    }
