"""Step 1 advisory table: five criteria for recipe comparison (no lock-in)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from shared.constants import GUARD_SEEDS

from .config import OUTPUT_DIR, REPORT_DIR, SWEEP_COMPLETE, SWEEP_RESULTS_DIR, all_sweep_points


def _load_all_sweep_markers() -> list[dict]:
    rows: list[dict] = []
    if not SWEEP_RESULTS_DIR.exists():
        return rows
    for path in sorted(SWEEP_RESULTS_DIR.rglob(SWEEP_COMPLETE)):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def load_sweep_results_seed42() -> pd.DataFrame:
    """Primary fair-comparison table: seed 42 only, excluding guard re-runs."""
    rows: list[dict] = []
    for point in all_sweep_points():
        path = point.result_path()
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("bad_seed_guard"):
            continue
        rows.append(data)
    if not rows:
        raise SystemExit("No sweep results found. Run step-1 sweep first.")
    return pd.DataFrame(rows)


def build_guard_outcomes_table(all_results: pd.DataFrame) -> pd.DataFrame:
    """
    For each seed-42 run that triggered the bad-seed guard, report seeds 43 and 44 outcomes.
    Separate from the seed-42 aggregation; informs all-encoder-stability judgement.
    """
    if all_results.empty:
        return pd.DataFrame()

    primary = all_results[(all_results["seed"] == 42) & (~all_results["bad_seed_guard"].fillna(False))]
    guard = all_results[
        all_results["bad_seed_guard"].fillna(False) | all_results["seed"].isin(GUARD_SEEDS)
    ]

    rows: list[dict] = []
    for _, p in primary.iterrows():
        if not p.get("degenerate"):
            continue
        for gseed in GUARD_SEEDS:
            sub = guard[
                (guard["model_id"] == p["model_id"])
                & (guard["lr"] == p["lr"])
                & (guard["warmup_label"] == p["warmup_label"])
                & (guard["seed"] == gseed)
            ]
            if sub.empty:
                rows.append(
                    {
                        "model_id": p["model_id"],
                        "lr": p["lr"],
                        "warmup_label": p["warmup_label"],
                        "primary_seed": 42,
                        "primary_degenerate": True,
                        "guard_seed": gseed,
                        "guard_benchmark_f1": np.nan,
                        "guard_degenerate": np.nan,
                        "guard_outcome": "not_run",
                    }
                )
                continue
            g = sub.iloc[0]
            recovered = not bool(g.get("degenerate"))
            rows.append(
                {
                    "model_id": p["model_id"],
                    "lr": p["lr"],
                    "warmup_label": p["warmup_label"],
                    "primary_seed": 42,
                    "primary_degenerate": True,
                    "guard_seed": gseed,
                    "guard_benchmark_f1": float(g["benchmark_f1"]),
                    "guard_degenerate": bool(g.get("degenerate")),
                    "guard_outcome": "recovered" if recovered else "also_collapsed",
                }
            )
    return pd.DataFrame(rows)


def build_advisory_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by recipe (lr, warmup) across four sweep encoders at seed 42."""
    rows: list[dict] = []
    weak_id = "distilbert_base"

    for (lr, warmup_label), sub in df.groupby(["lr", "warmup_label"]):
        bf1 = sub["benchmark_f1"].astype(float)
        epochs = sub["best_epoch_val_f1"].astype(float)
        n_degen = int(sub["degenerate"].sum()) if "degenerate" in sub.columns else 0
        weak = sub[sub["model_id"] == weak_id]["benchmark_f1"].astype(float)
        strong = sub[sub["model_id"] != weak_id]["benchmark_f1"].astype(float)
        cap_gap = float(strong.mean() - weak.mean()) if not weak.empty and not strong.empty else np.nan

        rows.append(
            {
                "lr": lr,
                "warmup_label": warmup_label,
                "n_encoders": len(sub),
                "benchmark_f1_mean": float(bf1.mean()),
                "benchmark_f1_min": float(bf1.min()),
                "benchmark_f1_max": float(bf1.max()),
                "benchmark_f1_spread": float(bf1.max() - bf1.min()),
                "best_epoch_mean": float(epochs.mean()),
                "best_epoch_std": float(epochs.std(ddof=0)),
                "capability_minus_weak_mean": cap_gap,
                "n_degenerate_seed42": n_degen,
                "all_encoders_stable_seed42": n_degen == 0,
                "deberta_f1": float(
                    sub.loc[sub["model_id"] == "deberta_base", "benchmark_f1"].iloc[0]
                )
                if (sub["model_id"] == "deberta_base").any()
                else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["benchmark_f1_spread", "benchmark_f1_mean"], ascending=[True, False]
    )


def print_advisory(table: pd.DataFrame, guard_table: pd.DataFrame) -> None:
    print("\n=== Step-1 recipe advisory (seed 42; you choose the recipe) ===")
    print(
        "Criteria: (1) low encoder spread, (2) high absolute F1, "
        "(3) capability vs weak-model gap, (4) all-encoder stability incl. DeBERTa, "
        "(5) stable best epoch."
    )
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Bad-seed guard outcomes (seeds 43 and 44; NOT in table above) ===")
    if guard_table.empty:
        print("  No guard re-runs were triggered (no seed-42 collapses).")
    else:
        print(guard_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(
            "\nGuard rows show whether seeds 43 and 44 recovered or also collapsed when "
            "seed 42 failed. Use them alongside the seed-42 table for stability judgement."
        )

    print(
        "\nThis table is advisory only. Set CHOSEN_RECIPE in config.py before step-2 training."
    )


def _write_guard_report(guard_table: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "sweep_guard_outcomes.md"
    lines = [
        "# Step 1 bad-seed guard outcomes",
        "",
        "The seed-42 advisory table compares recipes fairly at one seed. When seed 42 "
        "collapsed for an encoder and recipe, the guard re-ran that combination with "
        "seeds 43 and 44. Those outcomes are recorded here separately.",
        "",
    ]
    if guard_table.empty:
        lines.append("No guard re-runs were triggered.")
    else:
        for _, r in guard_table.iterrows():
            lines.append(
                f"{r['model_id']} at lr {r['lr']} warmup {r['warmup_label']}: "
                f"seed 42 degenerate; guard seed {int(r['guard_seed'])} "
                f"benchmark F1 {r.get('guard_benchmark_f1', 'n/a')} "
                f"({r['guard_outcome']})."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_advisory() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_markers = _load_all_sweep_markers()
    all_df = pd.DataFrame(all_markers) if all_markers else pd.DataFrame()
    if not all_df.empty:
        all_df.to_csv(OUTPUT_DIR / "sweep_all_runs.csv", index=False)

    df = load_sweep_results_seed42()
    df.to_csv(OUTPUT_DIR / "sweep_per_run_seed42.csv", index=False)

    guard_table = build_guard_outcomes_table(all_df)
    guard_table.to_csv(OUTPUT_DIR / "sweep_guard_outcomes.csv", index=False)

    table = build_advisory_table(df)
    table.to_csv(OUTPUT_DIR / "sweep_advisory_table.csv", index=False)

    print_advisory(table, guard_table)
    _write_guard_report(guard_table)
    return table
