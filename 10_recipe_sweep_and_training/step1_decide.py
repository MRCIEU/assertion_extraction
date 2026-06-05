"""Step 1 recipe decision aid: transparent flags and non-binding advisory prose."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    OUTPUT_DIR,
    REPORT_DIR,
    SWEEP_FIGURE_DIR,
    SWEEP_OUTPUT_DIR,
    SWEEP_REPORT_PATH,
)

# Explainable thresholds (stated in report prose, not used as hidden ranking weights).
CONVERGENCE_STD_FLAG = 2.5
DEBERTA_SUPPRESSION_GAP = 0.03
EPOCH1_UNDERTRAINED = 1

PALETTE = {
    "neutral": "#4477AA",
    "accent": "#CC6677",
    "warn": "#DDAA33",
    "grid": "#DDDDDD",
    "text": "#222222",
}
DPI = 300

# Manual offsets (points) for eight recipe labels on the scatter plot.
_RECIPE_LABEL_OFFSETS: dict[str, tuple[float, float]] = {
    "5e-06|none": (8, 10),
    "5e-06|warmup_10pct": (-48, 8),
    "1e-05|none": (8, -12),
    "1e-05|warmup_10pct": (-52, -10),
    "2e-05|none": (10, 8),
    "2e-05|warmup_10pct": (-46, 12),
    "3e-05|none": (8, 14),
    "3e-05|warmup_10pct": (-50, -14),
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _recipe_key(lr: float, warmup_label: str) -> str:
    return f"{lr}|{warmup_label}"


def _recipe_label(lr: float, warmup_label: str) -> str:
    lr_s = f"{lr:.0e}".replace("+", "").replace("e-0", "e-")
    warm = "warmup" if warmup_label != "none" else "none"
    return f"{lr_s}/{warm}"


def _resolve_csv(name: str) -> Path:
    for base in (SWEEP_OUTPUT_DIR, OUTPUT_DIR):
        path = base / name
        if path.exists():
            return path
    raise SystemExit(
        f"Missing {name}. Run step-1 sweep and --sweep-advisory-only first "
        f"(expected under {SWEEP_OUTPUT_DIR} or {OUTPUT_DIR})."
    )


def load_sweep_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    advisory = pd.read_csv(_resolve_csv("sweep_advisory_table.csv"))
    per_run = pd.read_csv(_resolve_csv("sweep_per_run_seed42.csv"))
    guard_path = SWEEP_OUTPUT_DIR / "sweep_guard_outcomes.csv"
    if not guard_path.exists():
        guard_path = OUTPUT_DIR / "sweep_guard_outcomes.csv"
    guard = pd.read_csv(guard_path) if guard_path.exists() else pd.DataFrame()
    return advisory, per_run, guard


def _run_at_extreme(row: pd.Series) -> bool:
    return bool(row.get("degenerate")) or int(row["best_epoch_val_f1"]) == EPOCH1_UNDERTRAINED


def _spread_drivers(sub: pd.DataFrame) -> dict[str, str | float | bool]:
    max_row = sub.loc[sub["benchmark_f1"].astype(float).idxmax()]
    min_row = sub.loc[sub["benchmark_f1"].astype(float).idxmin()]
    max_bad = _run_at_extreme(max_row)
    min_bad = _run_at_extreme(min_row)
    inflated = max_bad or min_bad
    notes: list[str] = []
    if max_bad:
        if bool(max_row.get("degenerate")):
            notes.append(f"max from {max_row['short_name']} (degenerate, epoch {int(max_row['best_epoch_val_f1'])})")
        elif int(max_row["best_epoch_val_f1"]) == EPOCH1_UNDERTRAINED:
            notes.append(f"max from {max_row['short_name']} (epoch-1 checkpoint, F1={float(max_row['benchmark_f1']):.3f})")
    if min_bad:
        if bool(min_row.get("degenerate")):
            notes.append(f"min from {min_row['short_name']} (degenerate, epoch {int(min_row['best_epoch_val_f1'])})")
        elif int(min_row["best_epoch_val_f1"]) == EPOCH1_UNDERTRAINED:
            notes.append(f"min from {min_row['short_name']} (epoch-1 checkpoint, F1={float(min_row['benchmark_f1']):.3f})")
    return {
        "spread_max_driver": (
            f"{max_row['short_name']} F1={float(max_row['benchmark_f1']):.3f} "
            f"best_epoch={int(max_row['best_epoch_val_f1'])}"
        ),
        "spread_min_driver": (
            f"{min_row['short_name']} F1={float(min_row['benchmark_f1']):.3f} "
            f"best_epoch={int(min_row['best_epoch_val_f1'])}"
        ),
        "spread_quality": "INFLATED" if inflated else "GENUINE",
        "spread_quality_note": "; ".join(notes) if notes else "extremes from well-trained non-degenerate runs",
    }


def _deberta_health(deberta_f1: float, deberta_sweep_best: float) -> dict[str, str | float]:
    gap = deberta_sweep_best - deberta_f1
    if gap <= DEBERTA_SUPPRESSION_GAP:
        status = "HEALTHY"
        note = f"within {DEBERTA_SUPPRESSION_GAP:.2f} of sweep-best DeBERTa ({deberta_sweep_best:.3f})"
    else:
        status = "SUPPRESSED"
        note = (
            f"DeBERTa F1 {deberta_f1:.3f} is {gap:.3f} below sweep-best {deberta_sweep_best:.3f}; "
            "spread may partly reflect suppressing this encoder"
        )
    return {"deberta_health": status, "deberta_health_note": note, "deberta_gap_from_best": gap}


def _convergence_flag(sub: pd.DataFrame, epoch_std: float) -> dict[str, str | bool]:
    epochs = sub["best_epoch_val_f1"].astype(int)
    span = int(epochs.max() - epochs.min())
    flagged = epoch_std >= CONVERGENCE_STD_FLAG
    if flagged:
        detail = (
            f"best epochs span {int(epochs.min())}-{int(epochs.max())} "
            f"(std={epoch_std:.2f}); encoders trained to different depths"
        )
    else:
        detail = f"epochs span {int(epochs.min())}-{int(epochs.max())} (std={epoch_std:.2f})"
    return {
        "convergence_consistency_flag": flagged,
        "convergence_consistency_note": detail,
    }


def _guard_summary(lr: float, warmup_label: str, guard: pd.DataFrame) -> str:
    if guard.empty:
        return "no guard re-runs"
    sub = guard[(guard["lr"] == lr) & (guard["warmup_label"] == warmup_label)]
    if sub.empty:
        return "no guard re-runs"
    parts: list[str] = []
    for _, g in sub.iterrows():
        f1 = g.get("guard_benchmark_f1", np.nan)
        f1_s = f"{float(f1):.3f}" if pd.notna(f1) else "n/a"
        parts.append(
            f"seed {int(g['guard_seed'])} {g['guard_outcome']} "
            f"(benchmark F1 {f1_s}, degenerate={g.get('guard_degenerate', 'n/a')})"
        )
    return "; ".join(parts)


def build_decision_table(
    advisory: pd.DataFrame, per_run: pd.DataFrame, guard: pd.DataFrame
) -> pd.DataFrame:
    deberta_best = float(advisory["deberta_f1"].max())
    rows: list[dict] = []
    for _, adv in advisory.iterrows():
        lr, warm = adv["lr"], adv["warmup_label"]
        sub = per_run[(per_run["lr"] == lr) & (per_run["warmup_label"] == warm)]
        spread = _spread_drivers(sub)
        deb = _deberta_health(float(adv["deberta_f1"]), deberta_best)
        conv = _convergence_flag(sub, float(adv["best_epoch_std"]))
        rows.append(
            {
                "recipe": _recipe_label(lr, warm),
                "lr": lr,
                "warmup_label": warm,
                "benchmark_f1_spread": adv["benchmark_f1_spread"],
                "benchmark_f1_mean": adv["benchmark_f1_mean"],
                "benchmark_f1_min": adv["benchmark_f1_min"],
                "benchmark_f1_max": adv["benchmark_f1_max"],
                "best_epoch_mean": adv["best_epoch_mean"],
                "best_epoch_std": adv["best_epoch_std"],
                "capability_minus_weak_mean": adv["capability_minus_weak_mean"],
                "all_encoders_stable_seed42": adv["all_encoders_stable_seed42"],
                "deberta_f1": adv["deberta_f1"],
                "deberta_sweep_best": deberta_best,
                **spread,
                **deb,
                **conv,
                "guard_summary": _guard_summary(lr, warm, guard),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["benchmark_f1_spread", "benchmark_f1_mean"], ascending=[True, False])


def _exclusion_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if not bool(row["all_encoders_stable_seed42"]):
        reasons.append("not all encoders stable at seed 42 (collapse on at least one encoder)")
    if row["spread_quality"] == "INFLATED":
        reasons.append(f"benchmark spread is inflated: {row['spread_quality_note']}")
    if row["deberta_health"] == "SUPPRESSED":
        reasons.append(row["deberta_health_note"])
    return reasons


def _build_advisory_sections(table: pd.DataFrame) -> tuple[list[str], list[str], list[str], str, str]:
    excluded: list[str] = []
    clean: list[str] = []
    for _, r in table.iterrows():
        reasons = _exclusion_reasons(r)
        label = r["recipe"]
        if reasons:
            excluded.append(f"{label}: " + "; ".join(reasons))
        else:
            clean.append(label)

    contrasts: list[str] = []
    r15 = table[table["recipe"] == "1e-5/none"]
    r25 = table[table["recipe"] == "2e-5/none"]
    if not r15.empty and not r25.empty:
        a, b = r15.iloc[0], r25.iloc[0]
        contrasts.append(
            f"{a['recipe']} and {b['recipe']} show similar spread "
            f"({float(a['benchmark_f1_spread']):.3f} vs {float(b['benchmark_f1_spread']):.3f}) "
            f"and mean benchmark F1 ({float(a['benchmark_f1_mean']):.3f} vs {float(b['benchmark_f1_mean']):.3f}), "
            f"but {b['recipe']} leaves DeBERTa at {float(b['deberta_f1']):.3f} while "
            f"{a['recipe']} keeps DeBERTa at {float(a['deberta_f1']):.3f} near the sweep-best "
            f"{float(a['deberta_sweep_best']):.3f}. The extra spread at {b['recipe']} aligns with "
            f"an epoch-1 PubMedBERT maximum ({b['spread_max_driver']}), not a uniform capability gain."
        )

    r35w = table[table["recipe"] == "3e-5/warmup"]
    if not r35w.empty:
        rw = r35w.iloc[0]
        contrasts.append(
            f"{rw['recipe']} posts the widest spread ({float(rw['benchmark_f1_spread']):.3f}) "
            f"with max driven by {rw['spread_max_driver']} and min by {rw['spread_min_driver']}; "
            "the high spread rests on an early PubMedBERT checkpoint."
        )

    r35n = table[table["recipe"] == "3e-5/none"]
    if not r35n.empty:
        rn = r35n.iloc[0]
        contrasts.append(
            f"{rn['recipe']} is excluded on stability: DeBERTa collapsed to benchmark F1 "
            f"{float(rn['deberta_f1']):.3f} at seed 42; guard re-runs recovered partially "
            f"({rn['guard_summary']}), so seed-42 fairness is broken for this recipe."
        )

    # Pick among clean recipes using stated priorities (not a weighted score).
    candidates = table[table["recipe"].isin(clean)].copy()
    if candidates.empty:
        rec = "none (all recipes carry at least one exclusion flag; review the table manually)"
        rec_reason = "Every recipe failed at least one stability or spread-trust check."
    else:
        candidates = candidates.sort_values(
            ["deberta_gap_from_best", "benchmark_f1_spread", "best_epoch_std"],
            ascending=[True, True, True],
        )
        pick = candidates.iloc[0]
        rec = pick["recipe"]
        rec_reason = (
            f"{rec} keeps all four encoders stable at seed 42, shows genuine spread "
            f"({pick['spread_quality_note']}), holds DeBERTa near the sweep best "
            f"({float(pick['deberta_f1']):.3f} vs {float(pick['deberta_sweep_best']):.3f}), "
            f"and peaks encoders at a comparable depth (epoch std {float(pick['best_epoch_std']):.2f}). "
            f"Mean benchmark F1 is {float(pick['benchmark_f1_mean']):.3f} with spread "
            f"{float(pick['benchmark_f1_spread']):.3f}."
        )

    closing = (
        "This advisory does not set the training recipe. After reading the table, figure, "
        "and guard notes, assign the recipe yourself in the project configuration before step 2."
    )
    return excluded, clean, contrasts, rec, rec_reason + " " + closing


def _apply_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.edgecolor": PALETTE["text"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def plot_stability_vs_spread(table: pd.DataFrame) -> Path:
    _apply_figure_style()
    SWEEP_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    x = table["benchmark_f1_spread"].astype(float)
    y = table["deberta_f1"].astype(float) - table["deberta_sweep_best"].astype(float)

    colors = []
    for _, r in table.iterrows():
        if not bool(r["all_encoders_stable_seed42"]) or r["spread_quality"] == "INFLATED":
            colors.append(PALETTE["warn"])
        elif r["deberta_health"] == "SUPPRESSED":
            colors.append(PALETTE["accent"])
        else:
            colors.append(PALETTE["neutral"])

    ax.scatter(x, y, c=colors, s=90, edgecolors=PALETTE["text"], linewidths=0.6, zorder=3)
    for _, r in table.iterrows():
        key = _recipe_key(r["lr"], r["warmup_label"])
        ox, oy = _RECIPE_LABEL_OFFSETS.get(key, (8, 8))
        ax.annotate(
            r["recipe"],
            (float(r["benchmark_f1_spread"]), float(r["deberta_f1"]) - float(r["deberta_sweep_best"])),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=9,
            color=PALETTE["text"],
        )

    ax.axhline(0.0, color=PALETTE["text"], linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)
    ax.set_xlabel("Benchmark F1 spread across four encoders (seed 42)")
    ax.set_ylabel("DeBERTa F1 minus sweep-best DeBERTa F1")
    ax.set_title("Recipe tradeoff: spread versus DeBERTa health")
    ax.margins(x=0.08, y=0.12)

    out = SWEEP_FIGURE_DIR / "recipe_spread_vs_deberta_health.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.12, facecolor="white", edgecolor="none")
    plt.close(fig)
    return out


def _prose_paragraphs(excluded: list[str], contrasts: list[str], rec: str, rec_reason: str) -> list[str]:
    paras: list[str] = []
    paras.append(
        "The step 1 sweep held seed 42 fixed and compared four encoders across learning rate "
        "and warmup. The enriched decision table names the encoder and best epoch behind each "
        "recipe's minimum and maximum benchmark F1, and flags spread that rests on an epoch-one "
        "checkpoint, a degenerate run, or a suppressed DeBERTa score."
    )
    if excluded:
        paras.append(
            "The following recipes are poor step 2 candidates on transparency grounds: "
            + " ".join(excluded)
        )
    if contrasts:
        paras.append(" ".join(contrasts))
    paras.append(f"The advisory recommendation is {rec}. {rec_reason}")
    return paras


def write_sweep_report(table: pd.DataFrame, figure_path: Path, prose: list[str]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    existing = SWEEP_REPORT_PATH.read_text(encoding="utf-8") if SWEEP_REPORT_PATH.exists() else ""
    if "## Recipe decision" in existing:
        existing = existing.split("## Recipe decision")[0].rstrip()

    lines = [
        existing,
        "",
        "## Recipe decision",
        "",
    ]
    lines.extend(prose)
    lines.extend(
        [
            "",
            f"See also the stability-versus-spread figure ({figure_path.name}) and the enriched "
            "decision table written alongside the sweep CSVs.",
            "",
        ]
    )
    SWEEP_REPORT_PATH.write_text("\n".join(l for l in lines if l is not None).strip() + "\n", encoding="utf-8")
    return SWEEP_REPORT_PATH


def print_decision_output(
    table: pd.DataFrame,
    excluded: list[str],
    clean: list[str],
    contrasts: list[str],
    rec: str,
    rec_reason: str,
) -> None:
    _log("\n=== Recipe decision table (diagnostic flags; you choose the recipe) ===")
    show_cols = [
        "recipe",
        "benchmark_f1_spread",
        "benchmark_f1_mean",
        "deberta_f1",
        "spread_quality",
        "spread_max_driver",
        "spread_min_driver",
        "deberta_health",
        "convergence_consistency_flag",
        "all_encoders_stable_seed42",
        "guard_summary",
    ]
    _log(table[show_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    _log("\n=== Excluded recipes (with reasons) ===")
    if excluded:
        for line in excluded:
            _log(f"  - {line}")
    else:
        _log("  (none flagged for hard exclusion)")

    _log("\n=== Cleanest candidates ===")
    if clean:
        _log("  " + ", ".join(clean))
    else:
        _log("  (none passed all exclusion checks)")
    for c in contrasts:
        _log(f"  {c}")

    _log("\n=== Advisory recommendation (non-binding) ===")
    _log(f"  Recommended recipe: {rec}")
    _log(f"  Reason: {rec_reason}")
    _log("\nThe final recipe choice is yours. Set CHOSEN_RECIPE in config.py before step 2.")


def run_decide_recipe() -> pd.DataFrame:
    SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    advisory, per_run, guard = load_sweep_tables()
    table = build_decision_table(advisory, per_run, guard)
    out_csv = SWEEP_OUTPUT_DIR / "recipe_decision_table.csv"
    table.to_csv(out_csv, index=False)

    figure_path = plot_stability_vs_spread(table)
    excluded, clean, contrasts, rec, rec_reason = _build_advisory_sections(table)
    prose = _prose_paragraphs(excluded, contrasts, rec, rec_reason)
    write_sweep_report(table, figure_path, prose)
    print_decision_output(table, excluded, clean, contrasts, rec, rec_reason)

    _log(f"\nWrote {out_csv}")
    _log(f"Wrote {figure_path}")
    _log(f"Updated {SWEEP_REPORT_PATH}")
    return table
