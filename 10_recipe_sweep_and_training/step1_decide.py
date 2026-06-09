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

PRIOR_DECISION_SNAPSHOT = SWEEP_OUTPUT_DIR / "recipe_decision_table_prior_string_match.csv"

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


def _deberta_collapsed(deberta_f1: float) -> bool:
    return float(deberta_f1) <= 0.0


def _exclusion_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if not bool(row["all_encoders_stable_seed42"]):
        reasons.append("not all encoders stable at seed 42 (collapse on at least one encoder)")
    if row["spread_quality"] == "INFLATED":
        reasons.append(f"benchmark spread is inflated: {row['spread_quality_note']}")
    if row["deberta_health"] == "SUPPRESSED" and not _deberta_collapsed(float(row["deberta_f1"])):
        reasons.append(row["deberta_health_note"])
    return reasons


def _guard_recovery_note(row: pd.Series) -> str | None:
    guard = str(row.get("guard_summary", ""))
    if guard == "no guard re-runs":
        return None
    return (
        f"Guard re-runs for {row['recipe']} recovered partially ({guard}), "
        "so seed-42 fairness is broken for this recipe."
    )


def _inflated_spread_note(row: pd.Series) -> str | None:
    if row["spread_quality"] != "INFLATED":
        return None
    return (
        f"{row['recipe']} posts spread {float(row['benchmark_f1_spread']):.3f} "
        f"with max driven by {row['spread_max_driver']} and min by {row['spread_min_driver']}; "
        f"the headline spread is not a genuine capability gradient ({row['spread_quality_note']})."
    )


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

    unstable = table[~table["all_encoders_stable_seed42"].astype(bool)]
    for _, row in unstable.iterrows():
        deb_f1 = float(row["deberta_f1"])
        if _deberta_collapsed(deb_f1):
            note = (
                f"{row['recipe']} is excluded on stability: DeBERTa collapsed to benchmark F1 "
                f"{deb_f1:.3f} at seed 42"
            )
            guard_note = _guard_recovery_note(row)
            if guard_note:
                note += f"; {guard_note}"
            else:
                note += "."
            contrasts.append(note)

    suppressed = table[
        (table["deberta_health"] == "SUPPRESSED")
        & table["all_encoders_stable_seed42"].astype(bool)
    ]
    for _, row in suppressed.iterrows():
        contrasts.append(
            f"{row['recipe']} is excluded on DeBERTa suppression: "
            f"DeBERTa F1 {float(row['deberta_f1']):.3f} is "
            f"{float(row['deberta_gap_from_best']):.3f} below sweep-best "
            f"{float(row['deberta_sweep_best']):.3f}"
            + (
                f", and spread is inflated ({row['spread_quality_note']})"
                if row["spread_quality"] == "INFLATED"
                else "."
            )
        )

    inflated_stable = table[
        (table["spread_quality"] == "INFLATED")
        & table["all_encoders_stable_seed42"].astype(bool)
        & (table["deberta_health"] != "SUPPRESSED")
    ]
    for _, row in inflated_stable.iterrows():
        note = _inflated_spread_note(row)
        if note:
            contrasts.append(note)

    wide_unstable = unstable.sort_values("benchmark_f1_spread", ascending=False)
    if not wide_unstable.empty:
        top = wide_unstable.iloc[0]
        contrasts.append(
            "The widest-spread recipes reflect DeBERTa collapse at seed 42, not a uniform "
            f"capability gain across encoders (for example {top['recipe']} spread "
            f"{float(top['benchmark_f1_spread']):.3f} with DeBERTa at "
            f"{float(top['deberta_f1']):.3f})."
        )

    narrow_alts = table[table["recipe"].isin(clean)].sort_values("benchmark_f1_spread")
    if len(narrow_alts) >= 2:
        names = ", ".join(narrow_alts["recipe"].head(3).tolist())
        contrasts.append(
            f"Among clean recipes, {names} offer narrower genuine spread "
            f"({float(narrow_alts.iloc[0]['benchmark_f1_spread']):.3f} to "
            f"{float(narrow_alts.iloc[min(2, len(narrow_alts) - 1)]['benchmark_f1_spread']):.3f}) "
            "with healthy DeBERTa scores, at the cost of a smaller encoder gradient."
        )

    if PRIOR_DECISION_SNAPSHOT.exists():
        prior = pd.read_csv(PRIOR_DECISION_SNAPSHOT)
        p15 = prior[prior["recipe"] == "1e-5/none"]
        n15 = table[table["recipe"] == "1e-5/none"]
        p35 = prior[prior["recipe"] == "3e-5/none"]
        n35 = table[table["recipe"] == "3e-5/none"]
        if not p15.empty and not n15.empty and not p35.empty and not n35.empty:
            contrasts.append(
                "Compared with the prior string-match sweep, the picture changes on clean data: "
                f"1e-5/none had DeBERTa {float(p15.iloc[0]['deberta_f1']):.3f} "
                f"({p15.iloc[0]['deberta_health']}) there but "
                f"{float(n15.iloc[0]['deberta_f1']):.3f} ({n15.iloc[0]['deberta_health']}) here; "
                f"3e-5/none had DeBERTa {float(p35.iloc[0]['deberta_f1']):.3f} "
                f"({p35.iloc[0]['deberta_health']}, stable={bool(p35.iloc[0]['all_encoders_stable_seed42'])}) "
                f"there but {float(n35.iloc[0]['deberta_f1']):.3f} "
                f"({n35.iloc[0]['deberta_health']}, stable={bool(n35.iloc[0]['all_encoders_stable_seed42'])}) "
                "here. The clean-data advisory supersedes the old one."
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
            f"({pick['spread_quality_note']}), holds DeBERTa at the sweep best "
            f"({float(pick['deberta_f1']):.3f} vs {float(pick['deberta_sweep_best']):.3f}), "
            f"and peaks encoders at a comparable depth (epoch std {float(pick['best_epoch_std']):.2f}). "
            f"Mean benchmark F1 is {float(pick['benchmark_f1_mean']):.3f} with spread "
            f"{float(pick['benchmark_f1_spread']):.3f}."
        )

    closing = (
        "This advisory does not set the training recipe. After reading the table, figure, "
        "and guard notes, assign the recipe yourself in the project configuration before step 2."
    )
    seed_caveat = (
        "This sweep is single-seed (seed 42). Stability at 3e-5/none is a seed-42 result and "
        "should be confirmed across the eight training seeds in step 2, watching DeBERTa "
        "specifically. If 3e-5 proves unstable across seeds, 5e-6/none or 1e-5/warmup are "
        "more conservative fallbacks with narrower genuine spread."
    )
    return excluded, clean, contrasts, rec, rec_reason + " " + seed_caveat + " " + closing


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
    paras: list[str] = [
        (
            "The step 1 sweep held seed 42 fixed and compared four encoders across learning rate "
            "and warmup. The enriched decision table names the encoder and best epoch behind each "
            "recipe's minimum and maximum benchmark F1, and flags spread that rests on an epoch-one "
            "checkpoint, a degenerate run, or a suppressed DeBERTa score."
        )
    ]
    if excluded:
        paras.append(
            "The following recipes are poor step 2 candidates on transparency grounds: "
            + "; ".join(excluded)
            + "."
        )

    stability = [c for c in contrasts if "excluded on stability" in c]
    suppression = [c for c in contrasts if "excluded on DeBERTa suppression" in c]
    widest = next((c for c in contrasts if c.startswith("The widest-spread")), None)
    narrow = next((c for c in contrasts if c.startswith("Among clean")), None)
    prior = next((c for c in contrasts if c.startswith("Compared with")), None)

    detail_parts = stability + suppression
    if detail_parts:
        paras.append(" ".join(detail_parts))
    if widest:
        paras.append(widest)
    if narrow:
        paras.append(narrow)
    if prior:
        paras.append(prior)

    paras.append(f"The advisory recommendation is {rec}. {rec_reason}")
    return paras


def write_sweep_report(table: pd.DataFrame, figure_path: Path, prose: list[str]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Step 1 recipe sweep report",
        "",
        "This report documents the recipe sweep on clean offset-marked training data "
        "(native entity offsets; marker_method=offset). "
        "It supersedes an earlier sweep run on string-match markers; conclusions below "
        "stand on the current pipeline only.",
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
    SWEEP_REPORT_PATH.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return SWEEP_REPORT_PATH


def print_prior_sweep_comparison(table: pd.DataFrame, rec: str) -> None:
    """High-level factual contrast with the pre-marker-fix sweep snapshot."""
    if not PRIOR_DECISION_SNAPSHOT.exists():
        _log("\n=== Comparison to prior string-match sweep ===")
        _log("  No prior snapshot (recipe_decision_table_prior_string_match.csv); skipped.")
        return

    prior = pd.read_csv(PRIOR_DECISION_SNAPSHOT)
    _, _, _, prior_rec, _ = _build_advisory_sections(prior)

    _log("\n=== Comparison to prior string-match sweep ===")
    _log(f"  Prior advisory (string-match data): {prior_rec}")
    _log(f"  Clean-data advisory: {rec}")
    if prior_rec != rec:
        _log("  Recipe recommendation changed on clean data.")
    else:
        _log("  Same recipe label recommended; compare metrics below (conclusion follows clean numbers).")

    prior_genuine = set(prior.loc[prior["spread_quality"] == "GENUINE", "recipe"])
    new_genuine = set(table.loc[table["spread_quality"] == "GENUINE", "recipe"])
    added = sorted(new_genuine - prior_genuine)
    removed = sorted(prior_genuine - new_genuine)
    _log(f"  GENUINE-spread recipes: prior {sorted(prior_genuine)} -> clean {sorted(new_genuine)}")
    if added:
        _log(f"    newly GENUINE on clean data: {added}")
    if removed:
        _log(f"    no longer GENUINE on clean data: {removed}")

    for recipe in ("1e-5/none", "2e-5/none", "3e-5/none"):
        p = prior[prior["recipe"] == recipe]
        n = table[table["recipe"] == recipe]
        if p.empty or n.empty:
            continue
        pr, nr = p.iloc[0], n.iloc[0]
        _log(
            f"  {recipe}: mean F1 {float(pr['benchmark_f1_mean']):.3f}->{float(nr['benchmark_f1_mean']):.3f}, "
            f"spread {float(pr['benchmark_f1_spread']):.3f}->{float(nr['benchmark_f1_spread']):.3f}, "
            f"DeBERTa {float(pr['deberta_f1']):.3f}->{float(nr['deberta_f1']):.3f} "
            f"({pr['deberta_health']}->{nr['deberta_health']}), "
            f"stable {bool(pr['all_encoders_stable_seed42'])}->{bool(nr['all_encoders_stable_seed42'])}"
        )

    deb_prior = prior.sort_values("lr").groupby("warmup_label")["deberta_f1"].apply(list)
    deb_new = table.sort_values("lr").groupby("warmup_label")["deberta_f1"].apply(list)
    _log("  DeBERTa F1 across lr (none warmup): prior vs clean")
    if "none" in deb_prior.index and "none" in deb_new.index:
        prior_vals = [f"{x:.3f}" for x in deb_prior["none"]]
        new_vals = [f"{x:.3f}" for x in deb_new["none"]]
        _log(f"    none: [{', '.join(prior_vals)}] -> [{', '.join(new_vals)}]")


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
    print_prior_sweep_comparison(table, rec)

    _log(f"\nWrote {out_csv}")
    _log(f"Wrote {figure_path}")
    _log(f"Updated {SWEEP_REPORT_PATH}")
    return table
