"""Regenerate budget-limited figures in shared style."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared.plot_style import COLORS, FIG_HEATMAP, FIG_SINGLE, FIG_WIDE, add_light_grid, apply_style, save_figure

from .paths import STEPS, step_paths


def _fig_dir(step_key: str) -> Path:
    return step_paths(STEPS[step_key])["figures"]


def _out_dir(step_key: str) -> Path:
    return step_paths(STEPS[step_key])["outputs"]


def _remove(path: Path) -> None:
    if path.exists():
        path.unlink()


def regenerate_step00() -> list[str]:
    apply_style()
    out = _out_dir("00")
    fig_dir = _fig_dir("00")
    pair = pd.read_csv(out / "entity_pair_breakdown.csv")
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    labels = pair["entity_pair_type"].astype(str).tolist()
    counts = pair["count"].tolist()
    colors = [COLORS["gene_drug"], COLORS["gene_disease"], COLORS["secondary"], COLORS["neutral_light"]][: len(labels)]
    ax.bar(labels, counts, color=colors, width=0.6)
    ax.set_ylabel("Evaluable targets")
    ax.set_title("CIViC evaluable targets by entity-pair type")
    add_light_grid(ax, "y")
    plt.xticks(rotation=25, ha="right")
    name = "entity_pair_distribution.png"
    save_figure(fig, fig_dir / name)
    for drop in ("direction_balance.png", "alignment_rates.png"):
        _remove(fig_dir / drop)
    return [name]


def regenerate_step01() -> list[str]:
    apply_style()
    out = _out_dir("01")
    fig_dir = _fig_dir("01")
    kept: list[str] = []

    ladder = pd.read_csv(out / "granularity_ladder.csv")
    if "level" in ladder.columns:
        y_col, x_col = "level", "n_relations"
    else:
        agg = ladder.groupby("granularity_level", as_index=False)["count"].sum()
        agg = agg.sort_values("granularity_level")
        y_col, x_col = "granularity_level", "count"
        ladder = agg
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.barh(ladder[y_col], ladder[x_col], color=COLORS["benchmark"], height=0.55)
    ax.set_xlabel("Training relations")
    ax.set_title("Corpus granularity ladder")
    add_light_grid(ax, "x")
    n1 = "01_corpus_granularity_ladder.png"
    save_figure(fig, fig_dir / n1)
    kept.append(n1)

    leak = pd.read_csv(out / "pmid_leakage.csv")
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    sub = leak[leak["corpus"] != "combined"]
    ax.bar(sub["corpus"], sub["overlap_count"], color=COLORS["highlight"], width=0.5)
    ax.set_ylabel("Leaked PMIDs into evaluation")
    ax.set_title("Training-evaluation PMID overlap")
    add_light_grid(ax, "y")
    n2 = "01_corpus_pmid_leakage.png"
    save_figure(fig, fig_dir / n2)
    kept.append(n2)

    for old in fig_dir.glob("01_*.png"):
        if old.name not in kept:
            _remove(old)
    return kept


def regenerate_step02() -> list[str]:
    apply_style()
    out = _out_dir("02")
    fig_dir = _fig_dir("02")
    import json

    proto = json.loads((out / "frozen_protocol.json").read_text(encoding="utf-8"))
    stats = proto["statistics"]
    labels = list(stats["targets_by_pair_type"].keys())
    vals = [stats["targets_by_pair_type"][k] for k in labels]
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    colors = [COLORS["gene_drug"], COLORS["gene_disease"]][: len(labels)]
    ax.bar(labels, vals, color=colors, width=0.5)
    ax.set_ylabel("Frozen ranking targets")
    ax.set_title("Frozen evaluation target composition")
    add_light_grid(ax, "y")
    name = "02_evaluation_protocol_composition.png"
    save_figure(fig, fig_dir / name)
    return [name]


def regenerate_step03() -> list[str]:
    apply_style()
    out = _out_dir("03")
    fig_dir = _fig_dir("03")
    kept: list[str] = []

    cov = pd.read_csv(out / "03_candidate_pool_positive_coverage.csv")
    n_eval = int(cov["evaluable"].sum())
    n_total = len(cov)
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.bar(["Evaluable", "Not evaluable"], [n_eval, n_total - n_eval], color=[COLORS["positive"], COLORS["neutral_light"]], width=0.5)
    ax.set_ylabel("CIViC primary targets")
    ax.set_title("Abstract coverage of frozen targets")
    add_light_grid(ax, "y")
    n1 = "03_candidate_pool_coverage.png"
    save_figure(fig, fig_dir / n1)
    kept.append(n1)

    import json

    summary = json.loads((out / "03_candidate_pool_entity_type_alignment_summary.json").read_text(encoding="utf-8"))
    n_matched = summary["n_matched_relations"]
    n_total_rel = 1812
    n_miss = n_total_rel - n_matched
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.barh(["Matched in pool", "No pool positive"], [n_matched, n_miss], color=[COLORS["benchmark"], COLORS["highlight"]], height=0.5)
    ax.set_xlabel("Primary CIViC relations")
    ax.set_title("PubTator recall on primary targets")
    add_light_grid(ax, "x")
    n2 = "03_candidate_pool_pubtator_recall_gap.png"
    save_figure(fig, fig_dir / n2)
    kept.append(n2)

    for old in fig_dir.glob("03_*.png"):
        if old.name not in kept:
            _remove(old)
    return kept


def regenerate_step04() -> list[str]:
    apply_style()
    out = _out_dir("04")
    fig_dir = _fig_dir("04")
    df = pd.read_csv(out / "04_pilot_study_benchmark_vs_kb.csv")
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.scatter(df["benchmark_f1"], df["mrr"], s=80, color=COLORS["benchmark"], edgecolors=COLORS["neutral"], linewidths=0.8)
    for _, row in df.iterrows():
        ax.annotate(
            row["short_name"].replace("-base", ""),
            (row["benchmark_f1"], row["mrr"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("In-distribution benchmark F1 (pre-fix pipeline)")
    ax.set_ylabel("Out-of-distribution KB MRR")
    ax.set_title("Pilot: benchmark vs knowledge-base ranking")
    add_light_grid(ax, "y")
    name = "04_pilot_study_benchmark_vs_kb.png"
    save_figure(fig, fig_dir / name)
    for old in fig_dir.glob("04_*.png"):
        if old.name != name:
            _remove(old)
    return [name]


def regenerate_step05() -> list[str]:
    apply_style()
    out = _out_dir("05")
    fig_dir = _fig_dir("05")
    checks = pd.read_csv(out / "quality_gate_checks.csv")
    before_vals: list[float] = []
    after_vals: list[float] = []
    labels: list[str] = []
    for _, row in checks.iterrows():
        if pd.isna(row.get("before")) or str(row["before"]).strip() in ("", "NaN", "nan"):
            continue
        v = row["value"]
        try:
            after_val = float(v)
        except (TypeError, ValueError):
            continue
        labels.append(str(row["name"]).replace("_", " ")[:28])
        b = str(row["before"]).replace("%", "")
        try:
            before_vals.append(float(b) if b.replace(".", "").replace("-", "").isdigit() else 0.0)
        except ValueError:
            before_vals.append(0.0)
        after_vals.append(after_val)
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    y = np.arange(len(labels))
    ax.barh(y - 0.15, before_vals[: len(y)], height=0.28, label="Before offset gate", color=COLORS["highlight"])
    ax.barh(y + 0.15, after_vals[: len(y)], height=0.28, label="After offset gate", color=COLORS["benchmark"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Rate (proportion)")
    ax.set_title("Marker quality gate: before and after")
    ax.legend(frameon=False, loc="lower right")
    add_light_grid(ax, "x")
    name = "05_marker_quality_gate_before_after.png"
    save_figure(fig, fig_dir / name)
    return [name]


def regenerate_step10() -> list[str]:
    apply_style()
    from shared.models import MODELS

    from .paths import OUTPUT_ROOT

    base = OUTPUT_ROOT / "outputs" / STEPS["10"]
    fig_base = OUTPUT_ROOT / "figures" / STEPS["10"]
    sweep_dir = fig_base / "sweep"
    matrix_dir = fig_base / "matrix"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []

    decision = pd.read_csv(base / "sweep" / "recipe_decision_table.csv")
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.scatter(
        decision["benchmark_f1_spread"],
        decision["deberta_f1"],
        c=decision["lr"],
        cmap="viridis",
        s=60,
        edgecolors=COLORS["neutral"],
        linewidths=0.5,
    )
    ax.set_xlabel("Benchmark F1 spread across encoders")
    ax.set_ylabel("DeBERTa benchmark F1")
    ax.set_title("Recipe sweep: spread vs DeBERTa health")
    add_light_grid(ax, "y")
    n1 = "sweep/recipe_spread_vs_deberta_health.png"
    save_figure(fig, sweep_dir / "recipe_spread_vs_deberta_health.png")
    kept.append(n1)

    mat = pd.read_csv(base / "matrix" / "matrix_per_run.csv")
    ok = mat[~mat["missing"].astype(bool)].copy()
    pivot = ok.pivot(index="short_name", columns="seed", values="benchmark_f1")
    order = [m.short_name for m in MODELS if m.short_name in pivot.index]
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=FIG_HEATMAP)
    data = pivot.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0.65, vmax=max(0.85, float(np.nanmax(data))))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Encoder")
    ax.set_title("Confirmed matrix: BioRED presence F1")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    n2 = "matrix/matrix_benchmark_f1_heatmap.png"
    save_figure(fig, matrix_dir / "matrix_benchmark_f1_heatmap.png")
    kept.append(n2)
    return kept


def regenerate_step20() -> list[str]:
    apply_style()
    out = _out_dir("20")
    fig_dir = _fig_dir("20")
    kept: list[str] = []

    paired = pd.read_csv(out / "20_within_seed_paired_changes.csv")
    agg = paired.groupby("short_name").agg(
        d_bench=("delta_benchmark_f1_val_f1_best", "mean"),
        d_kb_hard=("delta_kb_mrr_hard_val_f1_best", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    x = np.arange(len(agg))
    w = 0.35
    ax.bar(x - w / 2, agg["d_bench"], w, label="Benchmark F1 change", color=COLORS["benchmark"])
    ax.bar(x + w / 2, agg["d_kb_hard"], w, label="KB hard MRR change", color=COLORS["kb"])
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["short_name"].str.replace("-base", ""), rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean paired change (epoch 1 to best val F1)")
    ax.set_title("Within-seed paired change by encoder")
    ax.legend(frameon=False)
    add_light_grid(ax, "y")
    n1 = "fig2_within_seed_paired_change.png"
    save_figure(fig, fig_dir / n1)
    kept.append(n1)

    pt = pd.read_csv(out / "20_pair_type_breakdown.csv")
    sub = pt[pt["well_trained_definition"] == "val_f1_best"]
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    colors = [COLORS["gene_drug"], COLORS["gene_disease"]]
    ax.bar(sub["pair_type"], sub["mean_delta_kb_mrr"], color=colors, width=0.5)
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.8)
    ax.set_ylabel("Mean KB MRR change")
    ax.set_title("Pair-type asymmetry (best val F1 checkpoint)")
    add_light_grid(ax, "y")
    n2 = "fig3_hard_easy_pair_type.png"
    save_figure(fig, fig_dir / n2)
    kept.append(n2)

    gd = pd.read_csv(out / "20_gene_disease_subset_breakdown.csv")
    sub2 = gd[gd["well_trained_definition"] == "val_f1_best"]
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    labels = sub2["label"].str.replace("gene-disease ", "").tolist()
    ax.bar(labels, sub2["mean_delta_kb_mrr"], color=COLORS["gene_disease"], width=0.55)
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.8)
    ax.set_ylabel("Mean KB MRR change")
    ax.set_title("Gene-disease decline by subset")
    add_light_grid(ax, "y")
    plt.xticks(rotation=15, ha="right")
    n3 = "fig6_pair_type_subset_contrast.png"
    save_figure(fig, fig_dir / n3)
    kept.append(n3)

    for old in fig_dir.glob("fig*.png"):
        if old.name not in kept:
            _remove(old)
    return kept


REGENERATORS = {
    "00": regenerate_step00,
    "01": regenerate_step01,
    "02": regenerate_step02,
    "03": regenerate_step03,
    "04": regenerate_step04,
    "05": regenerate_step05,
    "10": regenerate_step10,
    "20": regenerate_step20,
}
