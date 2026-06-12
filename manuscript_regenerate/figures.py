"""Regenerate budget-limited figures in shared style."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared.models import MODELS, MODEL_BY_ID
from shared.plot_style import (
    COLORS,
    FIG_HEATMAP,
    FIG_PANEL,
    FIG_SINGLE,
    FIG_WIDE,
    add_light_grid,
    apply_style,
    encoder_point_color,
    save_figure,
)

from .paths import OUTPUT_ROOT, STEPS, step_paths


def _fig_dir(step_key: str) -> Path:
    return step_paths(STEPS[step_key])["figures"]


def _out_dir(step_key: str) -> Path:
    return step_paths(STEPS[step_key])["outputs"]


def _remove(path: Path) -> None:
    if path.exists():
        path.unlink()


def _short_name(name: str) -> str:
    return name.replace("-base", "")


def regenerate_step00() -> list[str]:
    apply_style()
    out = _out_dir("00")
    fig_dir = _fig_dir("00")
    pair = pd.read_csv(out / "entity_pair_breakdown.csv")
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    labels = pair["entity_pair_type"].astype(str).tolist()
    counts = pair["count"].tolist()
    role_colors = {
        "gene-drug": COLORS["gene_drug"],
        "gene-disease": COLORS["gene_disease"],
    }
    colors = [role_colors.get(l, COLORS["baseline"]) for l in labels]
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
    ax.bar(sub["corpus"], sub["overlap_count"], color=COLORS["kb"], width=0.5)
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
    role_colors = {"gene-drug": COLORS["gene_drug"], "gene-disease": COLORS["gene_disease"]}
    colors = [role_colors.get(k, COLORS["baseline"]) for k in labels]
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
    ax.bar(
        ["Evaluable", "Not evaluable"],
        [n_eval, n_total - n_eval],
        color=[COLORS["gene_drug"], COLORS["baseline"]],
        width=0.5,
    )
    ax.set_ylabel("CIViC primary targets")
    ax.set_title("Abstract coverage of frozen targets")
    add_light_grid(ax, "y")
    n1 = "03_candidate_pool_coverage.png"
    save_figure(fig, fig_dir / n1)
    kept.append(n1)

    import json

    summary = json.loads((out / "03_candidate_pool_entity_type_alignment_summary.json").read_text(encoding="utf-8"))
    n_matched = summary["n_matched_relations"]
    n_miss = 1812 - n_matched
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.barh(
        ["Matched in pool", "No pool positive"],
        [n_matched, n_miss],
        color=[COLORS["benchmark"], COLORS["kb"]],
        height=0.5,
    )
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
            _short_name(row["short_name"]),
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
    ax.barh(y - 0.15, before_vals[: len(y)], height=0.28, label="Before offset gate", color=COLORS["kb"])
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
        s=70,
        color=COLORS["benchmark"],
        edgecolors=COLORS["neutral"],
        linewidths=0.6,
        alpha=0.85,
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
    im = ax.imshow(data, aspect="auto", cmap="Blues", vmin=0.65, vmax=max(0.85, float(np.nanmax(data))))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_short_name(s) for s in pivot.index], fontsize=8)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Encoder")
    ax.set_title("Confirmed matrix: in-distribution benchmark F1")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    n2 = "matrix/matrix_benchmark_f1_heatmap.png"
    save_figure(fig, matrix_dir / "matrix_benchmark_f1_heatmap.png")
    kept.append(n2)
    return kept


def _plot_scatter_panel(ax, encoder_df, ymean, ylo, yhi, title, codes: dict[str, str]) -> None:
    mids = encoder_df["model_id"].tolist()
    x = encoder_df["benchmark_f1_mean"].astype(float).values
    y = encoder_df[ymean].astype(float).values
    xerr = np.array([
        x - encoder_df["benchmark_f1_ci_lo"].astype(float).values,
        encoder_df["benchmark_f1_ci_hi"].astype(float).values - x,
    ])
    yerr = np.array([
        y - encoder_df[ylo].astype(float).values,
        encoder_df[yhi].astype(float).values - y,
    ])
    ax.errorbar(
        x, y, xerr=xerr, yerr=yerr, fmt="none",
        ecolor=COLORS["neutral_light"], elinewidth=0.6, capsize=1.5, alpha=0.9, zorder=1,
    )
    for xi, yi, mid in zip(x, y, mids):
        c = encoder_point_color(mid)
        ax.plot(xi, yi, "o", color=c, markersize=7, zorder=3)
        ax.text(xi, yi, codes[mid], fontsize=8, ha="center", va="center", color="white", fontweight="bold", zorder=4)
    ax.set_xlabel("In-distribution benchmark F1")
    ax.set_ylabel("Out-of-distribution KB MRR")
    ax.set_title(title)


def regenerate_step11() -> list[str]:
    apply_style()
    out = _out_dir("11")
    fig_dir = _fig_dir("11")
    kept: list[str] = []

    enc = pd.read_csv(out / "11_encoder_summary.csv")
    enc = enc.sort_values("benchmark_f1_mean", ascending=False).reset_index(drop=True)
    letters = "ABCDEFGHI"
    codes = {row.model_id: letters[i] for i, row in enumerate(enc.itertuples())}

    xlo = float(enc["benchmark_f1_mean"].min()) - 0.015
    xhi = float(enc["benchmark_f1_mean"].max()) + 0.015

    fig, axes = plt.subplots(1, 2, figsize=FIG_PANEL, sharex=True)
    _plot_scatter_panel(
        axes[0], enc, "kb_mrr_gene_drug_mean",
        "kb_mrr_gene_drug_ci_lo", "kb_mrr_gene_drug_ci_hi", "Gene-drug", codes,
    )
    _plot_scatter_panel(
        axes[1], enc, "kb_mrr_gene_disease_mean",
        "kb_mrr_gene_disease_ci_lo", "kb_mrr_gene_disease_ci_hi", "Gene-disease", codes,
    )
    for ax in axes:
        ax.set_xlim(xlo, xhi)
        add_light_grid(ax, "y")

    legend_lines = [
        f"{codes[row.model_id]} = {_short_name(row.short_name)}"
        for row in enc.itertuples()
    ]
    fig.legend(
        legend_lines, loc="center left", bbox_to_anchor=(1.01, 0.5),
        frameon=False, fontsize=7, handlelength=0, handletextpad=0,
    )
    fig.suptitle("Benchmark vs knowledge-base ranking by pair type (encoder means; seed bars)", y=1.02, fontsize=11)
    fig.subplots_adjust(wspace=0.32, right=0.78)
    n1 = "fig1_benchmark_kb_scatter.png"
    save_figure(fig, fig_dir / n1)
    kept.append(n1)

    var = pd.read_csv(out / "11_variance_components.csv")
    boot = pd.read_csv(out / "11_variance_components_bootstrap.csv")
    metrics = [
        ("benchmark_f1", "Benchmark F1"),
        ("kb_mrr_gene_drug", "KB MRR\ngene-drug"),
        ("kb_mrr_gene_disease", "KB MRR\ngene-disease"),
    ]
    labels, shares, err_lo, err_hi, seed_txt = [], [], [], [], []
    for key, lab in metrics:
        row = var[var["metric"] == key].iloc[0]
        b = boot[boot["metric"] == key].iloc[0]
        enc_share = float(row["encoder_variance_share"])
        seed_share = float(row["seed_variance_share"])
        labels.append(lab)
        shares.append(enc_share)
        err_lo.append(enc_share - float(b["encoder_share_ci_lo"]))
        err_hi.append(float(b["encoder_share_ci_hi"]) - enc_share)
        seed_txt.append(f"seed {100 * seed_share:.0f}%")

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    x = np.arange(len(labels))
    ax.bar(x, shares, width=0.55, color=COLORS["benchmark"], yerr=[err_lo, err_hi], capsize=3,
           error_kw={"elinewidth": 0.8, "ecolor": COLORS["neutral"]})
    for i, (xi, yi, txt) in enumerate(zip(x, shares, seed_txt)):
        ax.text(xi, yi + err_hi[i] + 0.04, txt, ha="center", va="bottom", fontsize=8, color=COLORS["neutral"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Between-encoder variance share")
    ax.set_ylim(0, max(shares) + max(err_hi) + 0.15)
    ax.set_title("Discriminative power: between-encoder share by axis")
    add_light_grid(ax, "y")
    n2 = "fig2_variance_between_encoder.png"
    save_figure(fig, fig_dir / n2)
    kept.append(n2)

    easy_hard = pd.read_csv(out / "11_easy_hard_ranking.csv")
    encoder_order = enc["model_id"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=FIG_PANEL)
    for ax, (subset_key, title) in zip(
        axes,
        [("easy_co_sentence", "Co-sentence (easy)"), ("hard_cross_sentence", "Cross-sentence (hard)")],
    ):
        sub = easy_hard[easy_hard["subset"] == subset_key]
        dr = float(sub[sub["model_id"] == "distance_ranker"]["mrr"].iloc[0])
        rows = []
        for mid in encoder_order:
            runs = sub[(sub["model_id"] == mid) & (sub["model_id"] != "distance_ranker")]
            if runs.empty:
                continue
            rows.append({"model_id": mid, "mrr": float(runs["mrr"].mean()), "sd": float(runs["mrr"].std(ddof=1)) if len(runs) > 1 else 0.0})
        edf = pd.DataFrame(rows)
        y = np.arange(len(edf))
        ax.errorbar(edf["mrr"], y, xerr=edf["sd"], fmt="none", ecolor=COLORS["neutral_light"], elinewidth=0.6, capsize=1.5)
        for xi, yi, mid in zip(edf["mrr"], y, edf["model_id"]):
            ax.plot(xi, yi, "o", color=encoder_point_color(mid), markersize=7)
        ax.axvline(dr, color=COLORS["baseline"], linestyle="--", linewidth=1.5, label=f"Distance ranker ({dr:.3f})")
        ax.set_yticks(y)
        ax.set_yticklabels([_short_name(MODEL_BY_ID[m].short_name) for m in edf["model_id"]], fontsize=8)
        mrr_vals = edf["mrr"].tolist() + [dr]
        ax.set_xlim(max(0.0, min(mrr_vals) - 0.04), min(1.0, max(mrr_vals) + 0.06))
        ax.set_xlabel("Mean reciprocal rank")
        ax.set_title(title)
        ax.legend(loc="lower right", frameon=False, fontsize=7)
        add_light_grid(ax, "x")
    fig.suptitle("Ranking validity vs proximity-only baseline", y=1.02, fontsize=11)
    fig.subplots_adjust(wspace=0.38, left=0.22)
    n3 = "fig3_easy_hard_ranking_validity.png"
    save_figure(fig, fig_dir / n3)
    kept.append(n3)

    lift = pd.read_csv(out / "11_untrained_floor_lift.csv")
    order = lift.sort_values("lift_benchmark_f1", ascending=True)
    y = np.arange(len(order))
    bench_lift = order["lift_benchmark_f1"].astype(float).values
    kb_lift = ((order["lift_kb_mrr_gene_drug"] + order["lift_kb_mrr_gene_disease"]) / 2).astype(float).values
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.barh(y - 0.18, bench_lift, height=0.34, label="Benchmark F1 lift", color=COLORS["benchmark"])
    ax.barh(y + 0.18, kb_lift, height=0.34, label="KB MRR lift (pair-type mean)", color=COLORS["kb"])
    ax.axvline(0, color=COLORS["neutral"], linewidth=0.8, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([_short_name(s) for s in order["short_name"]], fontsize=8)
    ax.set_xlabel("Fine-tuned minus untrained floor")
    ax.set_title("Fine-tuning lift by axis")
    ax.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.28)
    add_light_grid(ax, "x")
    n4 = "fig4_finetuning_lift.png"
    save_figure(fig, fig_dir / n4)
    kept.append(n4)

    for old in fig_dir.glob("fig*.png"):
        if old.name not in kept:
            _remove(old)
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
    n1 = "fig1_within_seed_paired_change.png"
    save_figure(fig, fig_dir / n1)
    kept.append(n1)

    pt = pd.read_csv(out / "20_pair_type_breakdown.csv")
    sub = pt[pt["well_trained_definition"] == "val_f1_best"]
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    pair_order = ["gene-drug", "gene-disease"]
    sub = sub.set_index("pair_type").reindex(pair_order).reset_index()
    colors = [COLORS["gene_drug"], COLORS["gene_disease"]]
    ax.bar(sub["pair_type"], sub["mean_delta_kb_mrr"], color=colors, width=0.5)
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.8)
    ax.set_ylabel("Mean KB MRR change")
    ax.set_title("Pair-type asymmetry (best val F1 checkpoint)")
    add_light_grid(ax, "y")
    n2 = "fig2_pair_type_asymmetry.png"
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
    n3 = "fig3_gene_disease_subset_contrast.png"
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
    "11": regenerate_step11,
    "20": regenerate_step20,
}
