"""Regenerate budget-limited figures in shared style."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared.models import MODELS, MODEL_BY_ID
from shared.plot_style import (
    COLORS,
    ENCODER_COLORS,
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


def _plot_scatter_panel(ax, encoder_df, ymean, ylo, yhi, title) -> None:
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
    for i, mid in enumerate(mids):
        c = encoder_point_color(mid)
        ax.errorbar(
            x[i], y[i], xerr=xerr[:, i : i + 1], yerr=yerr[:, i : i + 1],
            fmt="o", color=c, ecolor=c, elinewidth=0.7, capsize=2,
            markersize=7, alpha=0.92, zorder=3,
        )
    ax.set_xlabel("In-distribution benchmark F1")
    ax.set_ylabel("Out-of-distribution KB MRR")
    ax.set_title(title)


def _encoder_legend(fig, model_ids: list[str], y_anchor: float = 0.02) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=encoder_point_color(mid),
            markeredgecolor=COLORS["neutral"], markeredgewidth=0.35, markersize=5.5,
            label=_short_name(MODEL_BY_ID[mid].short_name if mid in MODEL_BY_ID else mid),
        )
        for mid in model_ids
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y_anchor),
        ncol=5,
        frameon=False,
        fontsize=7,
        columnspacing=0.8,
        handletextpad=0.3,
    )


def regenerate_step11() -> list[str]:
    apply_style()
    out = _out_dir("11")
    fig_dir = _fig_dir("11")
    kept: list[str] = []

    enc = pd.read_csv(out / "11_encoder_summary.csv")
    enc = enc.sort_values("benchmark_f1_mean", ascending=False).reset_index(drop=True)
    model_ids = enc["model_id"].tolist()

    xlo = float(enc["benchmark_f1_mean"].min()) - 0.018
    xhi = float(enc["benchmark_f1_mean"].max()) + 0.018

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True)
    _plot_scatter_panel(
        axes[0], enc, "kb_mrr_gene_drug_mean",
        "kb_mrr_gene_drug_ci_lo", "kb_mrr_gene_drug_ci_hi", "Gene-drug",
    )
    _plot_scatter_panel(
        axes[1], enc, "kb_mrr_gene_disease_mean",
        "kb_mrr_gene_disease_ci_lo", "kb_mrr_gene_disease_ci_hi", "Gene-disease",
    )
    for ax in axes:
        ax.set_xlim(xlo, xhi)
        add_light_grid(ax, "y")

    fig.suptitle(
        "Benchmark vs knowledge-base ranking by pair type\n(encoder means; seed uncertainty bars)",
        y=0.98, fontsize=11,
    )
    fig.subplots_adjust(wspace=0.22, bottom=0.26, top=0.82, left=0.10, right=0.98)
    _encoder_legend(fig, model_ids, y_anchor=0.02)
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
    bars = ax.bar(x, shares, width=0.55, color=COLORS["benchmark"])
    ax.errorbar(
        x, shares, yerr=[err_lo, err_hi], fmt="none", capsize=3,
        ecolor=COLORS["neutral"], elinewidth=0.8,
    )
    for i, (xi, yi, txt) in enumerate(zip(x, shares, seed_txt)):
        ax.text(
            xi, yi + err_hi[i] + 0.02, f"{100 * yi:.0f}%",
            ha="center", va="bottom", fontsize=8, color=COLORS["benchmark"],
        )
        ax.text(xi, -0.14, txt, ha="center", va="top", fontsize=7, color=COLORS["neutral"],
                transform=ax.get_xaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Between-encoder variance share")
    ax.set_ylim(0, max(shares) + max(err_hi) + 0.12)
    ax.set_title("Discriminative power: between-encoder share by axis")
    add_light_grid(ax, "y")
    fig.subplots_adjust(bottom=0.22)
    n2 = "fig2_variance_between_encoder.png"
    save_figure(fig, fig_dir / n2)
    kept.append(n2)

    easy_hard = pd.read_csv(out / "11_easy_hard_ranking.csv")
    encoder_order = model_ids
    subsets = [("easy_co_sentence", "Co-sentence (easy)"), ("hard_cross_sentence", "Cross-sentence (hard)")]
    dr_by_subset: dict[str, float] = {}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8))
    for ax, (subset_key, title) in zip(axes, subsets):
        sub = easy_hard[easy_hard["subset"] == subset_key]
        dr = float(sub[sub["model_id"] == "distance_ranker"]["mrr"].iloc[0])
        dr_by_subset[subset_key] = dr
        rows = []
        for mid in encoder_order:
            runs = sub[sub["model_id"] == mid]
            if runs.empty:
                continue
            rows.append({
                "model_id": mid,
                "mrr": float(runs["mrr"].mean()),
                "sd": float(runs["mrr"].std(ddof=1)) if len(runs) > 1 else 0.0,
            })
        edf = pd.DataFrame(rows)
        y = np.arange(len(edf))
        for xi, yi, sd, mid in zip(edf["mrr"], y, edf["sd"], edf["model_id"]):
            c = encoder_point_color(mid)
            ax.errorbar(xi, yi, xerr=sd, fmt="o", color=c, ecolor=c, elinewidth=0.7,
                        capsize=2, markersize=7, zorder=3)
        ax.axvline(dr, color=COLORS["baseline"], linestyle="--", linewidth=1.4, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([_short_name(MODEL_BY_ID[m].short_name) for m in edf["model_id"]], fontsize=8)
        mrr_vals = edf["mrr"].tolist() + [dr]
        xmin = max(0.0, min(mrr_vals) - 0.05)
        xmax = min(1.0, max(mrr_vals) + 0.08)
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("Mean reciprocal rank")
        ax.set_title(title)
        add_light_grid(ax, "x")

    dr_easy = dr_by_subset["easy_co_sentence"]
    dr_hard = dr_by_subset["hard_cross_sentence"]
    fig.suptitle("Ranking validity vs proximity-only baseline", y=0.98, fontsize=11)
    fig.text(
        0.5, 0.13,
        f"Dashed line: distance ranker (easy {dr_easy:.3f}, hard {dr_hard:.3f})",
        ha="center", va="center", fontsize=8, color=COLORS["baseline"],
    )
    fig.subplots_adjust(wspace=0.32, left=0.20, bottom=0.24, top=0.88)
    _encoder_legend(fig, encoder_order, y_anchor=0.02)
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
    """Regenerate native step-20 figures; remove legacy manuscript-regen duplicates."""
    import sys
    from importlib import import_module

    fig_dir = _fig_dir("20")
    for legacy in (
        "fig1_within_seed_paired_change.png",
        "fig2_pair_type_asymmetry.png",
        "fig3_gene_disease_subset_contrast.png",
    ):
        _remove(fig_dir / legacy)

    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    out = _out_dir("20")
    cfg = import_module("20_round2_diagnostic.config")
    adj = import_module("20_round2_diagnostic.adjudication")
    mundane_mod = import_module("20_round2_diagnostic.mundane_explanations")
    enc_mod = import_module("20_round2_diagnostic.encoder_correlation")
    qual_mod = import_module("20_round2_diagnostic.qualitative_errors")
    fig_mod = import_module("20_round2_diagnostic.figures")

    results = adj.run_adjudication_analysis()
    mundane = mundane_mod.run_mundane_explanations(
        results["trajectory"],
        results["paired"],
        skip_stratum_inference=True,
    )
    stratum_summary = (
        pd.read_csv(cfg.POOL_STRATUM_SUMMARY_CSV)
        if cfg.POOL_STRATUM_SUMMARY_CSV.exists()
        else pd.DataFrame()
    )
    gd_enc = results.get("gene_disease", {}).get("encoder")
    encoder_corr = enc_mod.run_encoder_correlation(
        gd_enc if gd_enc is not None else __import__("pandas").DataFrame(),
        results["trajectory"],
    )
    qual = qual_mod.run_qualitative_errors()

    fig_mod.generate_all_figures(
        results["trajectory"],
        results["paired"],
        results["hard_easy"],
        results["pair_type"],
        results["robustness"],
        results.get("gene_disease", {}).get("pair_subset"),
        mundane.get("timing_summary"),
        stratum_summary if not stratum_summary.empty else None,
        encoder_corr.get("table"),
        qual.get("patterns"),
        qual.get("summary"),
    )
    kept = sorted(p.name for p in fig_dir.glob("fig*.png"))
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
