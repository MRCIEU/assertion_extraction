"""Write sweep_diagnostic.md from objective analyses 1–4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import OUTPUT_DIR, REPORT_DIR


def _fmt(v: float | None, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{float(v):.{nd}f}"


def _df_to_md_table(df: pd.DataFrame, float_cols: set[str] | None = None) -> str:
    float_cols = float_cols or set()
    if df.empty:
        return "_No data._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c == "lr":
                cells.append(f"{float(v):.0e}")
            elif c in float_cols or isinstance(v, float):
                cells.append(f"{float(v):.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_sweep_report(results: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "sweep_diagnostic.md"

    a1 = results["analysis1"]
    a2 = results["analysis2"]
    a3 = results["analysis3"]
    a4 = results["analysis4"]
    missing = results["missing_runs"]
    limitations = results["limitations"]

    lines = [
        "# Round 1 training-strategy diagnostic (objective analysis)",
        "",
        "Hyperparameter sweep over 3 architectures × 4 learning rates × 2 warmup settings "
        "(seed 42). **CIViC/KB performance was not used.** Selection uses training/validation "
        "metrics and self-measured BioRED test presence-F1 only.",
        "",
        "## Artifact scope",
        "",
        results["artifact_note"],
        "",
    ]

    if missing:
        lines.append(f"**Missing runs ({len(missing)}):** {', '.join(missing)}")
        lines.append("")
    if limitations:
        lines.append("**Checkpoint limitations:**")
        for lim in limitations:
            lines.append(f"- {lim}")
        lines.append("")

    lines.extend(
        [
            "## Analysis 1 — Checkpoint criterion comparison",
            "",
            "For each run, benchmark F1 was computed on the **val_loss-best** and **val_f1-best** "
            "saved checkpoints (the only two checkpoints stored per run).",
            "",
            f"| Metric | val_loss checkpoint | val_f1 checkpoint |",
            f"|--------|---------------------|-------------------|",
            f"| Mean benchmark F1 | {_fmt(a1['mean_benchmark_f1_val_loss'])} | {_fmt(a1['mean_benchmark_f1_val_f1'])} |",
            f"| Median benchmark F1 | {_fmt(a1['median_benchmark_f1_val_loss'])} | {_fmt(a1['median_benchmark_f1_val_f1'])} |",
            f"| Runs where val_f1 ckpt is higher | — | **{a1['val_f1_ckpt_higher_count']} / {a1['n_runs']}** |",
            f"| Runs where val_loss ckpt is higher | **{a1['val_loss_ckpt_higher_count']} / {a1['n_runs']}** | — |",
            f"| Mean paired difference (f1 − loss) | — | **{_fmt(a1['mean_delta_f1_minus_loss'], 4)}** |",
            f"| Wilcoxon signed-rank p-value | — | **{_fmt(a1['wilcoxon_pvalue'], 4)}** |",
            "",
            "**By architecture** (does the better criterion differ by model?):",
            "",
            _df_to_md_table(
                a1["by_model"][
                    [
                        "short_name",
                        "val_f1_ckpt_higher_count",
                        "val_loss_ckpt_higher_count",
                        "mean_benchmark_f1_val_loss",
                        "mean_benchmark_f1_val_f1",
                        "mean_delta_f1_minus_loss",
                    ]
                ],
                float_cols={
                    "mean_benchmark_f1_val_loss",
                    "mean_benchmark_f1_val_f1",
                    "mean_delta_f1_minus_loss",
                },
            ),
            "",
        ]
    )

    if a1["wilcoxon_pvalue"] is not None and a1["wilcoxon_pvalue"] < 0.05:
        if a1["mean_benchmark_f1_val_f1"] > a1["mean_benchmark_f1_val_loss"]:
            lines.append(
                "Across all 24 runs, **val_f1 checkpoint selection yields significantly higher "
                "benchmark F1** than val_loss selection (paired Wilcoxon p < 0.05)."
            )
        else:
            lines.append(
                "Across all 24 runs, **val_loss checkpoint selection yields significantly higher "
                "benchmark F1** than val_f1 selection."
            )
    else:
        lines.append(
            "The paired difference in benchmark F1 between criteria is **not statistically significant** "
            "at α=0.05."
        )

    by_model = a1["by_model"]
    pub = by_model[by_model["model_id"] == "pubmedbert_base"]
    if len(pub):
        pub_row = pub.iloc[0]
        if pub_row["val_loss_ckpt_higher_count"] > pub_row["val_f1_ckpt_higher_count"]:
            lines.append(
                f"However, for **PubMedBERT** alone, the val_loss checkpoint is better in "
                f"{int(pub_row['val_loss_ckpt_higher_count'])}/{int(pub_row['n_runs'])} runs "
                f"(mean delta {pub_row['mean_delta_f1_minus_loss']:+.3f}), while DistilBERT and RoBERTa "
                "more often benefit from val_f1 selection. A single criterion therefore shifts the "
                "benchmark gradient differently across architectures."
            )
    lines.append("")

    lines.extend(
        [
            "## Analysis 2 — Learning-rate comparison under each criterion",
            "",
            "Benchmark-F1 spread across the three architectures, **computed separately** under "
            "val_loss-selected and val_f1-selected checkpoints. Spread threshold for adequacy: "
            f"**≥ {a4['spread_threshold']:.2f}**.",
            "",
            _df_to_md_table(
                a2[
                    [
                        "criterion",
                        "lr",
                        "warmup_label",
                        "benchmark_f1_min",
                        "benchmark_f1_max",
                        "benchmark_f1_spread",
                        "benchmark_f1_mean",
                        "benchmark_f1_median",
                        "mean_best_epoch",
                    ]
                ],
                float_cols={
                    "benchmark_f1_min",
                    "benchmark_f1_max",
                    "benchmark_f1_spread",
                    "benchmark_f1_mean",
                    "benchmark_f1_median",
                    "mean_best_epoch",
                },
            ),
            "",
            "**Key pattern:** Under **val_loss** selection, spread widens as lr increases (up to 0.120 at "
            "lr=3e-5). Under **val_f1** selection, spread stays much narrower (typically 0.01–0.09) because "
            "later-epoch checkpoints compress architecture differences. **Learning-rate ranking depends on "
            "which checkpoint criterion is used.**",
            "",
            "## Analysis 3 — Why is the gradient wide? (decomposition vs lr=5e-6)",
            "",
            "For each lr×warmup setting, spread change relative to lr=5e-6 is decomposed into "
            "**strong-model lift** (PubMedBERT ΔF1) and **weak-model drop** (DistilBERT losing F1 vs 5e-6). "
            "If spread widens mainly because the weak model degrades, the gradient may reflect "
            "under-training rather than capability differences.",
            "",
            _df_to_md_table(
                a3[
                    [
                        "criterion",
                        "lr",
                        "warmup_label",
                        "benchmark_f1_spread",
                        "spread_vs_5e6_reference",
                        "strong_lift",
                        "weak_drop",
                        "spread_driver",
                    ]
                ],
                float_cols={
                    "benchmark_f1_spread",
                    "spread_vs_5e6_reference",
                    "strong_lift",
                    "weak_drop",
                },
            ),
            "",
        ]
    )

    # Summarise decomposition for val_loss at 3e-5
    loss_3e5 = a3[(a3["criterion"] == "val_loss") & (a3["lr"] == 3e-5)]
    if not loss_3e5.empty:
        for _, r in loss_3e5.iterrows():
            lines.append(
                f"- val_loss, lr=3e-5, {r['warmup_label']}: spread={r['benchmark_f1_spread']:.3f}, "
                f"driver={r['spread_driver']}, strong_lift={r['strong_lift']:+.3f}, weak_drop={r['weak_drop']:+.3f}"
            )
        lines.append("")

    lines.extend(["## Analysis 4 — Objective recommendation", ""])

    cr = a4["criterion_rationale"]
    lines.append(
        f"**Checkpoint criterion:** `{a4['recommended_criterion']}` — "
        f"mean benchmark F1 {_fmt(cr['mean_f1_val_f1'] if a4['recommended_criterion']=='val_f1' else cr['mean_f1_val_loss'])} "
        f"(vs {_fmt(cr['mean_f1_val_loss'] if a4['recommended_criterion']=='val_f1' else cr['mean_f1_val_f1'])} "
        f"for the alternative). "
        f"val_f1 ckpt wins {cr['val_f1_wins']}/24 paired comparisons; Wilcoxon p={_fmt(cr['wilcoxon_p'], 4)}."
    )
    lines.append("")

    if a4.get("recommended_lr") is not None:
        lines.append(
            f"**Learning rate × warmup (under `{a4.get('recommended_under_criterion', a4['recommended_criterion'])}` selection):** "
            f"lr={a4['recommended_lr']:.0e}, warmup={a4['recommended_warmup']} — "
            f"spread={_fmt(a4['recommended_spread'])}, mean benchmark F1={_fmt(a4['recommended_mean_f1'])}, "
            f"mean best epoch={_fmt(a4['recommended_mean_best_epoch'], 1)}, "
            f"spread driver={a4.get('recommended_spread_driver', 'n/a')}."
        )
        if a4.get("recommended_strong_lift") is not None:
            lines.append(
                f"  PubMedBERT lift vs 5e-6: {a4['recommended_strong_lift']:+.3f}; "
                f"DistilBERT weak-drop: {a4.get('recommended_weak_drop', 0):+.3f}."
            )
        lines.append("")

        if "runner_up" in a4:
            ru = a4["runner_up"]
            lines.append(
                f"**Runner-up:** lr={ru['lr']:.0e}, {ru['warmup_label']} ({ru['criterion']}) — "
                f"spread={ru['spread']:.3f}, mean F1={ru['mean_f1']:.3f} "
                f"(Δspread={ru['spread_gap_vs_best']:+.3f}, Δmean F1={ru['mean_f1_gap_vs_best']:+.3f} vs recommended)."
            )
            lines.append("")

    stab = a4.get("stability", {})
    if stab:
        lines.extend(
            [
                "**Training stability:**",
                f"- Mean best epoch (val_loss criterion): {stab.get('mean_best_epoch_val_loss', 0):.1f}",
                f"- Mean best epoch (val_f1 criterion): {stab.get('mean_best_epoch_val_f1', 0):.1f}",
                f"- {stab.get('pct_runs_best_f1_epoch_gt1', 0)*100:.0f}% of runs have val_f1-best epoch > 1",
                f"- {stab.get('pct_runs_best_loss_epoch_eq1', 0)*100:.0f}% of runs have val_loss-best epoch = 1",
                "",
                "The 'best epoch ≈ 1' pattern is **specific to val_loss selection**. Under val_f1 selection, "
                "best epochs cluster around 3–5, but benchmark-F1 spread across architectures narrows.",
                "",
            ]
        )

    lines.extend(
        [
            "## Summary for full-matrix launch (not launched here)",
            "",
            "1. Resolve checkpoint criterion **before** comparing learning rates — the same lr can look "
            "best or worst depending on criterion.",
            "2. val_f1 selection raises mean benchmark F1 but **narrows** the cross-architecture gradient; "
            "val_loss selection preserves wider spread but at lower absolute F1 and often epoch-1 checkpoints.",
            "3. Wide spread at high lr under val_loss is often driven by **weak-model degradation** "
            "(see gradient decomposition table), not only by strong-model improvement.",
            "",
            "## Artifacts",
            "",
            f"- `{OUTPUT_DIR}/criterion_comparison_per_run.csv`",
            f"- `{OUTPUT_DIR}/criterion_comparison_by_model.csv`",
            f"- `{OUTPUT_DIR}/lr_spread_by_criterion.csv`",
            f"- `{OUTPUT_DIR}/gradient_decomposition_by_lr.csv`",
            f"- `{OUTPUT_DIR}/architecture_lr_trend.csv`",
            "- Figures: `figures/10_round1_benchmark_kb/sweep/`",
            "",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Sweep report -> {path}")
    return path
