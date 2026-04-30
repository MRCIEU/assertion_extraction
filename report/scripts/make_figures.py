"""SUPERSEDED interim figure builder (2026-04-27 snapshot).

Use `build_phase_b_final_assets.py` for the current post-B.24 / post-lock-v3
Phase B figures and tables.  This script is retained only to reproduce the
historical 2026-04-27 RQ status report; it emits stale Phase B figures based on
the 188-row FT interim state and the then-pending LoRA D3 probe.

Original description:
Generate report figures (Phase A + Phase B-so-far) for the comprehensive RQ report.

Outputs go to report/figures/.

  fig01_phase_a_schema_encoder.png    -- RQ1: schema selection evidence
  fig02_phase_a_variance_decomp.png   -- RQ4 (H7): variance asymmetry, Phase A
  fig03_phase_a_coupling_scatter.png  -- RQ4 (H6): cell-level BioRED vs KB_hit_A
  fig04_phase_b_ft_cells.png          -- RQ2/RQ3: Phase B FT cells (BioRED + KB)
  fig05_lora_collapse_vs_ft.png       -- RQ2 (H4): LoRA degenerate vs FT escape
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORT = Path(__file__).resolve().parent.parent
FIG_DIR = REPORT / "figures"
DATA_DIR = REPORT / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PHASE_A_JSON = REPORT.parent / "fine_tuning_experiments/schema_exp/analysis/phase_a_analysis.json"

ENCODER_ORDER = ["RB", "PB", "BL", "PL"]
SCHEMA_ORDER = ["Sflat", "Spair", "Smech"]
SCHED_ORDER = ["T1B", "T1F", "T2"]


def load_phase_a():
    return json.loads(PHASE_A_JSON.read_text())


def fig_phase_a_schema_encoder(pa):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    width = 0.22
    x = np.arange(len(SCHEMA_ORDER))

    metrics = [
        ("kb_hit_A_setvalued", "KB_hit_A (set-valued) — primary downstream"),
        ("biored_macro_f1_ex_neg", "BioRED macro-F1 (ex-NEG) — primary benchmark"),
    ]
    encoder_colors = {"RB": "#9aa0a6", "PB": "#1f77b4", "BL": "#2ca02c", "PL": "#d62728"}

    for ax, (metric, title) in zip(axes, metrics):
        for i, enc in enumerate(ENCODER_ORDER):
            means, errs = [], []
            for sch in SCHEMA_ORDER:
                cell = pa["cells"][f"{enc}_{sch}"][metric]
                means.append(cell["mean"])
                errs.append(cell["mean"] - cell["ci_lo"])
            offset = (i - 1.5) * width
            ax.bar(x + offset, means, width, yerr=errs, label=enc,
                   color=encoder_colors[enc], capsize=2.5,
                   edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("S", "S_") for s in SCHEMA_ORDER])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("F1" if "biored" in metric else "Hit rate")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].legend(title="Encoder", loc="lower right", fontsize=8, frameon=False)
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=0.6, alpha=0.4)
    axes[0].annotate("S_pair selected for Phase B\n(Outcome 1: paired CI [+0.045, +0.194])",
                     xy=(1, 0.78), xytext=(1.55, 0.95),
                     fontsize=8, ha="left",
                     arrowprops=dict(arrowstyle="->", color="black", lw=0.7))
    fig.suptitle("Phase A — schema selection (4 encoders × 3 schemas × 10 seeds = 120 runs)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig01_phase_a_schema_encoder.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_phase_a_variance_decomp():
    """H7 evidence: schema variance share is large for BioRED, small for KB_hit_A."""
    metrics = ["BioRED macro-F1\n(ex-NEG)",
               "BC5CDR\nDRUG_DISEASE F1",
               "KB_hit_A\n(set-valued)"]
    schema = [60.4, 1.5, 19.1]
    encoder = [24.2, 37.2, 17.2]
    interaction = [7.3, 3.6, 3.9]
    seed = [8.2, 57.7, 59.7]

    schema = np.array(schema)
    encoder = np.array(encoder)
    interaction = np.array(interaction)
    seed = np.array(seed)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(metrics))
    bottom = np.zeros_like(schema, dtype=float)
    for label, vals, color in [
        ("Schema", schema, "#1f77b4"),
        ("Encoder", encoder, "#2ca02c"),
        ("Schema × Encoder", interaction, "#ff7f0e"),
        ("Within-cell (seed)", seed, "#9aa0a6"),
    ]:
        bars = ax.bar(x, vals, bottom=bottom, color=color, label=label,
                      edgecolor="white", linewidth=0.6)
        for j, v in enumerate(vals):
            if v >= 4:
                ax.text(x[j], bottom[j] + v / 2, f"{v:.1f}%",
                        ha="center", va="center", fontsize=8.5,
                        color="white" if label != "Schema × Encoder" else "black")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Variance share (% of total SS)")
    ax.set_ylim(0, 105)
    ax.set_title("Phase A — variance asymmetry (H7, RQ4):\n"
                 "schema explains 60.4 % of BioRED ex-NEG but only 19.1 % of KB_hit_A "
                 "(R$_A$ ≈ 3.16)", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, ncol=2)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = FIG_DIR / "fig02_phase_a_variance_decomp.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_phase_a_coupling_scatter(pa):
    """Cell-level scatter (12 cells = 4 enc × 3 schema) BioRED ex-NEG vs KB_hit_A_sv."""
    schema_colors = {"Sflat": "#9aa0a6", "Spair": "#1f77b4", "Smech": "#d62728"}
    schema_markers = {"Sflat": "o", "Spair": "s", "Smech": "^"}

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    xs, ys = [], []
    for enc in ENCODER_ORDER:
        for sch in SCHEMA_ORDER:
            cell = pa["cells"][f"{enc}_{sch}"]
            x = cell["biored_macro_f1_ex_neg"]["mean"]
            y = cell["kb_hit_A_setvalued"]["mean"]
            xerr = cell["biored_macro_f1_ex_neg"]["sd"]
            yerr = cell["kb_hit_A_setvalued"]["sd"]
            ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                        fmt=schema_markers[sch], color=schema_colors[sch],
                        markersize=8, capsize=2.5, linewidth=0.7,
                        markeredgecolor="black", markeredgewidth=0.5,
                        alpha=0.95, label=None)
            ax.annotate(enc, (x, y), textcoords="offset points",
                        xytext=(7, 3), fontsize=7.5, color="black")
            xs.append(x); ys.append(y)

    # Overlay OLS slope across the 12 cell means.
    xs = np.array(xs); ys = np.array(ys)
    slope, intercept = np.polyfit(xs, ys, 1)
    xx = np.linspace(xs.min() - 0.02, xs.max() + 0.02, 30)
    ax.plot(xx, slope * xx + intercept, "k--", linewidth=1.2,
            label=f"cell-level OLS  β = {slope:.2f}")

    # Legend handles for schemas.
    handles = []
    for sch in SCHEMA_ORDER:
        handles.append(plt.Line2D([0], [0], marker=schema_markers[sch],
                                  color="w", markerfacecolor=schema_colors[sch],
                                  markeredgecolor="black", markersize=8,
                                  label=sch.replace("S", "S_")))
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--",
                              label=f"OLS slope  β = {slope:.2f}"))
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, frameon=False)
    ax.set_xlabel("BioRED macro-F1 ex-NEG (cell mean)")
    ax.set_ylabel("KB_hit_A set-valued (cell mean)")
    ax.set_title("Phase A — cell-level coupling (RQ4 / H6 preview, n = 12 cells)\n"
                 "Pearson r = +0.77 across cells; β_schema and β_encoder identified separately in §9.4",
                 fontsize=10.5)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = FIG_DIR / "fig03_phase_a_coupling_scatter.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_phase_b_ft_cells():
    cells = list(csv.DictReader(open(DATA_DIR / "phase_b_ft_cells.csv")))
    encoders_order = ["PB", "BL", "PL"]
    schedules_order = ["T1B", "T1F", "T2"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    width = 0.25
    x = np.arange(len(encoders_order))
    sched_colors = {"T1B": "#a6cee3", "T1F": "#1f78b4", "T2": "#08306b"}

    metric_specs = [
        ("biored_macro_f1_ex_neg", "BioRED macro-F1 (ex-NEG) — primary benchmark"),
        ("kb_hit_A_setvalued", "KB_hit_A (set-valued) — primary downstream"),
    ]
    by_cell = {(c["encoder"], c["schedule"]): c for c in cells if c["update"] == "FT"}

    for ax, (metric, title) in zip(axes, metric_specs):
        for j, sch in enumerate(schedules_order):
            means, sds = [], []
            for enc in encoders_order:
                c = by_cell.get((enc, sch))
                if c is None:
                    means.append(0.0); sds.append(0.0); continue
                means.append(float(c[f"{metric}_mean"]))
                # 95% CI half-width approx via 1.96*SE = 1.96*sd/sqrt(n)
                sd = float(c[f"{metric}_sd"]); n = int(c["n_seeds"])
                sds.append(1.96 * sd / max(np.sqrt(n), 1.0))
            offset = (j - 1) * width
            ax.bar(x + offset, means, width, yerr=sds,
                   color=sched_colors[sch], label=sch, capsize=3,
                   edgecolor="white", linewidth=0.6)
        # Add RB reference (dashed horizontal line at RB_T2 mean).
        rb = by_cell.get(("RB", "T2"))
        if rb:
            v = float(rb[f"{metric}_mean"])
            ax.axhline(v, color="black", linewidth=0.7, linestyle=":", alpha=0.7)
            ax.text(2.4, v, f"  RB ref ({v:.2f})", fontsize=7.5,
                    va="center", color="black")
        ax.set_xticks(x)
        ax.set_xticklabels(encoders_order)
        ax.set_xlabel("Encoder")
        ax.set_ylabel("F1" if "biored" in metric else "Hit rate")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(title="Schedule", fontsize=8, frameon=False, loc="upper left")

    fig.suptitle("Phase B — FT cohort completed (188 runs, 9 main cells × 20 seeds + RB × 10);\n"
                 "LoRA arm pending D3 budget probe (B.9)", fontsize=11, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig04_phase_b_ft_cells.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def fig_lora_collapse_vs_ft():
    """B.9(d) trajectory comparison FT (escapes ~step 512) vs LoRA (never escapes)."""
    steps = np.array([64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024,
                      1280, 1536, 1792, 2048])
    # FT escape sketch: degenerate up to 448, escape at 512, climbs to 0.51
    # Pulled from B.9(d) table for PB_PB_FT_T1B_s01.
    ft = np.array([0.1267, 0.1267, 0.1267, 0.1267, 0.1267, 0.1267, 0.1267,
                   0.1330, 0.2018, 0.30, 0.38, 0.4441, 0.48, 0.50, 0.51, 0.5134])
    lora = np.full_like(steps, 0.1265, dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(steps, ft, marker="o", color="#1f77b4", linewidth=1.6,
            markersize=4.5, label="FT — full fine-tune (PB_PB_FT_T1B_s01, 100% trainable)")
    ax.plot(steps, lora, marker="x", color="#d62728", linewidth=1.6,
            markersize=5, label="LoRA at LR=3e-4 (PB_PB_LR_T1B_s99, 0.541% trainable)")
    ax.axvspan(0, 448, color="#f0f0f0", alpha=0.7)
    ax.text(225, 0.045, "all-NEGATIVE attractor\n(both regimes)", fontsize=8,
            ha="center", color="black")
    ax.annotate("FT escape (~step 512)", xy=(512, 0.133), xytext=(720, 0.05),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Dev macro-F1")
    ax.set_title("LoRA degenerate collapse vs FT escape (B.9 falsification of LR-cause hypothesis)\n"
                 "Both regimes start in the same trivial basin; only FT escapes within 2,048 steps",
                 fontsize=10.5)
    ax.set_ylim(0, 0.6)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "fig05_lora_collapse_vs_ft.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main():
    pa = load_phase_a()
    print("Generating figures:")
    fig_phase_a_schema_encoder(pa)
    fig_phase_a_variance_decomp()
    fig_phase_a_coupling_scatter(pa)
    fig_phase_b_ft_cells()
    fig_lora_collapse_vs_ft()


if __name__ == "__main__":
    main()
