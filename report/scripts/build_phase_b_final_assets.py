#!/usr/bin/env python3.11
"""Build final Phase B report assets from frozen analysis outputs.

Inputs are the authoritative post-lock outputs under
`fine_tuning_experiments/phase_b/analysis/output/*LATEST*`.

Generated artefacts:
  report/data/phase_b_ft_seedlevel.csv
  report/data/phase_b_ft_cells.csv
  report/tables/table02_phase_b_cell_results.{csv,md}
  report/tables/table03_phase_b_hypothesis_summary.{csv,md}
  report/tables/table04_rq_evidence_matrix.{csv,md}
  report/figures/fig02_phase_b_benchmark.png
  report/figures/fig03_kb_surfacing_profiles.png
  report/figures/fig04_h7_variance_ordinal.png
  report/figures/fig05_h6_slopes.png
  report/figures/fig06_lora_collapse_audit.png

This script intentionally does not overwrite fig01, the Phase A schema
selection figure, which remains the RQ1 anchor.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
REPORT = PROJECT / "report"
DATA_DIR = REPORT / "data"
FIG_DIR = REPORT / "figures"
TABLE_DIR = REPORT / "tables"
AN_OUT = PROJECT / "fine_tuning_experiments/phase_b/analysis/output"
RUN_ROOT = Path("/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b")
ARCHIVE_ROOT = Path("/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b_degenerate_lr_archive")

for d in (DATA_DIR, FIG_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

ENCODERS = ["PB", "BL", "PL"]
SCHEDULES = ["T1B", "T1F", "T2"]
KB_METRICS = [
    ("kb_hit_A_setvalued", "KB_hit_A"),
    ("kb_pmass_B_setvalued", "KB_pmass_B"),
    ("kb_auc_C_setvalued", "KB_auc_C"),
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def fnum(v: Any) -> float | None:
    if v is None or v == "" or v == "nan":
        return None
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def fmt(v: Any, nd: int = 3) -> str:
    x = fnum(v)
    if x is None:
        return "NA"
    return f"{x:.{nd}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_md_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    lines = ["| " + " | ".join(fields) + " |",
             "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def build_seed_and_cell_tables(agg_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    seed_rows: list[dict[str, Any]] = []
    for r in agg_rows:
        seed_rows.append({
            "run_id": r["run_id"],
            "role": r["role"],
            "encoder": r["encoder"],
            "update": r["update"],
            "schedule": r["schedule"],
            "seed": r["seed"],
            "biored_macro_f1_ex_neg": r["biored_macro_f1_ex_neg"],
            "bc5cdr_macro_f1": r["bc5cdr_macro_f1"],
            "bc5cdr_drug_disease_f1": r["bc5cdr_drug_disease_f1"],
            "kb_hit_A_setvalued": r["kb_hit_A_setvalued"],
            "kb_pmass_B_setvalued": r["kb_pmass_B_setvalued"],
            "kb_auc_C_setvalued": r["kb_auc_C_setvalued"],
            "n_targets_evaluable": r["n_targets_evaluable"],
        })
    write_csv(DATA_DIR / "phase_b_ft_seedlevel.csv", seed_rows)

    metrics = [
        "biored_macro_f1_ex_neg",
        "bc5cdr_macro_f1",
        "bc5cdr_drug_disease_f1",
        "kb_hit_A_setvalued",
        "kb_pmass_B_setvalued",
        "kb_auc_C_setvalued",
    ]
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in agg_rows:
        groups[(r["encoder"], r["update"], r["schedule"])].append(r)
    cell_rows: list[dict[str, Any]] = []
    for (enc, upd, sched), rs in sorted(groups.items()):
        row: dict[str, Any] = {
            "encoder": enc, "update": upd, "schedule": sched,
            "n_seeds": len(rs),
        }
        for m in metrics:
            vals = [fnum(r[m]) for r in rs]
            vals = [v for v in vals if v is not None]
            if vals:
                row[f"{m}_mean"] = mean(vals)
                row[f"{m}_median"] = median(vals)
                row[f"{m}_sd"] = pstdev(vals) if len(vals) > 1 else 0.0
                row[f"{m}_min"] = min(vals)
                row[f"{m}_max"] = max(vals)
                row[f"{m}_ci95_halfwidth"] = 1.96 * (row[f"{m}_sd"] / math.sqrt(len(vals)))
            else:
                for suffix in ("mean", "median", "sd", "min", "max", "ci95_halfwidth"):
                    row[f"{m}_{suffix}"] = ""
        cell_rows.append(row)
    write_csv(DATA_DIR / "phase_b_ft_cells.csv", cell_rows)
    return cell_rows


def table02_cell_results(cell_rows: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for r in cell_rows:
        if r["update"] != "FT":
            continue
        rows.append({
            "cell": f"{r['encoder']}_{r['schedule']}",
            "n": r["n_seeds"],
            "BioRED_exNEG_mean": fmt(r["biored_macro_f1_ex_neg_mean"]),
            "BioRED_exNEG_CI95": f"±{fmt(r['biored_macro_f1_ex_neg_ci95_halfwidth'])}",
            "BC5CDR_macro_mean": fmt(r["bc5cdr_macro_f1_mean"]),
            "KB_hit_A_mean": fmt(r["kb_hit_A_setvalued_mean"]),
            "KB_pmass_B_mean": fmt(r["kb_pmass_B_setvalued_mean"]),
            "KB_auc_C_mean": fmt(r["kb_auc_C_setvalued_mean"]),
        })
    write_csv(TABLE_DIR / "table02_phase_b_cell_results.csv", rows)
    write_md_table(TABLE_DIR / "table02_phase_b_cell_results.md", rows)


def table03_hypotheses(analysis: dict[str, Any], h6: dict[str, Any], rq3: dict[str, Any]) -> None:
    h1 = analysis["H1_encoder"]
    h2 = analysis["H2_corpus"]
    h3 = analysis["H3_schedule"]
    h7 = analysis["H7_variance_asymmetry"]
    rb = analysis["h7_R_B_bootstrap"]
    ordn = analysis["rq4_ordinal_instability"]
    rq3_int = rq3["anova_partial_ss"]["terms"]["encoder_x_kb_metric"]
    rows = [
        {
            "item": "H1 encoder",
            "verdict": h1["verdict"],
            "key_result": "PL not superior; PL-PB=-0.0076, PL-BL=-0.0211",
            "paper_implication": "No global larger-biomedical-model advantage.",
        },
        {
            "item": "H2 corpus",
            "verdict": h2["verdict"],
            "key_result": f"Δ={fmt(h2['mean_diff'])}, CI=[{fmt(h2['ci95'][0])},{fmt(h2['ci95'][1])}], d={fmt(h2['cohens_d'])}",
            "paper_implication": "Strong support for multi-corpus T1 training for OOD BC5CDR.",
        },
        {
            "item": "H3 staged schedule",
            "verdict": h3["verdict"],
            "key_result": f"{h3['n_confirmed']}/6 tests confirmed",
            "paper_implication": "T2 staging helps PB and some BL/PL endpoints, but not uniformly.",
        },
        {
            "item": "H4 update regime",
            "verdict": analysis["H4_update_regime"]["verdict"],
            "key_result": "LoRA comparator collapsed in B.8/B.9/B.24",
            "paper_implication": "Report methodological null, not FT>LoRA confirmation.",
        },
        {
            "item": "H5 architecture",
            "verdict": analysis["H5_architecture_deferred"]["verdict"],
            "key_result": "shared_multitask deferred",
            "paper_implication": "Transparent gap, no result claim.",
        },
        {
            "item": "H7 R_B",
            "verdict": h7["verdict"],
            "key_result": f"R_B={fmt(h7['R_B'])}, CI=[{fmt(rb['ci_lower'])},{fmt(rb['ci_upper'])}]",
            "paper_implication": "Original variance-asymmetry direction not confirmed.",
        },
        {
            "item": "H6 β_config",
            "verdict": h6["beta_config"]["label"],
            "key_result": f"β={fmt(h6['beta_config']['estimate'])}, CI=[{fmt(h6['beta_config']['ci_lo'])},{fmt(h6['beta_config']['ci_hi'])}]",
            "paper_implication": "Use inconclusive abstract framing.",
        },
        {
            "item": "Ordinal instability",
            "verdict": "large_point_estimate_wide_ci",
            "key_result": f"median ΔKB={fmt(ordn['median_delta_KB'])}, inversion rate={fmt(ordn['rank_inversion_rate'])}",
            "paper_implication": "Important RQ4 signal, but report with wide-CI caution.",
        },
        {
            "item": "RQ3 encoder×metric",
            "verdict": "weak_interaction",
            "key_result": f"partial SS share={fmt(rq3_int['partial_ss_share_of_corrected_total'], 4)}",
            "paper_implication": "Audit formulation changes scale, not encoder ranking; schedule dominates.",
        },
    ]
    write_csv(TABLE_DIR / "table03_phase_b_hypothesis_summary.csv", rows)
    write_md_table(TABLE_DIR / "table03_phase_b_hypothesis_summary.md", rows)


def table04_rq_matrix() -> None:
    rows = [
        {
            "RQ": "RQ1 schema operationalisation",
            "completion": "95%",
            "evidence": "First-principle schema rationale; Phase A S_pair selection; active/dead head audit; leakage and oncology projection fixes.",
            "verdict": "Ready-to-write; strongest and cleanest foundation.",
            "limitation": "Per-family KB breakdown still rendered as figure/table support, not new experiment.",
        },
        {
            "RQ": "RQ2 OOD training configuration",
            "completion": "80%",
            "evidence": "H2 confirmed; H3 partial; H1 null; H4 methodological null; H5 deferred.",
            "verdict": "Mixed but interpretable: corpus diversity is the main robust lever.",
            "limitation": "No valid LoRA comparison and no shared-multitask architecture result.",
        },
        {
            "RQ": "RQ3 model family × audit formulation",
            "completion": "80%",
            "evidence": "Exploratory encoder × KB metric partial-SS audit; per-encoder KB profiles.",
            "verdict": "No strong interaction; schedule dominates KB surfacing.",
            "limitation": "Exploratory, not pre-registered confirmatory.",
        },
        {
            "RQ": "RQ4 benchmark × KB coupling",
            "completion": "95%",
            "evidence": "Phase A positive coupling; Phase B R_B null; H6 inconclusive; ordinal instability large.",
            "verdict": "Benchmark is not a stable proxy for KB surfacing; near-ties often reverse KB order.",
            "limitation": "Headline must be revised away from original H7 confirmation framing.",
        },
    ]
    write_csv(TABLE_DIR / "table04_rq_evidence_matrix.csv", rows)
    write_md_table(TABLE_DIR / "table04_rq_evidence_matrix.md", rows)


def fig02_phase_b_benchmark(cell_rows: list[dict[str, Any]]) -> None:
    main = [r for r in cell_rows if r["encoder"] in ENCODERS and r["update"] == "FT"]
    by = {(r["encoder"], r["schedule"]): r for r in main}
    colors = {"T1B": "#a6cee3", "T1F": "#1f78b4", "T2": "#08306b"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    specs = [
        ("biored_macro_f1_ex_neg", "BioRED macro-F1 (ex-NEG)", "Benchmark"),
        ("bc5cdr_macro_f1", "BC5CDR macro-F1", "OOD benchmark"),
    ]
    x = np.arange(len(ENCODERS))
    width = 0.24
    for ax, (metric, title, ylabel) in zip(axes, specs):
        for j, sched in enumerate(SCHEDULES):
            vals, errs = [], []
            for enc in ENCODERS:
                r = by[(enc, sched)]
                vals.append(float(r[f"{metric}_mean"]))
                errs.append(float(r[f"{metric}_ci95_halfwidth"]))
            ax.bar(x + (j - 1) * width, vals, width, yerr=errs,
                   label=sched, color=colors[sched], capsize=3,
                   edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(ENCODERS)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Phase B FT benchmark results (realised 9 cells × 20 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_phase_b_benchmark.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig03_kb_profiles(rq3: dict[str, Any], cell_rows: list[dict[str, Any]]) -> None:
    profiles = rq3["encoder_metric_profiles"]
    terms = rq3["anova_partial_ss"]["terms"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(ENCODERS))
    width = 0.24
    colors = {"KB_hit_A": "#2ca02c", "KB_pmass_B": "#ff7f0e", "KB_auc_C": "#9467bd"}
    for j, (_col, metric) in enumerate(KB_METRICS):
        vals = [profiles[enc]["metrics"][metric]["mean"] for enc in ENCODERS]
        errs = [
            profiles[enc]["metrics"][metric]["ci95_normal"][1]
            - profiles[enc]["metrics"][metric]["mean"]
            for enc in ENCODERS
        ]
        axes[0].bar(x + (j - 1) * width, vals, width, yerr=errs,
                    color=colors[metric], label=metric, capsize=3,
                    edgecolor="white", linewidth=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ENCODERS)
    axes[0].set_ylabel("KB metric value")
    axes[0].set_title("Encoder × KB metric profiles")
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].legend(frameon=False, fontsize=8)

    term_order = ["encoder", "schedule_block", "kb_metric", "encoder_x_kb_metric"]
    vals = [100 * terms[t]["partial_ss_share_of_corrected_total"] for t in term_order]
    labels = ["Encoder", "Schedule", "KB metric", "Encoder×metric"]
    axes[1].barh(labels, vals, color=["#1f77b4", "#08306b", "#ff7f0e", "#d62728"])
    for i, v in enumerate(vals):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    axes[1].set_xlabel("Partial SS share (% corrected total)")
    axes[1].set_title("RQ3 exploratory audit: schedule dominates")
    axes[1].grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_kb_surfacing_profiles.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig04_h7_ordinal(analysis: dict[str, Any]) -> None:
    h7 = analysis["H7_variance_asymmetry"]
    ordn = analysis["rq4_ordinal_instability"]
    decomp = h7["decomposition"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    labels = ["Encoder", "Schedule", "Enc×Sched"]
    keys = ["encoder", "schedule", "encoder_x_schedule"]
    x = np.arange(2)
    bottom = np.zeros(2)
    colors = ["#1f77b4", "#08306b", "#ff7f0e"]
    for lab, key, color in zip(labels, keys, colors):
        vals = [
            100 * decomp["biored_macro_f1_ex_neg"][key],
            100 * decomp["kb_hit_A_setvalued"][key],
        ]
        axes[0].bar(x, vals, bottom=bottom, label=lab, color=color,
                    edgecolor="white", linewidth=0.6)
        bottom += vals
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["BioRED ex-NEG", "KB_hit_A"])
    axes[0].set_ylabel("Design-lever variance share (%)")
    axes[0].set_title(f"H7 not confirmed: R_B = {h7['R_B']:.2f}")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)

    deltas = ordn["delta_KB_distribution"]
    axes[1].hist(deltas, bins=10, color="#2ca02c", alpha=0.8, edgecolor="white")
    axes[1].axvline(ordn["median_delta_KB"], color="black", linestyle="--",
                    label=f"median ΔKB={ordn['median_delta_KB']:.2f}")
    axes[1].set_xlabel("ΔKB among benchmark near-ties")
    axes[1].set_ylabel("Pair count")
    axes[1].set_title(f"Ordinal instability: inversion rate = {ordn['rank_inversion_rate']:.2f}")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_h7_variance_ordinal.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig05_h6(h6: dict[str, Any]) -> None:
    keys = ["beta_within", "beta_schema", "beta_encoder", "beta_config"]
    labels = ["β_within", "β_schema", "β_encoder", "β_config"]
    y = np.arange(len(keys))
    est = np.array([h6[k]["estimate"] for k in keys])
    lo = np.array([h6[k]["ci_lo"] for k in keys])
    hi = np.array([h6[k]["ci_hi"] for k in keys])
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="o", color="#1f77b4",
                ecolor="#1f77b4", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvspan(-0.3, 0.3, color="#f0f0f0", alpha=0.8, label="weak band")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Slope: ΔKB_hit_A per ΔBioRED ex-NEG")
    ax.set_title("H6 mechanism-stratified coupling slopes (all labelled inconclusive)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig05_h6_slopes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_validation_history(run_dir: Path) -> list[tuple[int, float]]:
    path = run_dir / "metrics" / "validation_history.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("entries", [])
    out: list[tuple[int, float]] = []
    for e in data:
        f1 = e.get("dev_macro_f1") or e.get("macro_f1")
        if f1 is not None:
            out.append((int(e["step"]), float(f1)))
    return out


def fig06_lora_audit() -> None:
    traces = {
        "FT matched cell (escapes)": (
            RUN_ROOT / "PB_PB_FT_T1B_s01", "#1f77b4", "o"),
        "LoRA LR=2e-5 / 2048 (B.8)": (
            ARCHIVE_ROOT / "lr2e5_preamendment_20260424T160804Z/PB_PB_LR_T1B_s01", "#d62728", "x"),
        "LoRA LR=3e-4 / 2048 (B.9)": (
            ARCHIVE_ROOT / "lr3e4_postB8_20260427T101524Z/PB_PB_LR_T1B_s99", "#ff7f0e", "^"),
        "LoRA LR=2e-5 / 4096 (B.24)": (
            ARCHIVE_ROOT / "d3_budget_probe_20260430T134418Z/PB_PB_LR_T1B_s99", "#9467bd", "s"),
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for label, (path, color, marker) in traces.items():
        hist = _load_validation_history(path)
        if not hist:
            continue
        steps = [s for s, _ in hist]
        f1s = [f for _, f in hist]
        ax.plot(steps, f1s, label=label, color=color, marker=marker,
                markersize=3.5, linewidth=1.4)
    ax.axhline(0.13, color="black", linestyle=":", linewidth=0.9)
    ax.text(120, 0.145, "all-NEGATIVE floor", fontsize=8)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Dev macro-F1")
    ax.set_title("H4 methodological null: canonical LoRA never escapes the all-NEGATIVE basin")
    ax.set_ylim(0.08, 0.56)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_lora_collapse_audit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    agg_rows = load_csv(AN_OUT / "phase_b_eval_aggregate_LATEST.csv")
    analysis = load_json(AN_OUT / "phase_b_analysis_LATEST.json")
    h6 = load_json(AN_OUT / "h6_coupling_slopes_LATEST.json")
    rq3 = load_json(AN_OUT / "rq3_encoder_kb_interaction_LATEST.json")

    cell_rows = build_seed_and_cell_tables(agg_rows)
    table02_cell_results(cell_rows)
    table03_hypotheses(analysis, h6, rq3)
    table04_rq_matrix()
    fig02_phase_b_benchmark(cell_rows)
    fig03_kb_profiles(rq3, cell_rows)
    fig04_h7_ordinal(analysis)
    fig05_h6(h6)
    fig06_lora_audit()

    print("Generated final report assets:")
    for p in [
        DATA_DIR / "phase_b_ft_seedlevel.csv",
        DATA_DIR / "phase_b_ft_cells.csv",
        TABLE_DIR / "table02_phase_b_cell_results.csv",
        TABLE_DIR / "table03_phase_b_hypothesis_summary.csv",
        TABLE_DIR / "table04_rq_evidence_matrix.csv",
        FIG_DIR / "fig02_phase_b_benchmark.png",
        FIG_DIR / "fig03_kb_surfacing_profiles.png",
        FIG_DIR / "fig04_h7_variance_ordinal.png",
        FIG_DIR / "fig05_h6_slopes.png",
        FIG_DIR / "fig06_lora_collapse_audit.png",
    ]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
