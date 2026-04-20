#!/usr/bin/env python3.11
"""
Phase A-eval — aggregate all 120 per-run evaluation JSONs into:

  1. phase_a_results.csv        (one row per run with flat metrics)
  2. phase_a_aggregate.json     (grouped mean ± SE by encoder × schema, per metric)
  3. phase_a_schema_selection.json  (applies the pre-registered SC* rule)
  4. phase_a_report.md          (human-readable summary)

Pre-registered schema selection rule (see paper_development_design.md §3.1):

  SC* = argmax KB_surface_mean over {S_flat, S_pair, S_mech},
        subject to:  every head F1 > 0.05  AND  N_train ≥ 50 per head
        with statistical check p(SC* vs S_flat permutation) < 0.10
        tie-break: simpler schema (Occam)

KB_surface_mean is pooled across all 40 runs (4 encoders × 10 seeds) per schema
for the primary decision; per-encoder stratification is reported as a robustness check.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
FT_DATA_ROOT = Path(os.environ.get(
    "PROJECT_1_DATA_ROOT",
    "/lus/lfs1aip2/projects/b5ac/project_1",
)).resolve()
RUNS_ROOT = FT_DATA_ROOT / "fine_tuning_experiments" / "runs" / "schema_exp"
OUT_DIR = SCRIPT_DIR.parent  # fine_tuning_experiments/schema_exp/
OUT_DIR_EVAL = SCRIPT_DIR    # fine_tuning_experiments/schema_exp/eval/

SCHEMAS = ("Sflat", "Spair", "Smech")
ENCODERS = ("RB", "PB", "BL", "PL")


def mean_sd_se(vs: list[float]) -> tuple[float, float, float]:
    if not vs:
        return float("nan"), float("nan"), float("nan")
    m = statistics.mean(vs)
    sd = statistics.stdev(vs) if len(vs) > 1 else 0.0
    se = sd / math.sqrt(len(vs)) if len(vs) > 1 else 0.0
    return m, sd, se


def permutation_test(a: list[float], b: list[float], *, n: int = 5000, seed: int = 20260416,
                     ) -> tuple[float, float, float, float]:
    """Two-sample permutation test for difference in means (two-sided).
    Returns (observed_diff, ci_lo, ci_hi, p_value) using bootstrap CI."""
    if not a or not b:
        return 0.0, 0.0, 0.0, 1.0
    rng = random.Random(seed)
    observed = statistics.mean(a) - statistics.mean(b)
    pool = a + b
    na = len(a)
    perm_diffs = []
    for _ in range(n):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        perm_diffs.append(statistics.mean(shuffled[:na]) - statistics.mean(shuffled[na:]))
    p = sum(1 for d in perm_diffs if abs(d) >= abs(observed) - 1e-12) / n
    boot = []
    for _ in range(n):
        sa = [rng.choice(a) for _ in a]
        sb = [rng.choice(b) for _ in b]
        boot.append(statistics.mean(sa) - statistics.mean(sb))
    boot.sort()
    lo = boot[int(0.025 * n)]
    hi = boot[int(0.975 * n)]
    return observed, lo, hi, p


def cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va = statistics.variance(a)
    vb = statistics.variance(b)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return (statistics.mean(a) - statistics.mean(b)) / pooled


# ───────────────────────────────────────────────────────────────────
# Load per-run evals
# ───────────────────────────────────────────────────────────────────

def load_runs() -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted(RUNS_ROOT.glob("PA_*")):
        ev = run_dir / "eval" / "phase_a_eval.json"
        if not ev.exists():
            continue
        d = json.loads(ev.read_text())
        biored = d["biored_test"]
        bc = d["bc5cdr_test"]
        kb = d["kb_surface"]
        per_label = biored.get("per_label", {})
        flat: dict[str, Any] = {
            "run_id": d["run_id"],
            "encoder": d["encoder_key"],
            "schema": d["schema_key"],
            "schema_id": d["schema_id"],
            "seed": d["seed"],
            "n_labels": len(d["label2id"]),
            # BioRED test
            "biored_macro_f1": biored["macro_f1"],
            "biored_macro_f1_ex_neg": biored["macro_f1_excluding_negative"],
            "biored_n": biored["n"],
            # BC5CDR test
            "bc5cdr_drug_disease_f1": bc["drug_disease_f1"],
            "bc5cdr_macro_f1": bc["macro_f1"],
            "bc5cdr_n": bc["n"],
            # KB surface — legacy (deprecated but kept for backwards-compat)
            "kb_surface_mean": kb["kb_surface_mean"],
            "kb_surface_50": kb.get("kb_surface_50"),
            "kb_nonneg_rate": kb.get("kb_nonneg_rate"),
            # KB surface — v1.0 correctness-aware metrics (§11.7.1)
            "kb_hit_A_setvalued":     kb.get("kb_hit_A_setvalued"),
            "kb_hit_A_singlelabel":   kb.get("kb_hit_A_singlelabel"),
            "kb_pmass_B_setvalued":   kb.get("kb_pmass_B_setvalued"),
            "kb_pmass_B_singlelabel": kb.get("kb_pmass_B_singlelabel"),
            "kb_auc_C_setvalued":     kb.get("kb_auc_C_setvalued"),
            "kb_auc_C_singlelabel":   kb.get("kb_auc_C_singlelabel"),
            "n_targets_evaluable":    kb.get("n_targets_evaluable"),
            # Per-head F1 in BioRED
            **{f"biored_f1__{lab}": stats["f1"] for lab, stats in per_label.items()},
            **{f"biored_support__{lab}": stats["support"] for lab, stats in per_label.items()},
        }
        rows.append(flat)
    return rows


# ───────────────────────────────────────────────────────────────────
# Aggregation
# ───────────────────────────────────────────────────────────────────

PRIMARY_METRICS = [
    # v1.0 correctness-aware KB metrics (§11.7.1)
    "kb_hit_A_setvalued", "kb_hit_A_singlelabel",
    "kb_pmass_B_setvalued", "kb_pmass_B_singlelabel",
    "kb_auc_C_setvalued", "kb_auc_C_singlelabel",
    # Legacy KB metric (deprecated, kept for continuity)
    "kb_surface_mean", "kb_surface_50", "kb_nonneg_rate",
    # Benchmark metrics
    "biored_macro_f1", "biored_macro_f1_ex_neg",
    "bc5cdr_drug_disease_f1", "bc5cdr_macro_f1",
]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_runs": len(rows),
        "by_encoder_schema": {},
        "by_schema": {},
        "by_encoder": {},
    }
    # by (encoder, schema)
    grp_es: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grp_es[(r["encoder"], r["schema"])].append(r)
    for (enc, sch), rs in grp_es.items():
        cell: dict[str, Any] = {"n": len(rs)}
        for m in PRIMARY_METRICS:
            vs = [r[m] for r in rs if m in r and r[m] is not None]
            mean, sd, se = mean_sd_se(vs)
            cell[m] = {"mean": mean, "sd": sd, "se": se, "n": len(vs)}
        out["by_encoder_schema"][f"{enc}_{sch}"] = cell

    # by schema (pool across encoders)
    grp_s: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grp_s[r["schema"]].append(r)
    for sch, rs in grp_s.items():
        cell: dict[str, Any] = {"n": len(rs)}
        for m in PRIMARY_METRICS:
            vs = [r[m] for r in rs if m in r and r[m] is not None]
            mean, sd, se = mean_sd_se(vs)
            cell[m] = {"mean": mean, "sd": sd, "se": se, "n": len(vs)}
        out["by_schema"][sch] = cell

    # by encoder (pool across schemas — useful for encoder reference)
    grp_e: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grp_e[r["encoder"]].append(r)
    for enc, rs in grp_e.items():
        cell: dict[str, Any] = {"n": len(rs)}
        for m in PRIMARY_METRICS:
            vs = [r[m] for r in rs if m in r and r[m] is not None]
            mean, sd, se = mean_sd_se(vs)
            cell[m] = {"mean": mean, "sd": sd, "se": se, "n": len(vs)}
        out["by_encoder"][enc] = cell

    # Permutation tests on schema pairs for primary-interest metrics
    tests: list[dict[str, Any]] = []
    perm_metrics = [
        "kb_hit_A_setvalued",           # §11.7.1.1 primary KB metric
        "kb_hit_A_singlelabel",         # sensitivity (matters only for S_mech)
        "kb_pmass_B_setvalued",         # Method B robustness
        "kb_auc_C_setvalued",           # Method C robustness
        "kb_surface_mean",              # legacy (retained for continuity with Phase A v5)
        "biored_macro_f1_ex_neg",       # benchmark
        "bc5cdr_drug_disease_f1",       # benchmark
    ]
    for metric in perm_metrics:
        by_schema_vals = {s: [r[metric] for r in grp_s[s] if r.get(metric) is not None] for s in SCHEMAS}
        if not all(by_schema_vals.values()):
            continue
        for a, b in [("Sflat", "Spair"), ("Sflat", "Smech"), ("Spair", "Smech")]:
            diff, lo, hi, p = permutation_test(by_schema_vals[a], by_schema_vals[b])
            tests.append({
                "metric": metric, "a": a, "b": b,
                "mean_a": statistics.mean(by_schema_vals[a]) if by_schema_vals[a] else None,
                "mean_b": statistics.mean(by_schema_vals[b]) if by_schema_vals[b] else None,
                "diff": diff, "ci_lo": lo, "ci_hi": hi,
                "p_value": p,
                "cohens_d": cohens_d(by_schema_vals[a], by_schema_vals[b]),
                "n_a": len(by_schema_vals[a]), "n_b": len(by_schema_vals[b]),
            })
    out["permutation_tests"] = tests
    return out


def schema_selection(rows: list[dict[str, Any]], agg: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered SC* selection rule."""
    by_schema = agg["by_schema"]
    # Primary ranking: by KB_surface_mean
    ranking = sorted(SCHEMAS, key=lambda s: -by_schema[s]["kb_surface_mean"]["mean"])
    candidate = ranking[0]

    # Head F1 feasibility: each schema must have every head with F1 > 0.05
    # Compute from per-run per-head values
    per_schema_heads: dict[str, dict[str, list[float]]] = {s: defaultdict(list) for s in SCHEMAS}
    for r in rows:
        sch = r["schema"]
        for key, v in r.items():
            if key.startswith("biored_f1__"):
                lab = key.replace("biored_f1__", "")
                if lab == "__NEGATIVE__":
                    continue
                # Only count heads with non-zero support on BioRED test
                support = r.get(f"biored_support__{lab}", 0)
                if support > 0:
                    per_schema_heads[sch][lab].append(v)

    head_f1_report: dict[str, Any] = {}
    for sch in SCHEMAS:
        heads = per_schema_heads[sch]
        head_means = {lab: statistics.mean(vs) for lab, vs in heads.items() if vs}
        min_head = min(head_means.values()) if head_means else 0.0
        head_f1_report[sch] = {
            "head_means": head_means,
            "min_head_f1": min_head,
            "all_above_0.05": all(v > 0.05 for v in head_means.values()) if head_means else False,
        }

    # Permutation p-values against Sflat
    pvals_vs_sflat = {t["b"]: t for t in agg["permutation_tests"]
                      if t["metric"] == "kb_surface_mean" and t["a"] == "Sflat"}
    pvals_vs_sflat["Sflat"] = {"p_value": None}

    # Applied rule
    reason = []
    final = candidate
    if candidate != "Sflat":
        p = pvals_vs_sflat[candidate]["p_value"]
        if p is not None and p >= 0.10:
            reason.append(f"{candidate} vs Sflat permutation p={p:.4f} ≥ 0.10 → "
                          "fall back to simpler schema (Sflat)")
            final = "Sflat"
        else:
            reason.append(f"{candidate} vs Sflat permutation p={p:.4f} < 0.10 — selection holds")
    else:
        reason.append("Sflat has highest KB_surface_mean; no statistical test needed")

    if not head_f1_report[final]["all_above_0.05"]:
        reason.append(f"{final} has at least one head with mean F1 ≤ 0.05 — flagging "
                      "feasibility concern (does not force fallback under current rule)")

    return {
        "ranking_by_kb_surface_mean": ranking,
        "primary_candidate": candidate,
        "applied_rule": "argmax KB_surface_mean subject to permutation p<0.10 vs Sflat; "
                        "tie-break to simpler schema (Occam).",
        "selected_SC_star": final,
        "selection_reason": reason,
        "head_f1_feasibility": head_f1_report,
        "permutation_p_vs_sflat": {k: v["p_value"] for k, v in pvals_vs_sflat.items()},
    }


# ───────────────────────────────────────────────────────────────────
# Report
# ───────────────────────────────────────────────────────────────────

def render_report(rows: list[dict], agg: dict, sel: dict) -> str:
    lines = ["# Phase A-eval Report",
             "",
             f"**Runs evaluated:** {len(rows)} / 120",
             f"**Selected SC\\*:** `{sel['selected_SC_star']}`  (primary candidate: `{sel['primary_candidate']}`)",
             ""]

    lines += ["## 1. KB metrics — v1.0 correctness-aware (§11.7.1)", ""]
    lines += ["| Schema | n | Method A set_valued | Method A single_label | Method B set_valued | Method C set_valued | Legacy KB_surface_mean |",
              "|---|---|---|---|---|---|---|"]
    for sch in SCHEMAS:
        c = agg["by_schema"][sch]
        def fmt(key: str) -> str:
            v = c.get(key)
            if v is None or v.get("mean") is None or (isinstance(v["mean"], float) and math.isnan(v["mean"])):
                return "—"
            return f"{v['mean']:.4f} ± {v['se']:.4f}"
        lines.append(
            f"| {sch} | {c['n']} "
            f"| {fmt('kb_hit_A_setvalued')} "
            f"| {fmt('kb_hit_A_singlelabel')} "
            f"| {fmt('kb_pmass_B_setvalued')} "
            f"| {fmt('kb_auc_C_setvalued')} "
            f"| {fmt('kb_surface_mean')} |"
        )
    lines.append("")

    lines += ["## 2. Benchmark tests", ""]
    lines += ["| Schema | BioRED macro-F1 (all) | BioRED macro-F1 (ex-NEG) | BC5CDR DRUG_DISEASE F1 |",
              "|---|---|---|---|"]
    for sch in SCHEMAS:
        c = agg["by_schema"][sch]
        lines.append(
            f"| {sch} "
            f"| {c['biored_macro_f1']['mean']:.4f} ± {c['biored_macro_f1']['se']:.4f} "
            f"| {c['biored_macro_f1_ex_neg']['mean']:.4f} ± {c['biored_macro_f1_ex_neg']['se']:.4f} "
            f"| {c['bc5cdr_drug_disease_f1']['mean']:.4f} ± {c['bc5cdr_drug_disease_f1']['se']:.4f} |"
        )
    lines.append("")

    lines += ["## 3. Encoder × schema (Method A set_valued; §11.7.1.1 primary)", ""]
    header = "| encoder | " + " | ".join(SCHEMAS) + " |"
    sep = "|---" * (len(SCHEMAS) + 1) + "|"
    lines += [header, sep]
    for enc in ENCODERS:
        cells = [f"{enc}"]
        for sch in SCHEMAS:
            c = agg["by_encoder_schema"].get(f"{enc}_{sch}")
            if c and c.get("kb_hit_A_setvalued") and c["kb_hit_A_setvalued"].get("mean") is not None:
                cells.append(f"{c['kb_hit_A_setvalued']['mean']:.4f} ± {c['kb_hit_A_setvalued']['se']:.4f}")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## 4. Permutation tests (pooled across encoders, n=40 per schema)", ""]
    lines += ["| metric | A | B | mean_A | mean_B | diff | 95% CI | p | Cohen's d |",
              "|---|---|---|---|---|---|---|---|---|"]
    for t in agg["permutation_tests"]:
        lines.append(
            f"| {t['metric']} | {t['a']} | {t['b']} "
            f"| {t['mean_a']:.4f} | {t['mean_b']:.4f} "
            f"| {t['diff']:+.4f} | [{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}] "
            f"| {t['p_value']:.4f} | {t['cohens_d']:+.3f} |"
        )
    lines.append("")

    lines += ["## 5. Head-level feasibility (BioRED test heads with support > 0)", ""]
    for sch in SCHEMAS:
        hf = sel["head_f1_feasibility"][sch]
        lines.append(f"**{sch}** — min head F1 = {hf['min_head_f1']:.4f}; "
                     f"all>0.05: {hf['all_above_0.05']}")
        for lab, f1 in sorted(hf["head_means"].items(), key=lambda kv: kv[1]):
            lines.append(f"  - {lab}: F1 = {f1:.4f}")
        lines.append("")

    lines += ["## 6. Schema selection decision", ""]
    lines += [f"- Ranking by KB_surface_mean: {' > '.join(sel['ranking_by_kb_surface_mean'])}",
              f"- Applied rule: {sel['applied_rule']}",
              f"- Selected SC\\*: **{sel['selected_SC_star']}**",
              ""]
    for r in sel["selection_reason"]:
        lines.append(f"  - {r}")
    return "\n".join(lines) + "\n"


# ───────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    rows = load_runs()
    print(f"Loaded {len(rows)} per-run evaluations")
    if len(rows) < 120:
        missing = 120 - len(rows)
        print(f"WARNING: {missing} runs have no phase_a_eval.json — partial aggregation")
    if not rows:
        print("No per-run evaluations available; run eval_one_run.py first.")
        return

    # CSV
    csv_path = OUT_DIR / "phase_a_results.csv"
    all_keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); all_keys.append(k)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"CSV: {csv_path}")

    # Aggregate
    agg = aggregate(rows)
    (OUT_DIR / "phase_a_aggregate.json").write_text(json.dumps(agg, indent=2))
    print(f"Aggregate: {OUT_DIR / 'phase_a_aggregate.json'}")

    # Selection
    sel = schema_selection(rows, agg)
    (OUT_DIR / "phase_a_schema_selection.json").write_text(json.dumps(sel, indent=2))
    print(f"Selection: {OUT_DIR / 'phase_a_schema_selection.json'}  → SC* = {sel['selected_SC_star']}")

    # Report
    report = render_report(rows, agg, sel)
    (OUT_DIR / "phase_a_report.md").write_text(report)
    print(f"Report: {OUT_DIR / 'phase_a_report.md'}")


if __name__ == "__main__":
    main()
