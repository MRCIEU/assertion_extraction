#!/usr/bin/env python3.11
"""
Phase A RQ4 preview — recompute seed-level Spearman rho between BioRED benchmark
F1 and the corrected correctness-aware KB metric (Method A, set_valued).

The original Phase A RQ4 finding (rho ≈ 0) was computed using the legacy
KB_surface_mean metric which §10.10.2 L1 showed is correctness-blind. This
script replaces KB_surface_mean with Method A (set_valued and single_label)
and recomputes the correlation at seed level (n=120).

Outputs to fine_tuning_experiments/schema_exp/phase_a_rq4_preview.{json,md}
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = SCRIPT_DIR.parent  # fine_tuning_experiments/schema_exp/


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(a: list[float]) -> list[float]:
        order = sorted(range(len(a)), key=lambda i: a[i])
        r = [0.0] * len(a)
        # Average ranks for ties
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # 1-indexed
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    n = len(xs)
    if n < 3: return float("nan")
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(sum((r - mx)**2 for r in rx) * sum((r - my)**2 for r in ry))
    return num / den if den else 0.0


def bootstrap_rho_ci(xs: list[float], ys: list[float], n_boot: int = 5000,
                     seed: int = 20260420) -> tuple[float, float]:
    import random
    rng = random.Random(seed)
    n = len(xs)
    rhos = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        rxs = [xs[i] for i in idx]
        rys = [ys[i] for i in idx]
        rhos.append(spearman(rxs, rys))
    rhos.sort()
    return rhos[int(0.025 * n_boot)], rhos[int(0.975 * n_boot)]


def main() -> None:
    reanalysis = json.loads((OUT_DIR / "phase_a_reanalysis_precompute.json").read_text())
    per_run_a = {r["run_id"]: r for r in reanalysis["per_run"]}

    # Read benchmark numbers from the Phase A results CSV
    results_csv = OUT_DIR / "phase_a_results.csv"
    per_run_b = {r["run_id"]: r for r in csv.DictReader(open(results_csv))}

    # Join and build vectors
    joined = []
    for run_id, a in per_run_a.items():
        b = per_run_b.get(run_id)
        if b is None: continue
        joined.append({
            "run_id": run_id,
            "encoder": a["encoder"],
            "schema_id": a["schema_id"],
            "seed": a["seed"],
            "biored_macro_f1_exneg": float(b["biored_macro_f1_ex_neg"]),
            "bc5cdr_dd_f1": float(b["bc5cdr_drug_disease_f1"]),
            "kb_surface_legacy": float(b["kb_surface_mean"]),
            "method_a_sv": a["method_a_set_valued"],
            "method_a_sl": a["method_a_single_label"],
        })
    print(f"Joined {len(joined)} seed-level observations")

    out = {"n": len(joined), "correlations": {}}

    # All 120 joint
    pairs = [
        ("BioRED_ex_neg", "kb_surface_legacy", "BR_ex_neg_vs_KB_surface_legacy"),
        ("BioRED_ex_neg", "method_a_sv",       "BR_ex_neg_vs_MethodA_setvalued"),
        ("BioRED_ex_neg", "method_a_sl",       "BR_ex_neg_vs_MethodA_singlelabel"),
        ("BC5CDR_DD",     "kb_surface_legacy", "BC5CDR_DD_vs_KB_surface_legacy"),
        ("BC5CDR_DD",     "method_a_sv",       "BC5CDR_DD_vs_MethodA_setvalued"),
        ("BC5CDR_DD",     "method_a_sl",       "BC5CDR_DD_vs_MethodA_singlelabel"),
    ]

    # Vector extractors
    def vec(r, name):
        if name == "BioRED_ex_neg": return r["biored_macro_f1_exneg"]
        if name == "BC5CDR_DD": return r["bc5cdr_dd_f1"]
        if name == "kb_surface_legacy": return r["kb_surface_legacy"]
        if name == "method_a_sv": return r["method_a_sv"]
        if name == "method_a_sl": return r["method_a_sl"]

    # Pool = "all 120", "biomedical only (PB+BL+PL, n=90)", per schema
    splits = {
        "all_n120": joined,
        "biomedical_n90": [r for r in joined if r["encoder"] in ("PB","BL","PL")],
        "S_flat_n40": [r for r in joined if r["schema_id"] == "S_flat"],
        "S_pair_n40": [r for r in joined if r["schema_id"] == "S_pair"],
        "S_mech_n40": [r for r in joined if r["schema_id"] == "S_mech"],
    }
    for split_name, data in splits.items():
        block = {}
        for x_name, y_name, pair_key in pairs:
            xs = [vec(r, x_name) for r in data]
            ys = [vec(r, y_name) for r in data]
            rho = spearman(xs, ys)
            lo, hi = bootstrap_rho_ci(xs, ys, n_boot=2000)
            block[pair_key] = {
                "rho": round(rho, 4),
                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "n": len(xs),
            }
        out["correlations"][split_name] = block

    (OUT_DIR / "phase_a_rq4_preview.json").write_text(json.dumps(out, indent=2))
    print("Wrote phase_a_rq4_preview.json")

    # Markdown
    lines = ["# Phase A — RQ4 preview (corrected KB metric)", ""]
    lines.append(f"**n_total_runs:** {len(joined)}")
    lines.append("")
    for split_name, data in splits.items():
        lines.append(f"## Split: `{split_name}` (n={len(data)})")
        lines.append("")
        lines.append("| x | y | ρ | 95% CI |")
        lines.append("|---|---|---|---|")
        for x_name, y_name, pair_key in pairs:
            b = out["correlations"][split_name][pair_key]
            lines.append(f"| {x_name} | {y_name} | **{b['rho']:+.3f}** | [{b['ci_lo']:+.3f}, {b['ci_hi']:+.3f}] |")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The legacy KB_surface_mean produced near-zero correlation with benchmark F1 at the full 120-run pool (the original RQ4 decoupling signal). Method A (set_valued, biomedical-only) tests whether this remains true under a correctness-aware metric on Phase B's encoder shortlist.")
    lines.append("")
    lines.append("Note: the 5 per-split rows are not independent RQ4 tests — they are a landscape to see whether the decoupling is stable across (a) whether RB is included, (b) whether schema-level structure dominates the x-axis. The joint Phase A + Phase B mixed-effects model (§11.10 of the paper design) is the primary H6 statistic; this preview informs whether Phase B's n=720 is likely to change the picture.")

    (OUT_DIR / "phase_a_rq4_preview.md").write_text("\n".join(lines) + "\n")
    print("Wrote phase_a_rq4_preview.md")


if __name__ == "__main__":
    main()
