#!/usr/bin/env python3.11
"""Phase A — comprehensive scientific analysis of the 120-run factorial.

Reads:
  fine_tuning_experiments/schema_exp/phase_a_results.csv

Produces:
  fine_tuning_experiments/schema_exp/analysis/phase_a_analysis.json
  fine_tuning_experiments/schema_exp/analysis/phase_a_analysis.md

Analyses (paper_development_design.md §6.6 and Part 8):
  1. Apply the §6.6 SC* selection rule on `kb_hit_A_setvalued` (not the
     deprecated `kb_surface_mean`).
  2. Per-cell (encoder × schema) means with 95% bootstrap CIs for every
     primary metric.
  3. H7 variance decomposition (fraction of total variance attributable to
     schema / encoder / seed) for BioRED ex-NEG, BC5CDR DD, KB_hit_A.
  4. Cross-metric correlation preview (Phase A only, pooled):
     BioRED ex-NEG × KB_hit_A, BC5CDR DD × KB_hit_A.
  5. Per-head F1 by schema, tagged with support and feasibility against the
     0.05 floor.
  6. Permutation-test summary with BH-FDR across the 3 schema × 3 metric
     primary family.
  7. Encoder-stratified schema comparison (robustness check).
  8. Intraclass correlation (ICC) of BioRED ex-NEG and KB_hit_A (cell-level).
"""
from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve()
PHASE_A_DIR = SCRIPT.parent.parent        # .../schema_exp
ANALYSIS_DIR = SCRIPT.parent              # .../schema_exp/analysis
CSV_PATH = PHASE_A_DIR / "phase_a_results.csv"

SCHEMAS = ["Sflat", "Spair", "Smech"]
ENCODERS = ["RB", "PB", "BL", "PL"]

PRIMARY_KB = "kb_hit_A_setvalued"
BENCH_BIORED = "biored_macro_f1_ex_neg"
BENCH_BC5CDR = "bc5cdr_drug_disease_f1"
LEGACY_KB = "kb_surface_mean"

# -----------------------------------------------------------------------------
# Statistics helpers
# -----------------------------------------------------------------------------

def bootstrap_ci(values, *, n_boot=5000, seed=20260416, alpha=0.05):
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        samp = [rng.choice(values) for _ in values]
        boots.append(statistics.mean(samp))
    boots.sort()
    lo = boots[int(alpha/2 * n_boot)]
    hi = boots[int((1 - alpha/2) * n_boot)]
    return statistics.mean(values), lo, hi


def perm_test(a, b, *, n=5000, seed=20260416):
    rng = random.Random(seed)
    observed = statistics.mean(a) - statistics.mean(b)
    pool = a + b
    na = len(a)
    count = 0
    for _ in range(n):
        rng.shuffle(pool)
        d = statistics.mean(pool[:na]) - statistics.mean(pool[na:])
        if abs(d) >= abs(observed) - 1e-12:
            count += 1
    return observed, count / n


def boot_diff_ci(a, b, *, n=5000, seed=20260416, alpha=0.05):
    """Unpaired bootstrap difference-in-means CI.  Kept for places where
    the two samples are not paired (e.g. encoder-stratified comparisons that
    pool across different cell counts)."""
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        sa = [rng.choice(a) for _ in a]
        sb = [rng.choice(b) for _ in b]
        diffs.append(statistics.mean(sa) - statistics.mean(sb))
    diffs.sort()
    lo = diffs[int(alpha/2 * n)]
    hi = diffs[int((1 - alpha/2) * n)]
    return lo, hi


def boot_paired_diff_ci(pairs, *, n=10000, seed=20260416, alpha=0.05):
    """Paired bootstrap difference-in-means CI.

    pairs: list of (a_i, b_i) tuples, one per (encoder, seed) cell.
    Resamples the 40 cells jointly with replacement and averages the
    per-cell difference.  Narrower and better calibrated than the unpaired
    version when samples share structure (encoder identity, seed index)."""
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not pairs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n_cells = len(pairs)
    diffs = []
    for _ in range(n):
        idxs = [rng.randrange(n_cells) for _ in range(n_cells)]
        diffs.append(statistics.mean(pairs[i][0] - pairs[i][1] for i in idxs))
    diffs.sort()
    lo = diffs[int(alpha/2 * n)]
    hi = diffs[int((1 - alpha/2) * n)]
    return lo, hi


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2: return 0.0
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = math.sqrt(((len(a)-1)*va + (len(b)-1)*vb) / (len(a)+len(b)-2))
    if pooled == 0: return 0.0
    return (statistics.mean(a) - statistics.mean(b)) / pooled


def pearson(x, y):
    if len(x) != len(y) or len(x) < 3: return float("nan")
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = math.sqrt(sum((xi-mx)**2 for xi in x))
    sy = math.sqrt(sum((yi-my)**2 for yi in y))
    if sx == 0 or sy == 0: return float("nan")
    return sum((xi-mx)*(yi-my) for xi, yi in zip(x, y)) / (sx * sy)


def spearman(x, y):
    if len(x) != len(y) or len(x) < 3: return float("nan")
    def ranks(v):
        s = sorted(enumerate(v), key=lambda kv: kv[1])
        r = [0]*len(v)
        for pos, (idx, _) in enumerate(s):
            r[idx] = pos + 1
        return r
    return pearson(ranks(x), ranks(y))


def boot_corr_ci(x, y, *, fn=pearson, n=5000, seed=20260416, alpha=0.05):
    rng = random.Random(seed)
    idx = list(range(len(x)))
    rs = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        xs, ys = [x[i] for i in s], [y[i] for i in s]
        r = fn(xs, ys)
        if not math.isnan(r):
            rs.append(r)
    rs.sort()
    if not rs:
        return float("nan"), float("nan")
    lo = rs[int(alpha/2 * len(rs))]
    hi = rs[int((1-alpha/2) * len(rs))]
    return lo, hi


def benjamini_hochberg(pvals, *, alpha=0.05):
    """Return dict: index -> is_significant; plus adjusted p-values."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0]*m
    for rank, i in enumerate(order, 1):
        adj[i] = pvals[i] * m / rank
    # Enforce monotonicity (from bottom up)
    sorted_adj = sorted(range(m), key=lambda i: pvals[i], reverse=True)
    running = 1.0
    for i in sorted_adj:
        running = min(running, adj[i])
        adj[i] = running
    return [min(a, 1.0) for a in adj]


def iqr_stats(values):
    if not values: return {}
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "sd": statistics.stdev(s) if n > 1 else 0.0,
        "min": s[0], "max": s[-1],
        "q25": s[n//4], "q75": s[3*n//4],
    }


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

df = pd.read_csv(CSV_PATH)
assert len(df) == 120, f"Expected 120 rows, got {len(df)}"

analysis: dict = {"n_runs": len(df), "schemas": SCHEMAS, "encoders": ENCODERS}

# -----------------------------------------------------------------------------
# (1) Per-cell table: mean ± 95% bootstrap CI for every primary metric
# -----------------------------------------------------------------------------

METRICS = [PRIMARY_KB, "kb_hit_A_singlelabel", "kb_pmass_B_setvalued",
           "kb_auc_C_setvalued", LEGACY_KB,
           BENCH_BIORED, "biored_macro_f1", BENCH_BC5CDR, "bc5cdr_macro_f1"]

cells = {}
for enc in ENCODERS:
    for sch in SCHEMAS:
        sub = df[(df["encoder"] == enc) & (df["schema"] == sch)]
        assert len(sub) == 10, f"{enc}×{sch}: n={len(sub)}"
        cell = {"n": 10}
        for m in METRICS:
            vs = sub[m].dropna().tolist()
            if vs:
                mean, lo, hi = bootstrap_ci(vs)
                cell[m] = {"mean": mean, "ci_lo": lo, "ci_hi": hi,
                           "sd": statistics.stdev(vs) if len(vs)>1 else 0.0}
            else:
                cell[m] = None
        cells[f"{enc}_{sch}"] = cell
analysis["cells"] = cells

# Pooled by schema
by_schema = {}
for sch in SCHEMAS:
    sub = df[df["schema"] == sch]
    s = {"n": len(sub)}
    for m in METRICS:
        vs = sub[m].dropna().tolist()
        if vs:
            mean, lo, hi = bootstrap_ci(vs)
            s[m] = {"mean": mean, "ci_lo": lo, "ci_hi": hi,
                    "sd": statistics.stdev(vs) if len(vs) > 1 else 0.0}
    by_schema[sch] = s
analysis["by_schema"] = by_schema

# -----------------------------------------------------------------------------
# (2) §6.6 decision rule applied on kb_hit_A_setvalued
# -----------------------------------------------------------------------------

# Rank schemas by kb_hit_A_setvalued mean, pooled n=40
ranking = sorted(SCHEMAS, key=lambda s: -by_schema[s][PRIMARY_KB]["mean"])
top, second, third = ranking

# Pairwise permutation + *paired* bootstrap CI on kb_hit_A and BioRED ex-NEG.
# §6.6 pre-commits paired bootstrap at (encoder, seed) level, B=10000.
# Each of the 40 (encoder, seed) cells contributes one paired difference per
# schema-pair, controlling for between-encoder variance.
def _paired_list(df_sub, sa: str, sb: str, metric: str):
    """Build [(a_val, b_val), ...] across matched (encoder, seed) cells."""
    by_cell_a = {(r.encoder, r.seed): getattr(r, metric) for r in df_sub[df_sub.schema == sa].itertuples()}
    by_cell_b = {(r.encoder, r.seed): getattr(r, metric) for r in df_sub[df_sub.schema == sb].itertuples()}
    common = sorted(set(by_cell_a) & set(by_cell_b))
    return [(by_cell_a[c], by_cell_b[c]) for c in common]

pair_tests = {}
for a, b in [(top, second), (top, third), (second, third)]:
    va = df[df.schema == a][PRIMARY_KB].tolist()
    vb = df[df.schema == b][PRIMARY_KB].tolist()
    diff, p = perm_test(va, vb)
    lo, hi = boot_paired_diff_ci(_paired_list(df, a, b, PRIMARY_KB))
    d = cohens_d(va, vb)
    pair_tests[f"{a}_vs_{b}__KB_hit_A"] = {
        "diff": diff, "ci_lo": lo, "ci_hi": hi, "p_value": p, "cohens_d": d,
        "ci_excludes_zero": (lo > 0 or hi < 0),
        "pairing": "paired at (encoder, seed), B=10000",
    }
    ba = df[df.schema == a][BENCH_BIORED].tolist()
    bb = df[df.schema == b][BENCH_BIORED].tolist()
    diff_b, p_b = perm_test(ba, bb)
    lob, hib = boot_paired_diff_ci(_paired_list(df, a, b, BENCH_BIORED))
    db = cohens_d(ba, bb)
    pair_tests[f"{a}_vs_{b}__BIORED_ex_NEG"] = {
        "diff": diff_b, "ci_lo": lob, "ci_hi": hib, "p_value": p_b, "cohens_d": db,
        "ci_excludes_zero": (lob > 0 or hib < 0),
        "pairing": "paired at (encoder, seed), B=10000",
    }

# Apply §6.6 decision tree
top_vs_second_kb = pair_tests[f"{top}_vs_{second}__KB_hit_A"]
top_vs_second_bio = pair_tests[f"{top}_vs_{second}__BIORED_ex_NEG"]

decision = {"top": top, "second": second, "third": third,
            "ranking_by_KB_hit_A_setvalued": ranking}

reasoning = []
ci_excludes_zero = top_vs_second_kb["ci_excludes_zero"]
reasoning.append(f"KB_hit_A_setvalued: {top}={by_schema[top][PRIMARY_KB]['mean']:.4f} "
                 f"vs {second}={by_schema[second][PRIMARY_KB]['mean']:.4f}; "
                 f"diff CI=[{top_vs_second_kb['ci_lo']:.4f}, {top_vs_second_kb['ci_hi']:.4f}] "
                 f"({'excludes' if ci_excludes_zero else 'includes'} 0); "
                 f"Cohen's d={top_vs_second_kb['cohens_d']:.3f}")
reasoning.append(f"BioRED ex-NEG: {top}={by_schema[top][BENCH_BIORED]['mean']:.4f} "
                 f"vs {second}={by_schema[second][BENCH_BIORED]['mean']:.4f}; "
                 f"diff={top_vs_second_bio['diff']:+.4f}; "
                 f"Cohen's d={top_vs_second_bio['cohens_d']:+.3f}")

# Outcome decision
# Outcome 1: top strictly higher on KB (CI excludes 0) AND not worse on BioRED by d>0.3
top_worse_on_biored_by_03 = (top_vs_second_bio["cohens_d"] < -0.3)
# Outcome 2: top within d=0.3 of second on KB, OR direction disagreement
top_second_close_on_kb = (abs(top_vs_second_kb["cohens_d"]) < 0.3)
direction_disagreement = (
    (top_vs_second_kb["diff"] > 0 and top_vs_second_bio["diff"] < 0) or
    (top_vs_second_kb["diff"] < 0 and top_vs_second_bio["diff"] > 0)
)

if ci_excludes_zero and not top_worse_on_biored_by_03:
    outcome, phase_b_schemas = "Outcome 1 (single schema)", [top]
    reasoning.append(f"→ Outcome 1: {top} strictly higher on KB_hit_A (CI excludes zero), "
                     f"not worse on BioRED by Cohen's d > 0.3. Phase B uses {top} alone.")
elif top_second_close_on_kb or direction_disagreement:
    outcome, phase_b_schemas = "Outcome 2 (dual schema)", [top, second]
    reasoning.append(f"→ Outcome 2: {top}/{second} within d=0.3 on KB_hit_A "
                     f"OR benchmark/KB directions disagree. Phase B runs both.")
else:
    outcome, phase_b_schemas = "Outcome 3 (null)", ["Spair"]  # default fallback
    reasoning.append("→ Outcome 3: null result; Phase B retains S_pair alone.")

decision.update({
    "pair_tests": pair_tests,
    "outcome": outcome,
    "phase_b_schemas": phase_b_schemas,
    "reasoning": reasoning,
})
analysis["decision_666"] = decision

# -----------------------------------------------------------------------------
# (3) H7 variance decomposition — share of variance by schema / encoder / seed
# -----------------------------------------------------------------------------

def variance_share(df, metric):
    """Simple one-way variance decomposition: schema, encoder, residual (seed+noise)."""
    grand_mean = df[metric].mean()
    # Total SS
    ss_total = ((df[metric] - grand_mean)**2).sum()
    # Schema SS (between-schema, pooled across encoders/seeds)
    ss_schema = 0.0
    for sch in SCHEMAS:
        sub = df[df.schema == sch][metric]
        ss_schema += len(sub) * (sub.mean() - grand_mean)**2
    # Encoder SS (between-encoder, pooled across schemas/seeds)
    ss_encoder = 0.0
    for enc in ENCODERS:
        sub = df[df.encoder == enc][metric]
        ss_encoder += len(sub) * (sub.mean() - grand_mean)**2
    # Cell means (schema × encoder)
    ss_interaction = 0.0
    for sch in SCHEMAS:
        for enc in ENCODERS:
            sub = df[(df.schema == sch) & (df.encoder == enc)][metric]
            cell_mean = sub.mean()
            schema_mean = df[df.schema == sch][metric].mean()
            encoder_mean = df[df.encoder == enc][metric].mean()
            predicted = schema_mean + encoder_mean - grand_mean
            ss_interaction += len(sub) * (cell_mean - predicted)**2
    # Residual (within-cell, seed-driven)
    ss_within = 0.0
    for sch in SCHEMAS:
        for enc in ENCODERS:
            sub = df[(df.schema == sch) & (df.encoder == enc)][metric]
            ss_within += ((sub - sub.mean())**2).sum()
    total_check = ss_schema + ss_encoder + ss_interaction + ss_within
    return {
        "ss_total": float(ss_total),
        "share_schema":      float(ss_schema / ss_total),
        "share_encoder":     float(ss_encoder / ss_total),
        "share_interaction": float(ss_interaction / ss_total),
        "share_within_cell": float(ss_within / ss_total),
        "partition_sum":     float(total_check / ss_total),
    }

variance = {}
for m in [PRIMARY_KB, BENCH_BIORED, BENCH_BC5CDR, LEGACY_KB, "biored_macro_f1"]:
    variance[m] = variance_share(df, m)
analysis["H7_variance_decomposition"] = variance

# Variance asymmetry ratio: schema's share for KB / schema's share for benchmark
asymmetry = {
    "schema_share_KB_hit_A":    variance[PRIMARY_KB]["share_schema"],
    "schema_share_BIORED_ex_NEG": variance[BENCH_BIORED]["share_schema"],
    "schema_share_BC5CDR_DD":   variance[BENCH_BC5CDR]["share_schema"],
    "encoder_share_KB_hit_A":    variance[PRIMARY_KB]["share_encoder"],
    "encoder_share_BIORED_ex_NEG": variance[BENCH_BIORED]["share_encoder"],
    "encoder_share_BC5CDR_DD":   variance[BENCH_BC5CDR]["share_encoder"],
}
analysis["H7_asymmetry_preview"] = asymmetry

# -----------------------------------------------------------------------------
# (4) Cross-metric correlations (Phase A-only; for RQ4/Path β preview)
# -----------------------------------------------------------------------------

corr = {}
for (a, b) in [(BENCH_BIORED, PRIMARY_KB),
               (BENCH_BC5CDR, PRIMARY_KB),
               (BENCH_BIORED, BENCH_BC5CDR),
               ("biored_macro_f1", PRIMARY_KB)]:
    x, y = df[a].tolist(), df[b].tolist()
    r_p = pearson(x, y)
    r_s = spearman(x, y)
    p_lo, p_hi = boot_corr_ci(x, y, fn=pearson)
    s_lo, s_hi = boot_corr_ci(x, y, fn=spearman)
    corr[f"{a}__{b}"] = {
        "n": len(x),
        "pearson_r": r_p, "pearson_ci": [p_lo, p_hi],
        "spearman_rho": r_s, "spearman_ci": [s_lo, s_hi],
    }
# Also cell-level: per (encoder, schema) cell-mean correlation over 12 cells
cell_means = []
for enc in ENCODERS:
    for sch in SCHEMAS:
        sub = df[(df.encoder == enc) & (df.schema == sch)]
        cell_means.append({
            "encoder": enc, "schema": sch,
            BENCH_BIORED: sub[BENCH_BIORED].mean(),
            BENCH_BC5CDR: sub[BENCH_BC5CDR].mean(),
            PRIMARY_KB: sub[PRIMARY_KB].mean(),
        })
cm = pd.DataFrame(cell_means)
for (a, b) in [(BENCH_BIORED, PRIMARY_KB),
               (BENCH_BC5CDR, PRIMARY_KB)]:
    x, y = cm[a].tolist(), cm[b].tolist()
    corr[f"cell_mean__{a}__{b}"] = {
        "n": len(x),
        "pearson_r": pearson(x, y),
        "spearman_rho": spearman(x, y),
    }
analysis["cross_metric_correlations"] = corr

# -----------------------------------------------------------------------------
# (5) Per-head BioRED F1 by schema (feasibility against 0.05 floor)
# -----------------------------------------------------------------------------

head_f1 = {}
HEAD_LABELS_BY_SCHEMA = {
    "Sflat": ["ASSOCIATION_GENERAL", "DRUG_DISEASE", "DRUG_GENE_REGULATION"],
    "Spair": ["ASSOCIATION_GENERAL", "DRUG_DISEASE", "DRUG_GENE_REGULATION",
              "DRUG_VARIANT_ASSOC", "GENE_DISEASE", "GENE_GENE_ASSOC",
              "VARIANT_DISEASE"],
    "Smech": ["DRUG_DISEASE", "DGR_ACTIVATE", "DGR_INHIBIT", "DGR_METABOLIC",
              "DGR_REGULATE", "DGR_STRUCTURAL", "DRUG_VARIANT_ASSOC",
              "GENE_DISEASE", "GENE_GENE_ASSOC", "VARIANT_DISEASE",
              "ASSOCIATION_GENERAL", "DRUG_GENE_REGULATION"],
}
for sch in SCHEMAS:
    sub = df[df.schema == sch]
    heads = {}
    for lab in HEAD_LABELS_BY_SCHEMA[sch]:
        col = f"biored_f1__{lab}"
        sup_col = f"biored_support__{lab}"
        if col not in sub.columns: continue
        f1s = sub[col].dropna().tolist()
        sup = sub[sup_col].dropna().tolist() if sup_col in sub.columns else []
        mean_sup = statistics.mean(sup) if sup else 0
        if mean_sup == 0:
            heads[lab] = {"mean_f1": 0.0, "mean_support": 0, "status": "no_test_support",
                          "above_0.05": False}
            continue
        mean_f1 = statistics.mean(f1s) if f1s else 0.0
        sd_f1 = statistics.stdev(f1s) if len(f1s) > 1 else 0.0
        heads[lab] = {"mean_f1": mean_f1, "sd_f1": sd_f1, "mean_support": mean_sup,
                      "above_0.05": mean_f1 > 0.05, "status": "tested"}
    head_f1[sch] = heads
analysis["per_head_f1"] = head_f1

# -----------------------------------------------------------------------------
# (6) All permutation tests + BH-FDR
# -----------------------------------------------------------------------------

perm_results = []
for m in [PRIMARY_KB, "kb_hit_A_singlelabel", "kb_pmass_B_setvalued", "kb_auc_C_setvalued",
          BENCH_BIORED, "biored_macro_f1", BENCH_BC5CDR]:
    for (a, b) in [("Sflat", "Spair"), ("Sflat", "Smech"), ("Spair", "Smech")]:
        va = df[df.schema == a][m].tolist()
        vb = df[df.schema == b][m].tolist()
        diff, p = perm_test(va, vb)
        lo, hi = boot_diff_ci(va, vb)
        d = cohens_d(va, vb)
        perm_results.append({
            "metric": m, "a": a, "b": b,
            "mean_a": statistics.mean(va), "mean_b": statistics.mean(vb),
            "diff": diff, "ci_lo": lo, "ci_hi": hi, "p_value": p, "cohens_d": d,
        })

# BH-FDR across the primary KB × benchmark × 3 schema-pair family (7 metrics × 3 pairs = 21)
p_list = [r["p_value"] for r in perm_results]
p_adj = benjamini_hochberg(p_list, alpha=0.05)
for r, p_bh in zip(perm_results, p_adj):
    r["p_adj_BH"] = p_bh
    r["sig_BH_0.05"] = p_bh < 0.05
analysis["permutation_tests_BH"] = perm_results

# -----------------------------------------------------------------------------
# (7) Encoder-stratified schema comparison on KB_hit_A
# -----------------------------------------------------------------------------

strat = {}
for enc in ENCODERS:
    enc_rows = []
    for (a, b) in [("Sflat", "Spair"), ("Sflat", "Smech"), ("Spair", "Smech")]:
        va = df[(df.encoder == enc) & (df.schema == a)][PRIMARY_KB].tolist()
        vb = df[(df.encoder == enc) & (df.schema == b)][PRIMARY_KB].tolist()
        diff, p = perm_test(va, vb, n=2000)
        lo, hi = boot_diff_ci(va, vb, n=2000)
        d = cohens_d(va, vb)
        enc_rows.append({"a": a, "b": b,
                         "mean_a": statistics.mean(va), "mean_b": statistics.mean(vb),
                         "diff": diff, "ci_lo": lo, "ci_hi": hi, "p_value": p, "cohens_d": d})
    strat[enc] = enc_rows
analysis["encoder_stratified_KB_hit_A"] = strat

# -----------------------------------------------------------------------------
# (8) Intraclass correlation (ICC) at cell level for BioRED ex-NEG, KB_hit_A
# -----------------------------------------------------------------------------

def icc_oneway(df, metric):
    """One-way random-effects ICC(1,1) at the (encoder, schema) cell level."""
    cell_values = defaultdict(list)
    for _, row in df.iterrows():
        cell_values[(row.encoder, row.schema)].append(row[metric])
    cells_vals = list(cell_values.values())
    k = len(cells_vals[0])  # 10 seeds
    n = len(cells_vals)     # 12 cells
    N = n * k
    grand = sum(sum(v) for v in cells_vals) / N
    ms_between = k * sum((statistics.mean(v) - grand)**2 for v in cells_vals) / (n - 1)
    ms_within = sum(sum((x - statistics.mean(v))**2 for x in v) for v in cells_vals) / (N - n)
    icc = (ms_between - ms_within) / (ms_between + (k-1) * ms_within)
    return {"ICC_1_1": icc, "MS_between": ms_between, "MS_within": ms_within, "k": k, "n_cells": n}

icc = {}
for m in [PRIMARY_KB, BENCH_BIORED, BENCH_BC5CDR, "biored_macro_f1"]:
    icc[m] = icc_oneway(df, m)
analysis["ICC_cell_level"] = icc

# -----------------------------------------------------------------------------
# Write JSON + Markdown report
# -----------------------------------------------------------------------------

out_json = ANALYSIS_DIR / "phase_a_analysis.json"
out_json.write_text(json.dumps(analysis, indent=2, default=float))
print(f"WROTE: {out_json}")

def fmt(v, d=4):
    return f"{v:.{d}f}" if isinstance(v, (int, float)) and not math.isnan(v) else "—"

lines = ["# Phase A — Comprehensive Analysis", "",
         f"**Scope**: {analysis['n_runs']} runs, {len(SCHEMAS)} schemas × {len(ENCODERS)} encoders × 10 seeds",
         f"**Primary KB metric**: `kb_hit_A_setvalued` (§3.3 / §6.6)",
         ""]

lines += ["## 1. Schema selection (§6.6 applied)", "",
          f"Ranking by `kb_hit_A_setvalued`: " + " > ".join(
              f"**{s}** ({by_schema[s][PRIMARY_KB]['mean']:.4f} "
              f"[{by_schema[s][PRIMARY_KB]['ci_lo']:.4f}, "
              f"{by_schema[s][PRIMARY_KB]['ci_hi']:.4f}])"
              for s in ranking),
          ""]
lines += ["### Applied reasoning", ""]
for r in reasoning:
    lines.append(f"- {r}")
lines += ["", f"**Outcome: {outcome}**. Phase B schemas: `" + ", ".join(phase_b_schemas) + "`.", ""]

lines += ["## 2. Pooled schema means (n=40 each)", "",
          "| Schema | KB_hit_A_sv | KB_hit_A_sl | KB_pmass_B_sv | KB_auc_C_sv | BioRED macro | BioRED ex-NEG | BC5CDR DD | legacy KB_surface |",
          "|---|---|---|---|---|---|---|---|---|"]
for s in SCHEMAS:
    row = by_schema[s]
    lines.append("| " + " | ".join([
        s,
        f"{row[PRIMARY_KB]['mean']:.4f} ±{row[PRIMARY_KB]['sd']:.4f}",
        f"{row['kb_hit_A_singlelabel']['mean']:.4f} ±{row['kb_hit_A_singlelabel']['sd']:.4f}",
        f"{row['kb_pmass_B_setvalued']['mean']:.4f} ±{row['kb_pmass_B_setvalued']['sd']:.4f}",
        f"{row['kb_auc_C_setvalued']['mean']:.4f} ±{row['kb_auc_C_setvalued']['sd']:.4f}",
        f"{row['biored_macro_f1']['mean']:.4f} ±{row['biored_macro_f1']['sd']:.4f}",
        f"{row[BENCH_BIORED]['mean']:.4f} ±{row[BENCH_BIORED]['sd']:.4f}",
        f"{row[BENCH_BC5CDR]['mean']:.4f} ±{row[BENCH_BC5CDR]['sd']:.4f}",
        f"{row[LEGACY_KB]['mean']:.4f} ±{row[LEGACY_KB]['sd']:.4f}",
    ]) + " |")
lines.append("")

lines += ["## 3. Encoder × schema (primary: KB_hit_A_setvalued)", "",
          "| encoder | " + " | ".join(SCHEMAS) + " |",
          "|---" + "|---" * len(SCHEMAS) + "|"]
for enc in ENCODERS:
    cells_row = [enc]
    for sch in SCHEMAS:
        c = cells[f"{enc}_{sch}"][PRIMARY_KB]
        cells_row.append(f"{c['mean']:.4f} [{c['ci_lo']:.4f}, {c['ci_hi']:.4f}]")
    lines.append("| " + " | ".join(cells_row) + " |")
lines.append("")

lines += ["### Encoder × schema (BioRED ex-NEG)", "",
          "| encoder | " + " | ".join(SCHEMAS) + " |",
          "|---" + "|---" * len(SCHEMAS) + "|"]
for enc in ENCODERS:
    row = [enc]
    for sch in SCHEMAS:
        c = cells[f"{enc}_{sch}"][BENCH_BIORED]
        row.append(f"{c['mean']:.4f} [{c['ci_lo']:.4f}, {c['ci_hi']:.4f}]")
    lines.append("| " + " | ".join(row) + " |")
lines.append("")

lines += ["### Encoder × schema (BC5CDR DRUG_DISEASE F1)", "",
          "| encoder | " + " | ".join(SCHEMAS) + " |",
          "|---" + "|---" * len(SCHEMAS) + "|"]
for enc in ENCODERS:
    row = [enc]
    for sch in SCHEMAS:
        c = cells[f"{enc}_{sch}"][BENCH_BC5CDR]
        row.append(f"{c['mean']:.4f} [{c['ci_lo']:.4f}, {c['ci_hi']:.4f}]")
    lines.append("| " + " | ".join(row) + " |")
lines.append("")

lines += ["## 4. H7 variance decomposition (§ preview)", "",
          "Fraction of total SS attributable to each design factor (plus within-cell/seed residual).",
          "",
          "| Metric | schema | encoder | interaction | within-cell (seed) |",
          "|---|---|---|---|---|"]
for m in [PRIMARY_KB, BENCH_BIORED, BENCH_BC5CDR, LEGACY_KB, "biored_macro_f1"]:
    v = variance[m]
    lines.append(f"| `{m}` | {v['share_schema']*100:.1f}% | {v['share_encoder']*100:.1f}% "
                 f"| {v['share_interaction']*100:.1f}% | {v['share_within_cell']*100:.1f}% |")
lines.append("")
lines += [f"Schema-variance asymmetry: schema explains "
          f"{asymmetry['schema_share_BIORED_ex_NEG']*100:.1f}% of BioRED ex-NEG variance, "
          f"{asymmetry['schema_share_BC5CDR_DD']*100:.1f}% of BC5CDR DD variance, and "
          f"{asymmetry['schema_share_KB_hit_A']*100:.1f}% of KB_hit_A variance.", ""]

lines += ["## 5. Cross-metric correlations (n=120 seed-level, Phase A pooled)", "",
          "| Metric pair | n | Pearson r [95% CI] | Spearman ρ [95% CI] |",
          "|---|---|---|---|"]
for key, v in corr.items():
    if key.startswith("cell_mean__"): continue
    lines.append(f"| `{key}` | {v['n']} | "
                 f"{v['pearson_r']:+.3f} [{v['pearson_ci'][0]:+.3f}, {v['pearson_ci'][1]:+.3f}] | "
                 f"{v['spearman_rho']:+.3f} [{v['spearman_ci'][0]:+.3f}, {v['spearman_ci'][1]:+.3f}] |")
lines.append("")
lines += ["### Cell-mean correlations (n=12 cells)",
          "",
          "| Metric pair | Pearson r | Spearman ρ |",
          "|---|---|---|"]
for key, v in corr.items():
    if not key.startswith("cell_mean__"): continue
    lines.append(f"| `{key}` | {v['pearson_r']:+.3f} | {v['spearman_rho']:+.3f} |")
lines.append("")

lines += ["## 6. Per-head BioRED F1 (with support and 0.05 floor)", ""]
for sch in SCHEMAS:
    lines.append(f"### Schema: {sch}")
    lines.append("")
    lines.append("| head | mean F1 | sd | mean support | status | above 0.05? |")
    lines.append("|---|---|---|---|---|---|")
    for lab, h in sorted(head_f1[sch].items(), key=lambda kv: -kv[1]["mean_f1"]):
        lines.append(f"| {lab} | {h['mean_f1']:.4f} | {h.get('sd_f1', 0):.4f} "
                     f"| {h.get('mean_support', 0):.0f} | {h['status']} "
                     f"| {'yes' if h['above_0.05'] else 'NO'} |")
    lines.append("")

lines += ["## 7. Permutation tests with BH-FDR (α=0.05, family size = "
          f"{len(perm_results)})", "",
          "| metric | A | B | mean_A | mean_B | diff | 95% CI | p | p_BH | d | sig |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in perm_results:
    lines.append(f"| {r['metric']} | {r['a']} | {r['b']} "
                 f"| {r['mean_a']:.4f} | {r['mean_b']:.4f} "
                 f"| {r['diff']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
                 f"| {r['p_value']:.4f} | {r['p_adj_BH']:.4f} | {r['cohens_d']:+.3f} "
                 f"| {'✓' if r['sig_BH_0.05'] else ''} |")
lines.append("")

lines += ["## 8. Encoder-stratified schema comparison (KB_hit_A_setvalued)", ""]
for enc in ENCODERS:
    lines.append(f"### {enc}")
    lines.append("")
    lines.append("| A | B | mean_A | mean_B | diff | 95% CI | p | d |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in strat[enc]:
        lines.append(f"| {r['a']} | {r['b']} | {r['mean_a']:.4f} | {r['mean_b']:.4f} "
                     f"| {r['diff']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
                     f"| {r['p_value']:.4f} | {r['cohens_d']:+.3f} |")
    lines.append("")

lines += ["## 9. Intraclass correlation (cell-level, 12 cells × 10 seeds)", "",
          "| Metric | ICC(1,1) | MS_between | MS_within | interpretation |",
          "|---|---|---|---|---|"]
def icc_label(icc_val):
    if icc_val < 0: return "below zero (within > between)"
    if icc_val < 0.3: return "poor"
    if icc_val < 0.5: return "fair"
    if icc_val < 0.75: return "good"
    return "excellent"
for m, v in icc.items():
    lines.append(f"| {m} | {v['ICC_1_1']:.4f} | {v['MS_between']:.6f} "
                 f"| {v['MS_within']:.6f} | {icc_label(v['ICC_1_1'])} |")
lines.append("")

(ANALYSIS_DIR / "phase_a_analysis.md").write_text("\n".join(lines))
print(f"WROTE: {ANALYSIS_DIR / 'phase_a_analysis.md'}")
