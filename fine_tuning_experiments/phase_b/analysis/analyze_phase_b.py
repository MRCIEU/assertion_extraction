#!/usr/bin/env python3.11
"""Phase B — primary-hypothesis analysis scaffold.

Reads Phase B eval JSONs (one per run, analogous to Phase A's
`phase_a_eval.json`) aggregated into a flat results CSV, and evaluates the
six Phase B primary hypotheses H1–H5 and H7 per the pre-committed decision
rules in `paper_development_design.md` §7.2, §9.1, §9.2, §9.5, §9.6.

H6 (mechanism-stratified coupling slopes) has its own dedicated script:
    fine_tuning_experiments.phase_b.analysis.h6_coupling_slopes

Inputs (required once Phase B runs complete):
  - `fine_tuning_experiments/phase_b/phase_b_results.csv`
    (produced by an aggregator analogous to `schema_exp/eval/aggregate_phase_a.py`;
    each row is one of the 360 main runs + 5 RB reference seeds)
    Columns: run_id, encoder ∈ {PB, BL, PL, RB}, architecture ∈ {P, MT},
             update ∈ {FT, LR}, schedule ∈ {T1B, T1F, T2},
             seed ∈ 1..10, schema = Spair,
             biored_macro_f1, biored_macro_f1_ex_neg,
             bc5cdr_drug_disease_f1, kb_hit_A_setvalued, …

Outputs:
  - `fine_tuning_experiments/phase_b/phase_b_analysis.json`
  - `fine_tuning_experiments/phase_b/phase_b_analysis.md`

Status at lock time: SKELETON. Each hypothesis function is implemented up
to the decision rule but will short-circuit with a clear "phase_b data not
yet available" diagnostic until `phase_b_results.csv` is present and
populated with ≥ 36 complete cells × 10 seeds.

All bootstrap CIs use the same paired-(cell, seed) structure pre-committed
in §6.6: resample the matched cells with replacement, compute the per-cell
paired difference, average, B = 10000.  FDR uses Benjamini–Hochberg.
Deterministic under `--seed` (default 20260416).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

SCRIPT = Path(__file__).resolve()
PHASE_B_DIR = SCRIPT.parent.parent      # .../phase_b
DEFAULT_CSV = PHASE_B_DIR / "phase_b_results.csv"
OUT_JSON = PHASE_B_DIR / "phase_b_analysis.json"
OUT_MD = PHASE_B_DIR / "phase_b_analysis.md"

ENCODERS_MAIN = ("PB", "BL", "PL")
ENCODER_REFERENCE = "RB"
ARCHS = ("P", "MT")              # pipeline, shared-multitask
UPDATES = ("FT", "LR")           # full-FT, LoRA
SCHEDULES = ("T1B", "T1F", "T2") # T1_biored_only, T1_flat, T1→T2 staged
N_SEEDS = 10
BOOT_B = 10000
SEED_DEFAULT = 20260416

# Metric column names (align with Phase A CSV + Phase B evaluator output)
BIORED_EX_NEG = "biored_macro_f1_ex_neg"
BC5CDR_DD = "bc5cdr_drug_disease_f1"
KB_HIT_A = "kb_hit_A_setvalued"

# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    import csv as _csv
    with csv_path.open() as f:
        rows = list(_csv.DictReader(f))
    # Coerce numeric columns
    for r in rows:
        for k in (BIORED_EX_NEG, BC5CDR_DD, KB_HIT_A, "biored_macro_f1"):
            if r.get(k) not in (None, "", "nan"):
                r[k] = float(r[k])
            else:
                r[k] = None
        if r.get("seed") not in (None, ""):
            r["seed"] = int(r["seed"])
    return rows


def expected_row_count(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Phase B expects 360 main runs + 5 RB reference runs = 365."""
    n_total = len(rows)
    n_main = sum(1 for r in rows if r.get("encoder") in ENCODERS_MAIN)
    n_ref = sum(1 for r in rows if r.get("encoder") == ENCODER_REFERENCE)
    return {"n_total": n_total, "n_main": n_main, "n_reference": n_ref,
            "expected": {"main": 360, "reference": 5, "total": 365}}


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def paired_mean_diff_ci(pairs: list[tuple[float, float]], *, n: int = BOOT_B,
                        seed: int = SEED_DEFAULT, alpha: float = 0.05
                        ) -> tuple[float, float, float]:
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    mean = statistics.mean(a - b for a, b in pairs)
    rng = random.Random(seed)
    k = len(pairs)
    boots = []
    for _ in range(n):
        idxs = [rng.randrange(k) for _ in range(k)]
        boots.append(statistics.mean(pairs[i][0] - pairs[i][1] for i in idxs))
    boots.sort()
    return mean, boots[int(alpha / 2 * n)], boots[int((1 - alpha / 2) * n)]


def paired_t(pairs: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Returns (t, df, two-sided p). Uses Student's t approximation."""
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 2:
        return float("nan"), 0, float("nan")
    diffs = [a - b for a, b in pairs]
    m = statistics.mean(diffs)
    s = statistics.stdev(diffs)
    n = len(diffs)
    if s == 0:
        return float("nan"), n - 1, 0.0 if m == 0 else 0.0
    t = m / (s / math.sqrt(n))
    p = 2 * (1 - _t_cdf(abs(t), n - 1))
    return t, n - 1, p


def _t_cdf(t: float, df: int) -> float:
    """Student's t CDF via the symmetric beta-function identity.
    Uses math.lgamma + numerical integration; adequate for df ≥ 2."""
    # Uses the relationship P(T ≤ t) = 1 - 0.5 * I_{df/(df+t^2)}(df/2, 1/2)
    # with I the regularised incomplete beta.
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    # regularised incomplete beta via continued fraction (Lentz)
    a, b = df / 2.0, 0.5
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(max(x, 1e-300)) + b * math.log(max(1 - x, 1e-300)))
    def _betacf(a: float, b: float, x: float) -> float:
        fpmin = 1e-300
        qab, qap, qam = a + b, a + 1, a - 1
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < fpmin: d = fpmin
        d = 1.0 / d
        h = d
        for m in range(1, 200):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin: d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin: c = fpmin
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin: d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin: c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 3e-8:
                break
        return h
    if x < (a + 1.0) / (a + b + 2.0):
        inc = bt * _betacf(a, b, x) / a
    else:
        inc = 1.0 - bt * _betacf(b, a, 1.0 - x) / b
    # P(T ≤ t) depending on sign of t
    return 1.0 - 0.5 * inc if t >= 0 else 0.5 * inc


def wilcoxon_signed_rank_p(pairs: list[tuple[float, float]]) -> float:
    """Asymptotic normal approximation; adequate for n ≥ 15."""
    diffs = [a - b for a, b in pairs if a is not None and b is not None and (a - b) != 0]
    n = len(diffs)
    if n < 6:
        return float("nan")
    ranks = _ranks([abs(d) for d in diffs])
    w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    z = (w_plus - mean) / math.sqrt(var)
    # Two-sided p from standard normal
    return 2 * (1 - _phi(abs(z)))


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cohens_d_paired(pairs: list[tuple[float, float]]) -> float:
    diffs = [a - b for a, b in pairs if a is not None and b is not None]
    if len(diffs) < 2:
        return 0.0
    sd = statistics.stdev(diffs)
    if sd == 0:
        return 0.0
    return statistics.mean(diffs) / sd


def bh_fdr(pvals: list[float], alpha: float = 0.05) -> list[float]:
    """Benjamini–Hochberg adjusted p-values."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = n - rank + 1  # sorted rank (largest p first)
        q = min(prev, pvals[i] * n / k)
        adj[i] = q
        prev = q
    return adj


def tost_equivalence(pairs: list[tuple[float, float]], margin: float = 0.03,
                     alpha: float = 0.05) -> dict[str, Any]:
    """Two one-sided t tests for equivalence.  Returns 90% CI and verdict."""
    diffs = [a - b for a, b in pairs if a is not None and b is not None]
    n = len(diffs)
    if n < 2:
        return {"n": n, "ci90": [None, None], "equivalent": None,
                "reason": "insufficient paired observations"}
    m = statistics.mean(diffs)
    s = statistics.stdev(diffs)
    se = s / math.sqrt(n)
    # 90 % CI for mean diff
    from math import inf
    # two-sided 90 % ↔ one-sided 95 % critical for TOST symmetry
    # use asymptotic (normal) crit; n ≥ 10 is fine for margin 0.03
    crit = 1.6448536269514722  # 95% one-sided
    lo, hi = m - crit * se, m + crit * se
    equivalent = (-margin <= lo) and (hi <= margin)
    return {"n": n, "mean_diff": m, "ci90": [lo, hi], "margin": margin,
            "equivalent": equivalent}


# -----------------------------------------------------------------------------
# Cell lookup
# -----------------------------------------------------------------------------

CellKey = tuple[str, str, str, str, int]  # (encoder, arch, update, schedule, seed)


def index_cells(rows: list[dict[str, Any]]) -> dict[CellKey, dict[str, Any]]:
    return {(r["encoder"], r["architecture"], r["update"], r["schedule"], int(r["seed"])): r
            for r in rows if all(k in r for k in ("encoder", "architecture",
                                                  "update", "schedule", "seed"))}


def cells_matching(idx: dict[CellKey, dict], *, encoder=None, arch=None,
                   update=None, schedule=None) -> list[dict[str, Any]]:
    out = []
    for k, r in idx.items():
        enc, a, u, s, _sd = k
        if encoder and enc != encoder: continue
        if arch and a != arch: continue
        if update and u != update: continue
        if schedule and s != schedule: continue
        out.append(r)
    return out


def paired_over_seeds(idx: dict[CellKey, dict],
                      a: dict[str, str], b: dict[str, str],
                      metric: str) -> list[tuple[float, float]]:
    """Return [(metric_a[seed], metric_b[seed]) for each seed]."""
    pairs = []
    for seed in range(1, N_SEEDS + 1):
        ka = (a["encoder"], a["arch"], a["update"], a["schedule"], seed)
        kb = (b["encoder"], b["arch"], b["update"], b["schedule"], seed)
        ra, rb = idx.get(ka), idx.get(kb)
        if ra and rb and ra.get(metric) is not None and rb.get(metric) is not None:
            pairs.append((ra[metric], rb[metric]))
    return pairs


# -----------------------------------------------------------------------------
# Hypothesis tests
# -----------------------------------------------------------------------------

def h1_encoder(idx: dict[CellKey, dict]) -> dict[str, Any]:
    """PL > {PB, BL} on BioRED ex-NEG at matched config (anchor: P × FT × T2)."""
    anchor = {"arch": "P", "update": "FT", "schedule": "T2"}
    tests = []
    pvals_t, pvals_w, labels = [], [], []
    for (enc_a, enc_b) in [("PL", "PB"), ("PL", "BL"), ("PB", "BL")]:
        pairs = paired_over_seeds(idx, {**anchor, "encoder": enc_a},
                                  {**anchor, "encoder": enc_b}, BIORED_EX_NEG)
        if len(pairs) < 2:
            tests.append({"contrast": f"{enc_a}_vs_{enc_b}", "status": "no_data",
                          "n_pairs": len(pairs)})
            continue
        m, lo, hi = paired_mean_diff_ci(pairs)
        _, _, p_t = paired_t(pairs)
        p_w = wilcoxon_signed_rank_p(pairs)
        d = cohens_d_paired(pairs)
        tests.append({"contrast": f"{enc_a}_vs_{enc_b}",
                      "mean_diff": m, "ci95": [lo, hi], "cohens_d": d,
                      "paired_t_p": p_t, "wilcoxon_p": p_w,
                      "n_pairs": len(pairs)})
        pvals_t.append(p_t); pvals_w.append(p_w); labels.append(f"{enc_a}_vs_{enc_b}")
    # FDR over 3 pairwise tests
    if pvals_t:
        q_t = bh_fdr(pvals_t); q_w = bh_fdr(pvals_w)
        for lab, qt, qw in zip(labels, q_t, q_w):
            for t in tests:
                if t.get("contrast") == lab:
                    t["q_t"] = qt; t["q_w"] = qw
    # Decision rule: PL > both by Δ ≥ 0.02 with q < 0.05 on both tests
    pl_pb = next((t for t in tests if t.get("contrast") == "PL_vs_PB"), None)
    pl_bl = next((t for t in tests if t.get("contrast") == "PL_vs_BL"), None)
    def _confirmed(t):
        return (t and t.get("mean_diff") is not None and t["mean_diff"] >= 0.02
                and t.get("q_t", 1.0) < 0.05 and t.get("q_w", 1.0) < 0.05)
    def _null(t):
        return (t and t.get("mean_diff") is not None and abs(t["mean_diff"]) < 0.01
                and t.get("q_t", 0) > 0.10 and t.get("q_w", 0) > 0.10)
    def _inverted(t):
        return (t and t.get("mean_diff") is not None and t["mean_diff"] <= -0.02
                and t.get("q_t", 1.0) < 0.05)
    if not pvals_t:
        verdict = "phase_b_data_not_yet_available"
    elif _confirmed(pl_pb) and _confirmed(pl_bl):
        verdict = "confirmed"
    elif _null(pl_pb) and _null(pl_bl):
        verdict = "null"
    elif _inverted(pl_pb) or _inverted(pl_bl):
        verdict = "inverted"
    else:
        verdict = "partial_or_intermediate"
    return {"hypothesis": "H1", "tests": tests, "verdict": verdict,
            "anchor": anchor}


def h2_corpus(idx: dict[CellKey, dict]) -> dict[str, Any]:
    """T1_flat vs T1_biored_only on BC5CDR DD at PB × P × FT."""
    base = {"encoder": "PB", "arch": "P", "update": "FT"}
    pairs = paired_over_seeds(idx, {**base, "schedule": "T1F"},
                              {**base, "schedule": "T1B"}, BC5CDR_DD)
    if len(pairs) < 2:
        return {"hypothesis": "H2", "status": "phase_b_data_not_yet_available",
                "n_pairs": len(pairs)}
    m, lo, hi = paired_mean_diff_ci(pairs)
    _, _, p_t = paired_t(pairs)
    p_w = wilcoxon_signed_rank_p(pairs)
    d = cohens_d_paired(pairs)
    if m >= 0.03 and p_t < 0.05 and p_w < 0.05:
        verdict = "confirmed"
    elif abs(m) < 0.02:
        verdict = "null"
    elif m <= -0.03 and p_t < 0.05:
        verdict = "inverted"
    else:
        verdict = "partial_or_intermediate"
    return {"hypothesis": "H2", "mean_diff": m, "ci95": [lo, hi],
            "cohens_d": d, "paired_t_p": p_t, "wilcoxon_p": p_w,
            "n_pairs": len(pairs), "verdict": verdict}


def h3_schedule(idx: dict[CellKey, dict]) -> dict[str, Any]:
    """T1→T2 vs T1_flat on BioRED ex-NEG and BC5CDR DD at PB/BL/PL × P × FT."""
    tests, pvals_t, pvals_w, labels = [], [], [], []
    for enc in ENCODERS_MAIN:
        for metric, label in [(BIORED_EX_NEG, "biored_ex_neg"), (BC5CDR_DD, "bc5cdr_dd")]:
            pairs = paired_over_seeds(idx,
                {"encoder": enc, "arch": "P", "update": "FT", "schedule": "T2"},
                {"encoder": enc, "arch": "P", "update": "FT", "schedule": "T1F"},
                metric)
            if len(pairs) < 2:
                tests.append({"contrast": f"{enc}_{label}", "status": "no_data",
                              "n_pairs": len(pairs)})
                continue
            m, lo, hi = paired_mean_diff_ci(pairs)
            _, _, p_t = paired_t(pairs)
            p_w = wilcoxon_signed_rank_p(pairs)
            tests.append({"contrast": f"{enc}_{label}",
                          "mean_diff": m, "ci95": [lo, hi],
                          "paired_t_p": p_t, "wilcoxon_p": p_w,
                          "n_pairs": len(pairs)})
            pvals_t.append(p_t); pvals_w.append(p_w)
            labels.append(f"{enc}_{label}")
    if pvals_t:
        q_t = bh_fdr(pvals_t); q_w = bh_fdr(pvals_w)
        for lab, qt, qw in zip(labels, q_t, q_w):
            for t in tests:
                if t.get("contrast") == lab:
                    t["q_t"] = qt; t["q_w"] = qw
    # Decision rule: ≥ 4 of 6 show Δ ≥ 0.02 and q < 0.05
    n_confirmed = sum(1 for t in tests
                      if t.get("mean_diff") is not None
                      and t["mean_diff"] >= 0.02
                      and t.get("q_t", 1.0) < 0.05
                      and t.get("q_w", 1.0) < 0.05)
    if not pvals_t:
        verdict = "phase_b_data_not_yet_available"
    elif n_confirmed >= 4:
        verdict = "confirmed"
    elif n_confirmed >= 2:
        verdict = "partial"
    else:
        verdict = "null"
    return {"hypothesis": "H3", "tests": tests, "n_confirmed": n_confirmed,
            "verdict": verdict}


def h4_update(idx: dict[CellKey, dict]) -> dict[str, Any]:
    """Full-FT vs LoRA on BioRED ex-NEG at PB/BL/PL × P × T2."""
    tests, pvals_t, pvals_w, labels = [], [], [], []
    for enc in ENCODERS_MAIN:
        pairs = paired_over_seeds(idx,
            {"encoder": enc, "arch": "P", "update": "FT", "schedule": "T2"},
            {"encoder": enc, "arch": "P", "update": "LR", "schedule": "T2"},
            BIORED_EX_NEG)
        if len(pairs) < 2:
            tests.append({"encoder": enc, "status": "no_data"})
            continue
        m, lo, hi = paired_mean_diff_ci(pairs)
        _, _, p_t = paired_t(pairs)
        p_w = wilcoxon_signed_rank_p(pairs)
        d = cohens_d_paired(pairs)
        tests.append({"encoder": enc, "mean_diff": m, "ci95": [lo, hi],
                      "cohens_d": d, "paired_t_p": p_t, "wilcoxon_p": p_w,
                      "n_pairs": len(pairs)})
        pvals_t.append(p_t); pvals_w.append(p_w); labels.append(enc)
    if pvals_t:
        q_t = bh_fdr(pvals_t); q_w = bh_fdr(pvals_w)
        for lab, qt, qw in zip(labels, q_t, q_w):
            for t in tests:
                if t.get("encoder") == lab:
                    t["q_t"] = qt; t["q_w"] = qw
    # Confirmed: all 3 encoders show d ≥ 0.5 and q < 0.05
    passes = [t for t in tests
              if t.get("cohens_d", 0) >= 0.5
              and t.get("q_t", 1.0) < 0.05
              and t.get("q_w", 1.0) < 0.05]
    lora_pref = [t for t in tests
                 if t.get("mean_diff") is not None and t["mean_diff"] < 0
                 and t.get("q_t", 1.0) < 0.05]
    if not pvals_t:
        verdict = "phase_b_data_not_yet_available"
    elif len(passes) == 3:
        verdict = "confirmed"
    elif len(passes) == 2:
        verdict = "partial"
    elif lora_pref:
        verdict = "lora_preferred_counter_finding"
    else:
        verdict = "null_or_mixed"
    return {"hypothesis": "H4", "tests": tests, "n_confirmed": len(passes),
            "verdict": verdict}


def h5_architecture(idx: dict[CellKey, dict]) -> dict[str, Any]:
    """TOST pipeline vs shared-multitask at PB and PL × FT × T2, margin ±0.03."""
    tests = []
    for enc in ("PB", "PL"):
        pairs = paired_over_seeds(idx,
            {"encoder": enc, "arch": "P", "update": "FT", "schedule": "T2"},
            {"encoder": enc, "arch": "MT", "update": "FT", "schedule": "T2"},
            BIORED_EX_NEG)
        r = tost_equivalence(pairs, margin=0.03)
        r["encoder"] = enc
        tests.append(r)
    all_equiv = [t for t in tests if t.get("equivalent") is True]
    any_fail = [t for t in tests if t.get("equivalent") is False]
    if all(t.get("equivalent") is None for t in tests):
        verdict = "phase_b_data_not_yet_available"
    elif len(all_equiv) == len(tests):
        verdict = "equivalent"
    elif any_fail:
        verdict = "not_equivalent"
    else:
        verdict = "insufficient_power"
    return {"hypothesis": "H5", "tests": tests, "verdict": verdict}


def h7_variance_asymmetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Variance decomposition: R_B = (design-lever share in BioRED ex-NEG) /
    (design-lever share in KB_hit_A).  §9.5."""
    # Phase B design levers: encoder + arch + update + schedule + pairwise interactions
    main_rows = [r for r in rows if r.get("encoder") in ENCODERS_MAIN]
    if len(main_rows) < 36 * 2:  # at least 2 seeds per cell for any decomposition
        return {"hypothesis": "H7", "status": "phase_b_data_not_yet_available",
                "n_main_rows": len(main_rows)}
    # Simple fraction-of-SS decomposition via sum-of-squares by factor.
    # For each metric: SS_total = Σ (y - y̅)^2; SS_factor = Σ_level n_level (ȳ_level - ȳ)^2.
    def _ss_decomp(metric: str) -> dict[str, float]:
        ys = [r[metric] for r in main_rows if r.get(metric) is not None]
        if len(ys) < 2:
            return {}
        grand = statistics.mean(ys)
        ss_tot = sum((y - grand) ** 2 for y in ys)
        shares = {}
        for factor in ("encoder", "architecture", "update", "schedule"):
            by_level = defaultdict(list)
            for r in main_rows:
                if r.get(metric) is not None:
                    by_level[r[factor]].append(r[metric])
            ss_f = sum(len(vs) * (statistics.mean(vs) - grand) ** 2
                       for vs in by_level.values() if vs)
            shares[factor] = ss_f / ss_tot if ss_tot else 0.0
        # Simple two-way interactions (encoder×arch etc.)
        for fa, fb in [("encoder", "architecture"), ("encoder", "update"),
                       ("encoder", "schedule"), ("architecture", "update"),
                       ("architecture", "schedule"), ("update", "schedule")]:
            by_cell = defaultdict(list)
            for r in main_rows:
                if r.get(metric) is not None:
                    by_cell[(r[fa], r[fb])].append(r[metric])
            ss_ab = sum(len(vs) * (statistics.mean(vs) - grand) ** 2
                        for vs in by_cell.values() if vs)
            # subtract main effects (approximate; full ANOVA needs SS partition)
            shares[f"{fa}_x_{fb}"] = max(0.0, ss_ab / ss_tot - shares[fa] - shares[fb])
        return shares
    decomp = {m: _ss_decomp(m) for m in (BIORED_EX_NEG, KB_HIT_A)}
    if not all(decomp.values()):
        return {"hypothesis": "H7", "status": "insufficient_data",
                "decomposition": decomp}
    # Sum design-lever shares
    def _lever_share(d):
        return sum(v for k, v in d.items() if k != "within_cell_residual")
    R_B = _lever_share(decomp[BIORED_EX_NEG]) / max(1e-9, _lever_share(decomp[KB_HIT_A]))
    if R_B >= 2.0:
        verdict = "configuration_induced_asymmetry_confirmed"
    elif R_B > 1.0:
        verdict = "borderline"
    else:
        verdict = "null_no_asymmetry"
    return {"hypothesis": "H7", "R_B": R_B, "decomposition": decomp,
            "threshold": 2.0, "verdict": verdict}


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def analyze(csv_path: Path, out_json: Path, out_md: Path) -> dict[str, Any]:
    rows = load_rows(csv_path)
    coverage = expected_row_count(rows)
    idx = index_cells(rows)

    result = {
        "input_csv": str(csv_path),
        "coverage": coverage,
        "H1_encoder": h1_encoder(idx),
        "H2_corpus": h2_corpus(idx),
        "H3_schedule": h3_schedule(idx),
        "H4_update_regime": h4_update(idx),
        "H5_architecture_equivalence": h5_architecture(idx),
        "H7_variance_asymmetry": h7_variance_asymmetry(rows),
        "H6_note": ("H6 mechanism-stratified slopes are computed by "
                    "fine_tuning_experiments.phase_b.analysis.h6_coupling_slopes; "
                    "run that script separately with both Phase A and Phase B CSVs."),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, default=str))
    out_md.write_text(render_report(result))
    return result


def render_report(res: dict[str, Any]) -> str:
    cov = res["coverage"]
    lines = ["# Phase B — primary-hypothesis analysis", "",
             f"**Coverage**: {cov['n_total']} runs loaded "
             f"({cov['n_main']} main + {cov['n_reference']} RB reference); "
             f"expected {cov['expected']['total']}.", ""]
    for key in ("H1_encoder", "H2_corpus", "H3_schedule",
                "H4_update_regime", "H5_architecture_equivalence",
                "H7_variance_asymmetry"):
        h = res[key]
        lines.append(f"## {h.get('hypothesis', key)}")
        status = h.get("verdict") or h.get("status")
        lines.append(f"**Verdict**: `{status}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(h, indent=2, default=str))
        lines.append("```")
        lines.append("")
    lines.append("## H6")
    lines.append(res["H6_note"])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()
    res = analyze(args.input, args.out_json, args.out_md)
    cov = res["coverage"]
    print(f"Phase B analysis: {cov['n_total']}/{cov['expected']['total']} runs loaded")
    for key in ("H1_encoder", "H2_corpus", "H3_schedule",
                "H4_update_regime", "H5_architecture_equivalence",
                "H7_variance_asymmetry"):
        v = res[key].get("verdict") or res[key].get("status")
        print(f"  {key}: {v}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
