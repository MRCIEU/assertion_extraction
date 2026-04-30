#!/usr/bin/env python3.11
"""Phase B — primary-hypothesis analysis scaffold.

Reads Phase B eval JSONs (one per run, analogous to Phase A's
`phase_a_eval.json`) aggregated into a flat results CSV, and evaluates the
four Phase B primary hypotheses H1–H4 and H7 per the pre-committed decision
rules in `paper_development_design.md` §7.2, §9.1, §9.2, §9.5, §9.6.

H5 (pipeline ≈ shared-multitask TOST equivalence) is deferred; the
`shared_multitask` architecture was dropped from the Phase B factorial in the
post-lock amendment dated 2026-04-16 (Appendix B, row 2). H5 is retained as
an explicit `deferred_to_future_work` stub so that a reviewer scanning the
output can see the downgrade was deliberate.

H6 (mechanism-stratified coupling slopes) has its own dedicated script:
    fine_tuning_experiments.phase_b.analysis.h6_coupling_slopes

Inputs (required once Phase B runs complete):
  - `fine_tuning_experiments/phase_b/phase_b_results.csv`
    (produced by `fine_tuning_experiments/phase_b/aggregate_phase_b.py`;
    each row is one of the 360 main runs + 10 RB reference seeds)
    Columns: run_id, encoder ∈ {PB, BL, PL, RB},
             update ∈ {FT, LR}, schedule ∈ {T1B, T1F, T2},
             seed ∈ 1..20, schema = Spair,
             biored_macro_f1, biored_macro_f1_ex_neg,
             bc5cdr_drug_disease_f1, kb_hit_A_setvalued, …

Outputs:
  - `fine_tuning_experiments/phase_b/phase_b_analysis.json`
  - `fine_tuning_experiments/phase_b/phase_b_analysis.md`

Status at lock time: SKELETON. Each hypothesis function is implemented up
to the decision rule but will short-circuit with a clear "phase_b data not
yet available" diagnostic until `phase_b_results.csv` is present and
populated with ≥ 18 complete cells × 20 seeds.

All bootstrap CIs use the same paired-(cell, seed) structure pre-committed
in §6.6: resample the matched cells with replacement, compute the per-cell
paired difference, average, B = 10000.  FDR uses Benjamini–Hochberg.
Deterministic under `--seed` (default 20260416).

Post-lock amendment 2026-04-16 (Appendix B row 2): factorial reduced from
3 enc × 2 arch × 2 update × 3 schedule × 10 seeds = 360+5 RB = 365 to
3 enc × 2 update × 3 schedule × 20 seeds = 360+10 RB = 370; arch dropped
from CellKey; H5 → deferred stub; anchors drop the "arch" qualifier.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCRIPT = Path(__file__).resolve()
PHASE_B_DIR = SCRIPT.parent.parent      # .../phase_b
DEFAULT_CSV = PHASE_B_DIR / "phase_b_results.csv"
OUT_JSON = PHASE_B_DIR / "phase_b_analysis.json"
OUT_MD = PHASE_B_DIR / "phase_b_analysis.md"

ENCODERS_MAIN = ("PB", "BL", "PL")
ENCODER_REFERENCE = "RB"
# arch axis dropped per Appendix B row 2 (2026-04-16).
UPDATES = ("FT",)                # realised post-B.24: LoRA arm dropped
SCHEDULES = ("T1B", "T1F", "T2") # T1_biored_only, T1_flat, T1→T2 staged
N_SEEDS = 20
N_SEEDS_REFERENCE = 10
N_CELLS_MAIN = len(ENCODERS_MAIN) * len(UPDATES) * len(SCHEDULES)  # 9
EXPECTED_MAIN = N_CELLS_MAIN * N_SEEDS                              # 180
EXPECTED_REFERENCE = N_SEEDS_REFERENCE                              # 10 (RB × FT × T2 × 10 seeds)
EXPECTED_TOTAL = EXPECTED_MAIN + EXPECTED_REFERENCE                 # 190
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
    """Phase B realised factorial expects 180 FT main + 10 RB = 190."""
    n_total = len(rows)
    n_main = sum(1 for r in rows if r.get("encoder") in ENCODERS_MAIN)
    n_ref = sum(1 for r in rows if r.get("encoder") == ENCODER_REFERENCE)
    return {"n_total": n_total, "n_main": n_main, "n_reference": n_ref,
            "expected": {"main": EXPECTED_MAIN,
                         "reference": EXPECTED_REFERENCE,
                         "total": EXPECTED_TOTAL}}


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def paired_mean_diff_ci(pairs: list[tuple[float, float]],
                        *, n: int = BOOT_B, seed: int = SEED_DEFAULT,
                        alpha: float = 0.05) -> tuple[float, float, float]:
    """Paired bootstrap CI of mean(a - b)."""
    if not pairs:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    n_cells = len(pairs)
    diffs = [a - b for a, b in pairs]
    mean_diff = statistics.mean(diffs)
    boots: list[float] = []
    for _ in range(n):
        idxs = [rng.randrange(n_cells) for _ in range(n_cells)]
        boots.append(statistics.mean(diffs[i] for i in idxs))
    boots.sort()
    lo = boots[int(alpha / 2 * n)]
    hi = boots[int((1 - alpha / 2) * n)]
    return mean_diff, lo, hi


def paired_t(pairs: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Paired t on differences. Returns (t, df, p_two_sided)."""
    if len(pairs) < 2:
        return (float("nan"), float("nan"), float("nan"))
    diffs = [a - b for a, b in pairs]
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    if sd == 0:
        return (float("nan"), len(pairs) - 1, 1.0 if m == 0 else 0.0)
    se = sd / math.sqrt(len(pairs))
    t = m / se
    df = len(pairs) - 1
    # Two-sided p via survival function (approximate; student-t → normal for df > ~30)
    # For smaller df we use a conservative normal-approx since scipy is optional.
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return (t, df, p)


def wilcoxon_signed_rank_p(pairs: list[tuple[float, float]]) -> float:
    """Two-sided Wilcoxon signed-rank p (normal-approx)."""
    if len(pairs) < 6:
        return float("nan")
    diffs = [(a - b) for a, b in pairs if a != b]
    if not diffs:
        return 1.0
    abs_d = sorted((abs(d), 1 if d > 0 else -1) for d in diffs)
    # Assign ranks (averaging ties)
    ranks: list[float] = [0.0] * len(abs_d)
    i = 0
    while i < len(abs_d):
        j = i
        while j + 1 < len(abs_d) and abs_d[j + 1][0] == abs_d[i][0]:
            j += 1
        avg_r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_r
        i = j + 1
    W_plus = sum(r for r, (_, s) in zip(ranks, abs_d) if s > 0)
    n = len(abs_d)
    mean_W = n * (n + 1) / 4
    var_W = n * (n + 1) * (2 * n + 1) / 24
    if var_W == 0:
        return 1.0
    z = (W_plus - mean_W) / math.sqrt(var_W)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return p


def cohens_d_paired(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return float("nan")
    diffs = [a - b for a, b in pairs]
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    return m / sd if sd else float("nan")


def bh_fdr(pvals: list[float], alpha: float = 0.05) -> list[float]:
    """Benjamini–Hochberg adjusted q-values, in the same order as input."""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        r = n - rank + 1
        adj = pvals[i] * n / r
        prev = min(prev, adj)
        q[i] = min(prev, 1.0)
    return q


def tost_equivalence(pairs: list[tuple[float, float]], *,
                     margin: float = 0.03, alpha: float = 0.05) -> dict[str, Any]:
    """Two one-sided tests against ±margin."""
    if len(pairs) < 2:
        return {"status": "insufficient_data", "n_pairs": len(pairs)}
    m, lo, hi = paired_mean_diff_ci(pairs)
    # Equivalent iff (1-2α) CI is within [-margin, +margin].
    # Using the two-sided 90 %-equivalent CI is standard for α = 0.05 TOST.
    m2, lo90, hi90 = paired_mean_diff_ci(pairs, alpha=0.10)
    equivalent = (lo90 >= -margin) and (hi90 <= margin)
    return {"mean_diff": m, "ci95": [lo, hi], "ci90_for_tost": [lo90, hi90],
            "margin": margin, "equivalent": equivalent,
            "n_pairs": len(pairs)}


# -----------------------------------------------------------------------------
# Factorial indexing (arch axis dropped per Appendix B amendment 2026-04-16)
# -----------------------------------------------------------------------------

CellKey = tuple[str, str, str, int]  # (encoder, update, schedule, seed)

_CELL_FIELDS = ("encoder", "update", "schedule", "seed")


def index_cells(rows: list[dict[str, Any]]) -> dict[CellKey, dict[str, Any]]:
    return {(r["encoder"], r["update"], r["schedule"], int(r["seed"])): r
            for r in rows if all(k in r for k in _CELL_FIELDS)}


def cells_matching(idx: dict[CellKey, dict], *, encoder=None,
                   update=None, schedule=None) -> list[dict[str, Any]]:
    out = []
    for k, r in idx.items():
        enc, u, s, _sd = k
        if encoder and enc != encoder: continue
        if update and u != update: continue
        if schedule and s != schedule: continue
        out.append(r)
    return out


def paired_over_seeds(idx: dict[CellKey, dict],
                      a: dict[str, str], b: dict[str, str],
                      metric: str) -> list[tuple[float, float]]:
    """Return [(metric_a[seed], metric_b[seed]) for each seed ∈ 1..N_SEEDS]."""
    pairs = []
    for seed in range(1, N_SEEDS + 1):
        ka = (a["encoder"], a["update"], a["schedule"], seed)
        kb = (b["encoder"], b["update"], b["schedule"], seed)
        ra, rb = idx.get(ka), idx.get(kb)
        if ra and rb and ra.get(metric) is not None and rb.get(metric) is not None:
            pairs.append((ra[metric], rb[metric]))
    return pairs


# -----------------------------------------------------------------------------
# Hypothesis tests
# -----------------------------------------------------------------------------

def h1_encoder(idx: dict[CellKey, dict]) -> dict[str, Any]:
    """PL > {PB, BL} on BioRED ex-NEG at matched config (anchor: FT × T2)."""
    anchor = {"update": "FT", "schedule": "T2"}
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
    if pvals_t:
        q_t = bh_fdr(pvals_t); q_w = bh_fdr(pvals_w)
        for lab, qt, qw in zip(labels, q_t, q_w):
            for t in tests:
                if t.get("contrast") == lab:
                    t["q_t"] = qt; t["q_w"] = qw
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
    """T1_flat vs T1_biored_only on BC5CDR DD at PB × FT."""
    base = {"encoder": "PB", "update": "FT"}
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
    """T1→T2 vs T1_flat on BioRED ex-NEG and BC5CDR DD at PB/BL/PL × FT."""
    tests, pvals_t, pvals_w, labels = [], [], [], []
    for enc in ENCODERS_MAIN:
        for metric, label in [(BIORED_EX_NEG, "biored_ex_neg"), (BC5CDR_DD, "bc5cdr_dd")]:
            pairs = paired_over_seeds(idx,
                {"encoder": enc, "update": "FT", "schedule": "T2"},
                {"encoder": enc, "update": "FT", "schedule": "T1F"},
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
    """H4 is a methodological null after B.24; do not run FT-vs-LoRA tests."""
    return {
        "hypothesis": "H4",
        "status": "methodological_null",
        "verdict": "empirically_undefined_lora_collapsed",
        "reason": (
            "LoRA arm dropped per Appendix B.24 after three attempts "
            "(LR=2e-5/2048, LR=3e-4/2048, LR=2e-5/4096) all collapsed "
            "to bit-identical 100%-NEGATIVE dev predictions. A collapsed "
            "LoRA comparator is not a fair FT-vs-LoRA test."
        ),
        "tests": [],
    }


def h5_architecture_deferred() -> dict[str, Any]:
    """H5 deferred — shared_multitask architecture dropped from factorial.

    See paper_development_design.md Appendix B row dated 2026-04-16.  The
    architecture axis was removed pre-Phase-B because the Phase B trainer did
    not implement the shared_multitask head and an honest implementation was
    estimated at 2–3 days of engineering for a hypothesis not central to the
    paper's RQ4 headline claim.  H5 is intentionally recorded here so that
    the analysis output documents the deferral rather than silently omitting
    the hypothesis.
    """
    return {
        "hypothesis": "H5",
        "status": "deferred_to_future_work",
        "reason": ("shared_multitask architecture dropped from Phase B "
                   "factorial; Appendix B amendment 2026-04-16."),
        "verdict": "deferred",
    }


def h7_variance_asymmetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Variance decomposition: R_B = (design-lever share in BioRED ex-NEG) /
    (design-lever share in KB_hit_A).  §9.5.

    Under the realised post-B.24 factorial, design levers are
    (encoder + schedule + their interaction); architecture and update regime
    no longer contribute.  Threshold R_B ≥ 2 is retained unchanged because
    the threshold is first-principles (R = 1 = perfect benchmark-KB proxy,
    R ≥ 2 = ≥ 2× disparity) and axis-count independent.
    """
    main_rows = [r for r in rows if r.get("encoder") in ENCODERS_MAIN]
    if len(main_rows) < N_CELLS_MAIN * 2:  # at least 2 seeds per cell for any decomp
        return {"hypothesis": "H7", "status": "phase_b_data_not_yet_available",
                "n_main_rows": len(main_rows),
                "expected_min": N_CELLS_MAIN * 2}
    def _ss_decomp(metric: str) -> dict[str, float]:
        ys = [r[metric] for r in main_rows if r.get(metric) is not None]
        if len(ys) < 2:
            return {}
        grand = statistics.mean(ys)
        ss_tot = sum((y - grand) ** 2 for y in ys)
        shares = {}
        for factor in PHASE_B_FACTORS:
            by_level = defaultdict(list)
            for r in main_rows:
                if r.get(metric) is not None:
                    by_level[r[factor]].append(r[metric])
            ss_f = sum(len(vs) * (statistics.mean(vs) - grand) ** 2
                       for vs in by_level.values() if vs)
            shares[factor] = ss_f / ss_tot if ss_tot else 0.0
        for fa, fb in itertools.combinations(PHASE_B_FACTORS, 2):
            by_cell = defaultdict(list)
            for r in main_rows:
                if r.get(metric) is not None:
                    by_cell[(r[fa], r[fb])].append(r[metric])
            ss_ab = sum(len(vs) * (statistics.mean(vs) - grand) ** 2
                        for vs in by_cell.values() if vs)
            shares[f"{fa}_x_{fb}"] = max(0.0, ss_ab / ss_tot - shares[fa] - shares[fb])
        return shares
    decomp = {m: _ss_decomp(m) for m in (BIORED_EX_NEG, KB_HIT_A)}
    if not all(decomp.values()):
        return {"hypothesis": "H7", "status": "insufficient_data",
                "decomposition": decomp}
    def _lever_share(d):
        return sum(v for k, v in d.items() if k != "within_cell_residual")
    lever_num = _lever_share(decomp[BIORED_EX_NEG])
    lever_den = _lever_share(decomp[KB_HIT_A])
    R_B = lever_num / max(1e-9, lever_den)
    if R_B >= 2.0:
        verdict = "configuration_induced_asymmetry_confirmed"
    elif R_B > 1.0:
        verdict = "borderline"
    else:
        verdict = "null_no_asymmetry"
    return {"hypothesis": "H7", "R_B": R_B,
            "lever_share_biored_ex_neg": lever_num,
            "lever_share_kb_hit_a": lever_den,
            "decomposition": decomp,
            "threshold": 2.0,
            "threshold_justification": ("first-principles: R = 1 = perfect "
                "benchmark-KB proxy; R >= 2 = >=2x disparity. Axis-count "
                "independent; retained unchanged after arch drop."),
            "verdict": verdict}


# -----------------------------------------------------------------------------
# §8.5 R_B cluster-bootstrap percentile CI
# §8.6 Ordinal-instability quantification
#
# These two modules are the pre-committed Phase B reporting routines whose
# point estimates are emitted by `h7_variance_asymmetry` and the H6 slope
# script respectively. They are kept here so that one invocation of
# `analyze_phase_b.py` produces every Phase B confirmatory number the paper
# needs, including R_B's bootstrap CI and the ordinal-instability headline
# numbers (§8.5, §8.6 of `paper_methods_draft.md`).
#
# Both routines are designed to be agnostic to the specific factor set so
# that the same code can be smoke-tested on Phase A (factors =
# {schema, encoder}, 12 cells × 10 seeds, expected R_A ≈ 2.29) before
# Phase B unblinds.
# -----------------------------------------------------------------------------

# Pre-specified per §8.6: matching radius ρ is pinned to the Phase A
# within-cell BioRED ex-NEG SD; this is constant by design and must not
# be recomputed from Phase B data.
ORDINAL_RHO = 0.03
RB_BOOTSTRAP_SEED = 20260417   # distinct from Phase A's 20260416
ORDINAL_BOOTSTRAP_SEED = 20260418
N_BOOTSTRAP_DEFAULT = 5000

# Canonical factor sets ↔ §8.5 / §6.9.3.
PHASE_B_FACTORS = ("encoder", "schedule")
PHASE_A_FACTORS = ("encoder", "schema")


def _cell_id(row: dict[str, Any], factors: Sequence[str]) -> tuple:
    return tuple(row[f] for f in factors)


def _compute_lever_shares(rows: list[dict[str, Any]],
                          metric: str,
                          factors: Sequence[str]) -> dict[str, float] | None:
    """Type-I-style sequential SS decomposition for one metric.

    Returns the share (fraction of total SS) attributable to each main
    factor and to each two-way interaction, mirroring `_ss_decomp` inside
    `h7_variance_asymmetry` but parametric in `factors` so that it can
    also be applied to Phase A.

    Returns None if `ss_total == 0` (degenerate metric column on this
    sample, e.g. a bootstrap resample where the metric collapses to a
    constant).
    """
    ys = [r[metric] for r in rows if r.get(metric) is not None]
    if len(ys) < 2:
        return None
    grand = statistics.mean(ys)
    ss_tot = sum((y - grand) ** 2 for y in ys)
    if ss_tot <= 0:
        return None

    shares: dict[str, float] = {}

    for f in factors:
        by_level: dict[Any, list[float]] = defaultdict(list)
        for r in rows:
            if r.get(metric) is not None:
                by_level[r[f]].append(r[metric])
        ss_f = sum(len(vs) * (statistics.mean(vs) - grand) ** 2
                   for vs in by_level.values() if vs)
        shares[f] = ss_f / ss_tot

    for fa, fb in itertools.combinations(factors, 2):
        by_cell: dict[tuple, list[float]] = defaultdict(list)
        for r in rows:
            if r.get(metric) is not None:
                by_cell[(r[fa], r[fb])].append(r[metric])
        ss_ab = sum(len(vs) * (statistics.mean(vs) - grand) ** 2
                    for vs in by_cell.values() if vs)
        # Two-way share = SS(cell of {A,B}) − SS(A) − SS(B), clipped at 0.
        shares[f"{fa}_x_{fb}"] = max(0.0, ss_ab / ss_tot - shares[fa] - shares[fb])

    return shares


def _R_from_shares(shares_num: dict[str, float] | None,
                   shares_den: dict[str, float] | None) -> float | None:
    if shares_num is None or shares_den is None:
        return None
    lever_num = sum(shares_num.values())
    lever_den = sum(shares_den.values())
    if lever_den <= 1e-12:
        return None
    return lever_num / lever_den


def bootstrap_RB(per_seed_data: list[dict[str, Any]],
                 *,
                 factors: Sequence[str] = PHASE_B_FACTORS,
                 metric_num: str = BIORED_EX_NEG,
                 metric_den: str = KB_HIT_A,
                 n_resamples: int = N_BOOTSTRAP_DEFAULT,
                 seed: int = RB_BOOTSTRAP_SEED) -> dict[str, Any]:
    """Cell-level cluster-bootstrap percentile CI on R_B (§8.5).

    Parameters
    ----------
    per_seed_data : list of dict
        One dict per seed-level run, with at least the factor columns,
        `metric_num`, and `metric_den`. (For Phase B: factors =
        ('encoder', 'update', 'schedule'); for the Phase A smoke test:
        factors = ('encoder', 'schema').)
    factors : sequence of str
        Factor names that jointly identify a cell.
    metric_num, metric_den : str
        Metric column names for numerator (BioRED ex-NEG by default) and
        denominator (KB_hit_A_setvalued by default) variance shares.
    n_resamples : int
        Number of cluster-bootstrap resamples. 5000 per §8.5.
    seed : int
        Deterministic RNG seed.

    Returns
    -------
    dict with:
        point_estimate : R_B from the observed data.
        bootstrap_median : median R_B across successful resamples.
        ci_lower, ci_upper : 2.5th / 97.5th percentile R_B (95 % CI).
        n_resamples : as requested.
        n_cells_used : number of unique cells in the input.
        failed_resamples : count of resamples whose ANOVA was degenerate
            (e.g. ss_total = 0 or denominator share ≈ 0). The percentile
            CI is taken over the successful resamples only.

    Procedure (matches §8.5 verbatim):
        1. Identify unique cell ids.
        2. For each of `n_resamples` resamples:
           a. Sample `n_cells` cell ids with replacement.
           b. Take all seeds within each sampled cell (cluster
              bootstrap; preserves seed structure within cell).
           c. Compute lever shares on numerator and denominator metrics.
           d. R_B = (Σ lever shares on num) / (Σ lever shares on den).
        3. 95 % CI = (2.5th, 97.5th) percentile of successful R_B values.
    """
    cells_to_rows: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in per_seed_data:
        cells_to_rows[_cell_id(r, factors)].append(r)
    cell_ids = list(cells_to_rows.keys())
    n_cells = len(cell_ids)
    if n_cells < 2:
        return {"status": "insufficient_cells", "n_cells_used": n_cells}

    point_num = _compute_lever_shares(per_seed_data, metric_num, factors)
    point_den = _compute_lever_shares(per_seed_data, metric_den, factors)
    point_RB = _R_from_shares(point_num, point_den)

    rng = random.Random(seed)
    bootstrap_RBs: list[float] = []
    failed = 0
    for _ in range(n_resamples):
        sampled = [cell_ids[rng.randrange(n_cells)] for _ in range(n_cells)]
        resample_rows: list[dict[str, Any]] = []
        for cid in sampled:
            resample_rows.extend(cells_to_rows[cid])
        sn = _compute_lever_shares(resample_rows, metric_num, factors)
        sd = _compute_lever_shares(resample_rows, metric_den, factors)
        rb = _R_from_shares(sn, sd)
        if rb is None or not math.isfinite(rb):
            failed += 1
            continue
        bootstrap_RBs.append(rb)

    if not bootstrap_RBs:
        return {"status": "all_resamples_failed",
                "point_estimate": point_RB,
                "n_resamples": n_resamples,
                "failed_resamples": failed,
                "n_cells_used": n_cells}

    bootstrap_RBs.sort()
    n_ok = len(bootstrap_RBs)
    lo = bootstrap_RBs[int(0.025 * n_ok)]
    hi = bootstrap_RBs[min(n_ok - 1, int(0.975 * n_ok))]
    median = bootstrap_RBs[n_ok // 2]

    return {
        "point_estimate": point_RB,
        "bootstrap_median": median,
        "ci_lower": lo,
        "ci_upper": hi,
        "n_resamples": n_resamples,
        "n_successful_resamples": n_ok,
        "failed_resamples": failed,
        "n_cells_used": n_cells,
        "factors": list(factors),
        "metric_num": metric_num,
        "metric_den": metric_den,
        "seed": seed,
        "lever_shares_observed": {
            metric_num: point_num,
            metric_den: point_den,
        },
    }


def ordinal_instability(per_seed_data: list[dict[str, Any]],
                        *,
                        factors: Sequence[str] = PHASE_B_FACTORS,
                        metric_bench: str = BIORED_EX_NEG,
                        metric_kb: str = KB_HIT_A,
                        rho: float = ORDINAL_RHO,
                        n_resamples: int = N_BOOTSTRAP_DEFAULT,
                        seed: int = ORDINAL_BOOTSTRAP_SEED,
                        exclude_RB: bool = True,
                        pairs_csv_path: Path | None = None) -> dict[str, Any]:
    """Ordinal-instability quantification per §8.6.

    Computes:
      - eligible-pair table (|Δbench| < ρ),
      - median |ΔKB| over eligible pairs (point estimate + bootstrap CI),
      - rank-inversion rate (fraction of eligible non-zero pairs where
        sign(Δbench) ≠ sign(ΔKB)) with bootstrap CI.

    The matching radius ρ is fixed at 0.03 (Phase A within-cell BioRED
    ex-NEG SD). This pin is hard-coded to prevent the circular dependency
    flagged by §8.6 (a Phase-B-derived radius would be a function of the
    LoRA-arm gating outcome).

    Bootstrap is cluster bootstrap **by cell, not by pair**: each
    resample draws cells with replacement and re-enumerates eligible
    pairs in the resample, so the CI absorbs the underlying cell-level
    sampling variability.
    """
    cells_to_rows: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in per_seed_data:
        cells_to_rows[_cell_id(r, factors)].append(r)

    if exclude_RB and "encoder" in factors:
        enc_idx = factors.index("encoder")
        cells_to_rows = {cid: rs for cid, rs in cells_to_rows.items()
                         if cid[enc_idx] != ENCODER_REFERENCE}

    cell_ids = list(cells_to_rows.keys())
    if len(cell_ids) < 2:
        return {"status": "insufficient_cells",
                "n_cells_used": len(cell_ids),
                "rho": rho}

    def _cell_means(rows_per_cell: dict[tuple, list[dict[str, Any]]]
                    ) -> dict[tuple, tuple[float, float]]:
        means: dict[tuple, tuple[float, float]] = {}
        for cid, rs in rows_per_cell.items():
            bvs = [r[metric_bench] for r in rs if r.get(metric_bench) is not None]
            kvs = [r[metric_kb] for r in rs if r.get(metric_kb) is not None]
            if bvs and kvs:
                means[cid] = (statistics.mean(bvs), statistics.mean(kvs))
        return means

    def _enumerate_eligible(means: dict[tuple, tuple[float, float]]
                            ) -> list[dict[str, Any]]:
        keys = list(means.keys())
        eligible: list[dict[str, Any]] = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                bi, ki = means[keys[i]]
                bj, kj = means[keys[j]]
                d_bench = bi - bj
                d_kb = ki - kj
                if abs(d_bench) < rho:
                    inv = (d_bench != 0 and d_kb != 0
                           and ((d_bench > 0) != (d_kb > 0)))
                    eligible.append({
                        "cell_i": keys[i],
                        "cell_j": keys[j],
                        "delta_bench": d_bench,
                        "delta_kb_signed": d_kb,
                        "delta_kb_abs": abs(d_kb),
                        "rank_inverted": inv,
                        "rank_tied": (d_bench == 0 or d_kb == 0),
                    })
        return eligible

    def _summary(eligible: list[dict[str, Any]]) -> tuple[float, float, int, int]:
        """(median |ΔKB|, rank-inversion rate, n_eligible, n_inversion_denom)."""
        if not eligible:
            return (float("nan"), float("nan"), 0, 0)
        kb_abs = sorted(p["delta_kb_abs"] for p in eligible)
        median_kb = kb_abs[len(kb_abs) // 2]
        nz = [p for p in eligible if not p["rank_tied"]]
        if nz:
            inv_rate = sum(1 for p in nz if p["rank_inverted"]) / len(nz)
        else:
            inv_rate = float("nan")
        return (median_kb, inv_rate, len(eligible), len(nz))

    observed_means = _cell_means(cells_to_rows)
    observed_eligible = _enumerate_eligible(observed_means)
    median_kb, inv_rate, n_elig, n_nz = _summary(observed_eligible)

    rng = random.Random(seed)
    boot_medians: list[float] = []
    boot_inv_rates: list[float] = []
    failed = 0
    n_cells = len(cell_ids)
    for _ in range(n_resamples):
        sampled_indices = [rng.randrange(n_cells) for _ in range(n_cells)]
        resample_rows_per_cell: dict[tuple, list[dict[str, Any]]] = {}
        for k, idx in enumerate(sampled_indices):
            cid = cell_ids[idx]
            # Each resample slot is a distinct "position" — avoid
            # collapsing duplicate draws. Position-tagged keys preserve
            # multiplicity and produce the self-pairs (Δ = 0) that
            # cluster-bootstrap requires.
            resample_rows_per_cell[(k, cid)] = cells_to_rows[cid]
        means_r = _cell_means(resample_rows_per_cell)
        elig_r = _enumerate_eligible(means_r)
        m, ir, _, n_nz_r = _summary(elig_r)
        if not (math.isfinite(m) and (math.isnan(ir) or math.isfinite(ir))):
            failed += 1
            continue
        boot_medians.append(m)
        if not math.isnan(ir):
            boot_inv_rates.append(ir)

    def _percentile_ci(xs: list[float]) -> list[float]:
        if not xs:
            return [float("nan"), float("nan")]
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        return [xs_sorted[int(0.025 * n)],
                xs_sorted[min(n - 1, int(0.975 * n))]]

    median_ci = _percentile_ci(boot_medians)
    inv_ci = _percentile_ci(boot_inv_rates)

    if pairs_csv_path is not None and observed_eligible:
        pairs_csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        with pairs_csv_path.open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["cell_i", "cell_j", "delta_bench",
                        "delta_kb_signed", "delta_kb_abs",
                        "rank_inverted", "rank_tied"])
            for p in observed_eligible:
                w.writerow([
                    "|".join(str(x) for x in p["cell_i"]),
                    "|".join(str(x) for x in p["cell_j"]),
                    f"{p['delta_bench']:.6f}",
                    f"{p['delta_kb_signed']:.6f}",
                    f"{p['delta_kb_abs']:.6f}",
                    int(p["rank_inverted"]),
                    int(p["rank_tied"]),
                ])

    return {
        "rho": rho,
        "n_cells_used": n_cells,
        "n_eligible_pairs": n_elig,
        "n_eligible_nonzero_pairs": n_nz,
        "median_delta_KB": median_kb,
        "median_delta_KB_ci": median_ci,
        "rank_inversion_rate": inv_rate,
        "rank_inversion_rate_ci": inv_ci,
        "delta_KB_distribution": [p["delta_kb_abs"] for p in observed_eligible],
        "n_resamples": n_resamples,
        "failed_resamples": failed,
        "n_successful_resamples_median": len(boot_medians),
        "n_successful_resamples_inversion": len(boot_inv_rates),
        "factors": list(factors),
        "metric_bench": metric_bench,
        "metric_kb": metric_kb,
        "exclude_RB": exclude_RB,
        "pairs_csv_path": str(pairs_csv_path) if pairs_csv_path else None,
        "seed": seed,
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

_HYPOTHESIS_KEYS = (
    "H1_encoder", "H2_corpus", "H3_schedule",
    "H4_update_regime", "H5_architecture_deferred",
    "H7_variance_asymmetry",
    "h7_R_B_bootstrap",
    "rq4_ordinal_instability",
)


def analyze(csv_path: Path, out_json: Path, out_md: Path) -> dict[str, Any]:
    rows = load_rows(csv_path)
    coverage = expected_row_count(rows)
    idx = index_cells(rows)

    result = {
        "input_csv": str(csv_path),
        "coverage": coverage,
        "factorial": {
            "encoders_main": list(ENCODERS_MAIN),
            "encoder_reference": ENCODER_REFERENCE,
            "updates": list(UPDATES),
            "schedules": list(SCHEDULES),
            "n_seeds_main": N_SEEDS,
            "n_seeds_reference": N_SEEDS_REFERENCE,
            "n_cells_main": N_CELLS_MAIN,
            "arch_axis_note": ("dropped per Appendix B amendment 2026-04-16; "
                               "shared_multitask not implemented in trainer"),
        },
        "H1_encoder": h1_encoder(idx),
        "H2_corpus": h2_corpus(idx),
        "H3_schedule": h3_schedule(idx),
        "H4_update_regime": h4_update(idx),
        "H5_architecture_deferred": h5_architecture_deferred(),
        "H7_variance_asymmetry": h7_variance_asymmetry(rows),
        "h7_R_B_bootstrap": (
            bootstrap_RB([r for r in rows if r.get("encoder") in ENCODERS_MAIN])
            if any(r.get("encoder") in ENCODERS_MAIN for r in rows)
            else {"status": "phase_b_data_not_yet_available"}
        ),
        "rq4_ordinal_instability": (
            ordinal_instability(
                [r for r in rows if r.get("encoder") in ENCODERS_MAIN],
                pairs_csv_path=out_json.parent / "ordinal_instability_pairs.csv",
            )
            if any(r.get("encoder") in ENCODERS_MAIN for r in rows)
            else {"status": "phase_b_data_not_yet_available"}
        ),
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
             f"expected {cov['expected']['total']}.", "",
             f"**Factorial** (post-amendment 2026-04-16): "
             f"{len(res['factorial']['encoders_main'])} encoders × "
             f"{len(res['factorial']['updates'])} update × "
             f"{len(res['factorial']['schedules'])} schedule "
             f"= {res['factorial']['n_cells_main']} cells × "
             f"{res['factorial']['n_seeds_main']} seeds = "
             f"{res['factorial']['n_cells_main'] * res['factorial']['n_seeds_main']} main + "
             f"{res['factorial']['n_seeds_reference']} RB reference.", ""]
    for key in _HYPOTHESIS_KEYS:
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
    for key in _HYPOTHESIS_KEYS:
        v = res[key].get("verdict") or res[key].get("status")
        print(f"  {key}: {v}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
