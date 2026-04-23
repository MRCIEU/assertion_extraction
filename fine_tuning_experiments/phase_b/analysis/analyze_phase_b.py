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
# arch axis dropped per Appendix B row 2 (2026-04-16).
UPDATES = ("FT", "LR")           # full-FT, LoRA
SCHEDULES = ("T1B", "T1F", "T2") # T1_biored_only, T1_flat, T1→T2 staged
N_SEEDS = 20
N_SEEDS_REFERENCE = 10
N_CELLS_MAIN = len(ENCODERS_MAIN) * len(UPDATES) * len(SCHEDULES)  # 18
EXPECTED_MAIN = N_CELLS_MAIN * N_SEEDS                              # 360
EXPECTED_REFERENCE = N_SEEDS_REFERENCE                              # 10 (RB × FT × T2 × 10 seeds)
EXPECTED_TOTAL = EXPECTED_MAIN + EXPECTED_REFERENCE                 # 370
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
    """Phase B expects 360 main runs + 10 RB reference runs = 370."""
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
    """Full-FT vs LoRA on BioRED ex-NEG at PB/BL/PL × T2."""
    tests, pvals_t, pvals_w, labels = [], [], [], []
    for enc in ENCODERS_MAIN:
        pairs = paired_over_seeds(idx,
            {"encoder": enc, "update": "FT", "schedule": "T2"},
            {"encoder": enc, "update": "LR", "schedule": "T2"},
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

    Under the amended factorial (Appendix B 2026-04-16), design levers are
    (encoder + update + schedule + all pairwise interactions among these);
    architecture no longer contributes.  Threshold R_B ≥ 2 is retained
    unchanged because the threshold is first-principles (R = 1 = perfect
    benchmark-KB proxy, R ≥ 2 = ≥ 2× disparity) and axis-count independent.
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
        for factor in ("encoder", "update", "schedule"):
            by_level = defaultdict(list)
            for r in main_rows:
                if r.get(metric) is not None:
                    by_level[r[factor]].append(r[metric])
            ss_f = sum(len(vs) * (statistics.mean(vs) - grand) ** 2
                       for vs in by_level.values() if vs)
            shares[factor] = ss_f / ss_tot if ss_tot else 0.0
        for fa, fb in [("encoder", "update"), ("encoder", "schedule"),
                       ("update", "schedule")]:
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
# Driver
# -----------------------------------------------------------------------------

_HYPOTHESIS_KEYS = (
    "H1_encoder", "H2_corpus", "H3_schedule",
    "H4_update_regime", "H5_architecture_deferred",
    "H7_variance_asymmetry",
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
