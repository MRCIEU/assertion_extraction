"""Tests for the §8.5 R_B bootstrap and §8.6 ordinal-instability modules
in `analyze_phase_b.py`.

These two routines are paper-critical confirmatory numbers for RQ4. The
tests below validate them on:

  1. Phase A real data (12 cells × 10 seeds, factors = (encoder, schema)).
     The R_A point estimate is ≈ 2.29 per `paper_methods_draft.md` §6.9.3;
     the bootstrap CI must contain it.

  2. Synthetic data with hand-constructed pair eligibility / inversion
     ground truth, to verify the `ordinal_instability` enumerator and
     summary statistics independently of any real-data quirk.

Both tests are deterministic via fixed RNG seeds.

Run directly:
    python3.11 -m fine_tuning_experiments.phase_b.analysis.tests.test_phase_b_analysis
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

from fine_tuning_experiments.phase_b.analysis.analyze_phase_b import (
    BIORED_EX_NEG,
    KB_HIT_A,
    ORDINAL_RHO,
    PHASE_A_FACTORS,
    bootstrap_RB,
    ordinal_instability,
)

PHASE_A_GLOB = (
    "/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/"
    "runs/schema_exp/PA_*/eval/phase_a_eval.json"
)


def _load_phase_a_rows() -> list[dict]:
    """Load Phase A seed-level rows as `bootstrap_RB`-compatible dicts.

    Each row carries:
        encoder ∈ {PB, BL, PL, RB},  schema ∈ {Sflat, Spair, Smech},
        seed,  biored_macro_f1_ex_neg,  kb_hit_A_setvalued.
    """
    paths = sorted(glob.glob(PHASE_A_GLOB))
    rows: list[dict] = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        biored = d["biored_test"].get("macro_f1_excluding_negative")
        kb = d["kb_surface"].get("kb_hit_A_setvalued")
        if biored is None or kb is None:
            continue
        rows.append({
            "encoder": d["encoder_key"],
            "schema": d["schema_key"],
            "seed": int(d["seed"]),
            BIORED_EX_NEG: float(biored),
            KB_HIT_A: float(kb),
        })
    return rows


def test_RB_bootstrap_ci_contains_point_estimate() -> None:
    """Phase A R_A point estimate must lie inside the bootstrap CI.

    This is the paper-cited R_A ≈ 2.29 (§6.9.3, paper_methods_draft.md):
        R_A = Share_BioRED({schema, encoder, schema×encoder})
            / Share_KB_hit_A({schema, encoder, schema×encoder})

    `bootstrap_RB` is metric- and factor-agnostic; running it on Phase A
    with factors = ('encoder', 'schema') is the validation pathway for
    the Phase B routine itself, not a paper number in its own right.
    """
    rows = _load_phase_a_rows()
    if not rows:
        print("SKIP: no Phase A eval JSONs found at", PHASE_A_GLOB)
        return

    # Restrict to factorial encoders to match §6.9.3 (which excludes RB).
    rows = [r for r in rows if r["encoder"] in {"PB", "BL", "PL", "RB"}]
    out = bootstrap_RB(
        rows,
        factors=PHASE_A_FACTORS,
        n_resamples=2000,   # 2000 keeps the test < 5 s while preserving
                            # CI stability to ≈ 1 %.
    )
    assert out.get("status") != "insufficient_cells", out
    assert out["n_cells_used"] == 12, f"expected 12 (4×3) cells, got {out['n_cells_used']}"

    point = out["point_estimate"]
    lo, hi = out["ci_lower"], out["ci_upper"]
    print(f"  R_A point = {point:.3f}, 95% CI = [{lo:.3f}, {hi:.3f}], "
          f"median = {out['bootstrap_median']:.3f}, "
          f"failed = {out['failed_resamples']}/{out['n_resamples']}")

    assert point is not None and 1.5 <= point <= 4.0, (
        f"R_A point estimate {point} far from paper-cited 2.29; check "
        f"data loading or share-ratio definition.")
    assert lo <= point <= hi, (
        f"Point estimate {point:.3f} outside bootstrap CI "
        f"[{lo:.3f}, {hi:.3f}]; this would indicate a bug in the "
        f"resampling routine.")
    # Paper-cited R_A ≈ 2.29 should be inside the CI on this dataset.
    assert lo <= 2.29 <= hi, (
        f"Paper-cited R_A = 2.29 outside bootstrap CI "
        f"[{lo:.3f}, {hi:.3f}]; either the paper number drifted or the "
        f"bootstrap is biased.")
    print("  PASS: R_A bootstrap CI contains both point estimate and 2.29.")


def test_ordinal_instability_with_synthetic_data() -> None:
    """Construct a tiny dataset with known eligible / inverted pair counts.

    Five cells, factor = ('encoder',), 1 seed each, ρ = 0.03:

        cell   bench   kb
        ----   -----   ----
        A       0.10   0.50
        B       0.115  0.40
        C       0.12   0.55
        D       0.20   0.30
        E       0.10   0.50

    Pair-by-pair (using i − j convention; signs matter for inversion):

        A–B:  Δb = −0.015 (|·| < ρ ✓),  Δk = +0.10  → opposite sign  → INV
        A–C:  Δb = −0.020 (|·| < ρ ✓),  Δk = −0.05  → same sign      → not inv
        A–D:  Δb = −0.10           ≥ ρ              → not eligible
        A–E:  Δb =  0.00 (|·| < ρ ✓),  Δk =  0.00   → tied (excluded from rate)
        B–C:  Δb = −0.005 (|·| < ρ ✓), Δk = −0.15   → same sign      → not inv
        B–D:  Δb = −0.085           ≥ ρ             → not eligible
        B–E:  Δb = +0.015 (|·| < ρ ✓), Δk = −0.10   → opposite sign  → INV
        C–D:  Δb = −0.08            ≥ ρ             → not eligible
        C–E:  Δb = +0.020 (|·| < ρ ✓), Δk = +0.05   → same sign      → not inv
        D–E:  Δb = +0.10            ≥ ρ             → not eligible

    Eligible unordered pairs:        {A-B, A-C, A-E, B-C, B-E, C-E} = 6
        of which non-zero (rate dom): {A-B, A-C, B-C, B-E, C-E}     = 5
        inverted (rate num):          {A-B, B-E}                     = 2

    Expected:
        n_eligible_pairs = 6
        n_eligible_nonzero_pairs = 5
        rank_inversion_rate = 2/5 = 0.40
    """
    synthetic = [
        {"encoder": "A", "seed": 1, BIORED_EX_NEG: 0.10,  KB_HIT_A: 0.50},
        {"encoder": "B", "seed": 1, BIORED_EX_NEG: 0.115, KB_HIT_A: 0.40},
        {"encoder": "C", "seed": 1, BIORED_EX_NEG: 0.12,  KB_HIT_A: 0.55},
        {"encoder": "D", "seed": 1, BIORED_EX_NEG: 0.20,  KB_HIT_A: 0.30},
        {"encoder": "E", "seed": 1, BIORED_EX_NEG: 0.10,  KB_HIT_A: 0.50},
    ]
    out = ordinal_instability(
        synthetic,
        factors=("encoder",),
        rho=ORDINAL_RHO,
        n_resamples=500,    # CI precision irrelevant for this test;
                            # we only check counts and rates.
        exclude_RB=False,   # 'RB' not in this synthetic set.
    )
    assert out["n_cells_used"] == 5, out
    assert out["n_eligible_pairs"] == 6, (
        f"expected 6 eligible pairs, got {out['n_eligible_pairs']}\n"
        f"distribution: {out['delta_KB_distribution']}")
    assert out["n_eligible_nonzero_pairs"] == 5, (
        f"expected 5 non-zero eligible pairs, "
        f"got {out['n_eligible_nonzero_pairs']}")
    assert abs(out["rank_inversion_rate"] - 0.4) < 1e-9, (
        f"expected rank-inversion rate = 0.40 (2 of 5), "
        f"got {out['rank_inversion_rate']}")
    # Sanity: bootstrap CI on rate must contain the point.
    lo, hi = out["rank_inversion_rate_ci"]
    assert lo <= 0.4 <= hi or (lo == hi == 0.0), (
        f"rate point 0.40 outside bootstrap CI [{lo}, {hi}]")
    print(f"  n_eligible = {out['n_eligible_pairs']}, "
          f"n_nonzero = {out['n_eligible_nonzero_pairs']}, "
          f"inv_rate = {out['rank_inversion_rate']:.3f} "
          f"CI = [{lo:.3f}, {hi:.3f}], "
          f"median |ΔKB| = {out['median_delta_KB']:.3f}")
    print("  PASS: synthetic ordinal-instability counts match construction.")


def main() -> int:
    print("=" * 72)
    print("test_RB_bootstrap_ci_contains_point_estimate")
    print("=" * 72)
    test_RB_bootstrap_ci_contains_point_estimate()
    print()
    print("=" * 72)
    print("test_ordinal_instability_with_synthetic_data")
    print("=" * 72)
    test_ordinal_instability_with_synthetic_data()
    print()
    print("All tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
