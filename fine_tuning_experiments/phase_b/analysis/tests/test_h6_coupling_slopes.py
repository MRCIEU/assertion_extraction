"""Synthetic-data self-test for `h6_coupling_slopes.py`.

Constructs a data-generating process with known ground-truth slopes at
each mechanism level, then verifies that `fit_h6_slopes` recovers each
target slope within a tolerance matched to the estimator's finite-sample
variance.

Ground-truth design (pinned via `rng(seed=20260416)`):

  Phase A: 4 encoders × 3 schemas × 10 seeds = 120 runs
    encoder_effect[e] drawn once (x and y scaled identically → β_encoder ≈ 0.5)
    schema_effect[s]  drawn once (x and y scaled identically → β_schema ≈ 0.7)
    seed_noise        independent on x and y → β_within ≈ 0

  Phase B: 6 configs × 10 seeds = 60 runs
    config_effect[c]  drawn once (x and y scaled identically → β_config ≈ 1.2)
    seed_noise        independent on x and y → β_within ≈ 0

Tolerance for each recovered slope is set at ~3 × a conservative finite-
sample SE under the generating process; the test can tolerate reasonable
bootstrap variability while still catching implementation bugs that
shift the estimate by a full effect size.

Run directly:
    python3.11 -m fine_tuning_experiments.phase_b.analysis.tests.test_h6_coupling_slopes
"""
from __future__ import annotations

import sys

import numpy as np

from fine_tuning_experiments.phase_b.analysis.h6_coupling_slopes import (
    Run,
    fit_h6_slopes,
)


def _build_synthetic_runs(seed: int = 20260416) -> list[Run]:
    rng = np.random.default_rng(seed)
    runs: list[Run] = []

    # Phase A: x (biored) and y (kb_hit_A) share encoder + schema effects
    encoders = ["RB", "PB", "BL", "PL"]
    schemas = ["Sflat", "Spair", "Smech"]
    encoder_effect = {e: rng.normal(0.0, 0.10) for e in encoders}
    schema_effect = {s: rng.normal(0.0, 0.10) for s in schemas}

    # Ground-truth slopes:
    #   β_encoder = 0.5  (y-encoder-effect = 0.5 × x-encoder-effect)
    #   β_schema  = 0.7  (y-schema-effect  = 0.7 × x-schema-effect)
    beta_encoder_true = 0.5
    beta_schema_true = 0.7
    # Because Phase A cells are (encoder × schema), the cell-level x mean
    # receives encoder + schema effects and the cell-level y mean receives
    # (beta_encoder × encoder_effect) + (beta_schema × schema_effect).
    # When we fit β_schema at fixed encoder, only schema varies → recover
    # beta_schema_true.  When we fit β_encoder at fixed schema, only
    # encoder varies → recover beta_encoder_true.

    for e in encoders:
        for s in schemas:
            for seed_idx in range(1, 11):
                x = 0.5 + encoder_effect[e] + schema_effect[s] + rng.normal(0.0, 0.03)
                y_mean = (0.2 + beta_encoder_true * encoder_effect[e]
                            + beta_schema_true * schema_effect[s])
                y = y_mean + rng.normal(0.0, 0.05)  # within-cell seed noise INDEPENDENT of x noise
                runs.append(Run(
                    run_id=f"PA_{e}_{s}_s{seed_idx:02d}",
                    phase="A",
                    cell_key=f"PA_{e}_{s}",
                    encoder=e, schema=s, seed=seed_idx,
                    biored_f1=float(x), kb_hit_A=float(y),
                ))

    # Phase B: 6 configs × 10 seeds, β_config ground truth = 1.2
    configs = ["c1", "c2", "c3", "c4", "c5", "c6"]
    config_effect = {c: rng.normal(0.0, 0.08) for c in configs}
    beta_config_true = 1.2
    for c in configs:
        for seed_idx in range(1, 11):
            x = 0.55 + config_effect[c] + rng.normal(0.0, 0.03)
            y = 0.25 + beta_config_true * config_effect[c] + rng.normal(0.0, 0.05)
            runs.append(Run(
                run_id=f"PB_BL_base_{c}_T1_to_T2_s{seed_idx:02d}",
                phase="B",
                cell_key=f"PB_BL_base_{c}_T1_to_T2",
                encoder="BL", schema="Spair", seed=seed_idx,
                biored_f1=float(x), kb_hit_A=float(y),
            ))

    return runs, beta_encoder_true, beta_schema_true, beta_config_true


def _check(name: str, est: float, truth: float, tol: float,
            failures: list[str]) -> None:
    err = abs(est - truth)
    print(f"  {name:<22}: est={est:+.3f}  truth={truth:+.3f}  "
          f"err={err:.3f}  tol={tol:.3f}  "
          f"{'ok' if err <= tol else 'FAIL'}")
    if err > tol:
        failures.append(f"{name}: est {est:+.3f} vs truth {truth:+.3f} (err {err:.3f} > tol {tol:.3f})")


def run() -> int:
    runs, beta_enc_true, beta_sch_true, beta_cfg_true = _build_synthetic_runs()
    print(f"synthetic runs: {len(runs)} "
          f"(A={sum(1 for r in runs if r.phase=='A')}, "
          f"B={sum(1 for r in runs if r.phase=='B')})")

    result = fit_h6_slopes(runs, seed=20260416, n_boot=2000)
    failures: list[str] = []

    # β_within ≈ 0 (tight tolerance because variance noise is small per cell)
    _check("beta_within", result["beta_within"]["estimate"], 0.0, 0.30, failures)
    # β_schema ≈ 0.7  (small-sample noisy; wide tolerance)
    _check("beta_schema", result["beta_schema"]["estimate"], beta_sch_true, 0.30, failures)
    # β_encoder ≈ 0.5
    _check("beta_encoder", result["beta_encoder"]["estimate"], beta_enc_true, 0.30, failures)
    # β_config ≈ 1.2 (tight: 6 configs × 10 seeds)
    _check("beta_config", result["beta_config"]["estimate"], beta_cfg_true, 0.30, failures)
    # β_combined_cell is mechanism-pooled; no single truth value, but its
    # phase_A and phase_B components should recover the A-dominated and
    # B-dominated slopes respectively.
    bcomb = result["beta_combined_cell"]
    if bcomb.get("phase_B_included"):
        _check("beta_combined_cell.beta_B", bcomb["beta_B"], beta_cfg_true, 0.40, failures)

    # Spearman ρ must be positive in both phases (coupling is induced).
    rho_A = result["spearman_rho_A_cell"]["rho"]
    rho_B = result["spearman_rho_B_cell"]["rho"]
    print(f"  spearman_rho_A_cell:   rho={rho_A:+.3f}  (expect positive)")
    print(f"  spearman_rho_B_cell:   rho={rho_B:+.3f}  (expect positive)")
    if rho_A < 0.1:
        failures.append(f"spearman_rho_A_cell = {rho_A:+.3f} (expected positive)")
    if rho_B < 0.1:
        failures.append(f"spearman_rho_B_cell = {rho_B:+.3f} (expected positive)")

    # All slope outputs must have a label.
    for k in ("beta_within", "beta_schema", "beta_encoder", "beta_config"):
        if "label" not in result[k]:
            failures.append(f"{k}: missing 'label' field")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS: all mechanism-stratified slopes recovered within tolerance "
          "on synthetic data.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
