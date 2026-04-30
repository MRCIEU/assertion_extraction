"""h6_coupling_slopes — mechanism-stratified coupling-slope fit for H6.

Implements exactly the §9.4 specification of `paper_development_design.md`:
five mechanism-stratified slopes between a benchmark metric (BioRED macro-F1
ex-NEG) and the primary KB-audit metric (`KB_hit_A_setvalued`):

    β_within          — seed-level within-cell OLS, cell-count-weighted average
    β_schema          — between-schema slope at fixed encoder (Phase A)
    β_encoder         — between-encoder slope at fixed schema (Phase A)
    β_config          — between-config slope under S_pair (Phase B only)
    β_combined_cell   — pooled between-cell slope with phase dummy

Also reports:
  - phase-interaction test on β_combined_cell (H0: β_A = β_B)
  - Phase A-only and Phase B-only Spearman ρ (Fisher-z 95 % CIs)

All confidence intervals use cluster-bootstrap over whole cells (5,000
resamples by default) except β_config / β_combined_cell, which also
report Wald CIs for comparison.  Determinism: all randomness flows from
`numpy.random.default_rng(--seed)` which defaults to 20260416.

**Phase A arm usability.**  The script can be run on Phase A data alone
(120 runs, 12 Phase A cells).  In that mode β_config returns None and
β_combined_cell reports an A-only between-cell slope with no phase dummy.
After Phase B data are produced, re-run with the combined 480-run input.

CLI:
    python3.11 -m fine_tuning_experiments.phase_b.analysis.h6_coupling_slopes \
        --input fine_tuning_experiments/schema_exp/phase_a_results.csv \
        --out results/phase_a_h6_postlock.json

    python3.11 -m fine_tuning_experiments.phase_b.analysis.h6_coupling_slopes \
        --input-dirs $PROJECT_1_DATA_ROOT/fine_tuning_experiments/runs/schema_exp \
                     $PROJECT_1_DATA_ROOT/fine_tuning_experiments/runs/phase_b \
        --out results/h6_combined.json

Unit test (synthetic data with known true slopes):
    python3.11 -m fine_tuning_experiments.phase_b.analysis.tests.test_h6_coupling_slopes
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Input normalisation
# ─────────────────────────────────────────────────────────────────────

_PHASE_A_RE = re.compile(r"^PA_([A-Z]+)_([A-Za-z]+)_s(\d+)$")
# Post-lock amendment 2026-04-16 (Appendix B row 2): arch axis dropped.
# Phase B run_id layout is now PB_{enc}_{upd}_{sched}_s{NN} (3 factor tokens).
_PHASE_B_RE = re.compile(r"^PB_([A-Z]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)_s(\d+)$")
PHASE_B_MAIN_ENCODERS = {"PB", "BL", "PL"}


@dataclass
class Run:
    """One trained-model observation (seed-level).  The `cell_key` is the
    coarsest grouping that still preserves within-cell seed variance;
    matching 10 seeds/cell for Phase A and 20 seeds/cell for Phase B
    (see Appendix B amendment 2026-04-16; arch axis dropped, seeds
    doubled to preserve total compute)."""
    run_id: str
    phase: str           # "A" or "B"
    cell_key: str        # e.g. "PA_BL_Sflat"  or  "PB_BL_base_standard_T1_to_T2_Spair"
    encoder: str
    schema: str
    seed: int
    biored_f1: float     # x  (benchmark side)
    kb_hit_A: float      # y  (primary KB side)
    kb_pmass_B: float | None = None
    kb_auc_C: float | None = None


def _parse_phase_a_run(run_id: str) -> tuple[str, str, int] | None:
    m = _PHASE_A_RE.match(run_id)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def _parse_phase_b_run(run_id: str) -> tuple[str, str, str, int] | None:
    """Parse Phase B run_id of form `PB_{encoder}_{update}_{schedule}_s{NN}`.

    Returns (encoder, update, schedule, seed) or None.  The arch axis was
    dropped per Appendix B amendment 2026-04-16.
    """
    m = _PHASE_B_RE.match(run_id)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), int(m.group(4))


def runs_from_eval_dir(eval_dir: Path) -> list[Run]:
    """Scan a runs directory (Phase A or B) for `*/eval/phase_a_eval.json`
    files and build Run records."""
    out: list[Run] = []
    for ev in sorted(eval_dir.glob("*/eval/phase_a_eval.json")):
        try:
            d = json.loads(ev.read_text())
        except Exception as exc:
            print(f"  skip {ev}: {exc}", file=sys.stderr)
            continue
        run_id = d.get("run_id") or ev.parent.parent.name
        biored = d.get("biored_test") or {}
        kb = d.get("kb_surface") or {}
        x = biored.get("macro_f1_excluding_negative")
        y = kb.get("kb_hit_A_setvalued")
        if x is None or y is None:
            continue

        pa = _parse_phase_a_run(run_id)
        pb = _parse_phase_b_run(run_id)
        if pa is not None:
            enc, sch, seed = pa
            out.append(Run(
                run_id=run_id, phase="A",
                cell_key=f"PA_{enc}_{sch}",
                encoder=enc, schema=sch, seed=seed,
                biored_f1=float(x), kb_hit_A=float(y),
                kb_pmass_B=_opt_float(kb.get("kb_pmass_B_setvalued")),
                kb_auc_C=_opt_float(kb.get("kb_auc_C_setvalued")),
            ))
        elif pb is not None:
            enc, upd, sched, seed = pb
            if enc not in PHASE_B_MAIN_ENCODERS:
                continue
            # Phase B is S_pair-only; carry schema="Spair" by convention.
            out.append(Run(
                run_id=run_id, phase="B",
                cell_key=f"PB_{enc}_{upd}_{sched}",
                encoder=enc, schema="Spair", seed=seed,
                biored_f1=float(x), kb_hit_A=float(y),
                kb_pmass_B=_opt_float(kb.get("kb_pmass_B_setvalued")),
                kb_auc_C=_opt_float(kb.get("kb_auc_C_setvalued")),
            ))
    return out


def _opt_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f):
            return None
        return f
    except Exception:
        return None


def runs_from_csv(csv_path: Path) -> list[Run]:
    """Load aggregate CSV rows into H6 run records.

    Historically this accepted only the Phase A `aggregate_phase_a.py`
    output.  Phase B now feeds H6 through `phase_b/aggregate_phase_b.py`,
    which emits `PB_{ENC}_{UPD}_{SCHED}_sNN` rows with the same benchmark
    and KB columns.  Support both formats so one H6 invocation can combine
    Phase A eval directories with the Phase B aggregate CSV.
    """
    out: list[Run] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get("run_id", "")
            try:
                x = float(row["biored_macro_f1_ex_neg"])
                y = float(row["kb_hit_A_setvalued"])
            except Exception:
                continue

            pa = _parse_phase_a_run(run_id)
            if pa is not None:
                enc, sch, seed = pa
                out.append(Run(
                    run_id=run_id, phase="A",
                    cell_key=f"PA_{enc}_{sch}",
                    encoder=enc, schema=sch, seed=seed,
                    biored_f1=x, kb_hit_A=y,
                    kb_pmass_B=_opt_float(row.get("kb_pmass_B_setvalued")),
                    kb_auc_C=_opt_float(row.get("kb_auc_C_setvalued")),
                ))
                continue

            pb = _parse_phase_b_run(run_id)
            if pb is not None:
                enc, upd, sched, seed = pb
                out.append(Run(
                    run_id=run_id, phase="B",
                    cell_key=f"PB_{enc}_{upd}_{sched}",
                    encoder=enc, schema="Spair", seed=seed,
                    biored_f1=x, kb_hit_A=y,
                    kb_pmass_B=_opt_float(row.get("kb_pmass_B_setvalued")),
                    kb_auc_C=_opt_float(row.get("kb_auc_C_setvalued")),
                ))
    return out


# ─────────────────────────────────────────────────────────────────────
# OLS primitive (no statsmodels dependency — keeps the module standalone)
# ─────────────────────────────────────────────────────────────────────

def _ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (intercept, slope, residual_se) for y = a + b x using OLS.
    residual_se is the estimated residual standard deviation
    (sqrt(SSR / (n - 2))).  For n < 3 returns (mean(y), 0.0, 0.0)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)
    if n < 2:
        return (float(y.mean()) if n else 0.0, 0.0, 0.0)
    xbar = x.mean(); ybar = y.mean()
    sxx = ((x - xbar) ** 2).sum()
    if sxx <= 0:
        return (float(ybar), 0.0, 0.0)
    sxy = ((x - xbar) * (y - ybar)).sum()
    b = sxy / sxx
    a = ybar - b * xbar
    resid = y - (a + b * x)
    sse = float((resid ** 2).sum())
    dof = max(1, n - 2)
    return a, float(b), math.sqrt(sse / dof)


def _ols_slope_with_wald_ci(
    x: np.ndarray, y: np.ndarray, alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """Return (slope, se_slope, ci_lo, ci_hi) via textbook OLS + normal Wald CI."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)
    _, b, resid_se = _ols_slope(x, y)
    xbar = x.mean()
    sxx = float(((x - xbar) ** 2).sum())
    if n < 3 or sxx <= 0:
        return b, float("inf"), b - 9.99, b + 9.99
    se_b = resid_se / math.sqrt(sxx)
    # 95 % CI — using z critical (1.96); for very small n this is mildly
    # anti-conservative, but we dominate the reporting with bootstrap CIs
    # anyway for small-n slopes (β_schema / β_encoder / β_within per-cell).
    z = 1.959964
    return b, se_b, b - z * se_b, b + z * se_b


def _ols_with_phase_dummy(
    x: np.ndarray, y: np.ndarray, phase_dummy: np.ndarray,
) -> dict[str, float]:
    """Fit y = a + b x + c phase + d (x × phase) and return
    {beta_a, beta_b, beta_interaction, beta_interaction_se,
     beta_interaction_ci_lo, beta_interaction_ci_hi}.

    `phase_dummy` is a binary vector (0 / 1) of the same length as x.
    """
    n = len(x)
    if n < 4 or len(set(phase_dummy.tolist())) < 2:
        return {"beta_A": float("nan"), "beta_B": float("nan"),
                "beta_interaction": float("nan"),
                "beta_interaction_se": float("nan"),
                "beta_interaction_ci_lo": float("nan"),
                "beta_interaction_ci_hi": float("nan"),
                "beta_combined": float("nan")}
    X = np.column_stack([np.ones(n), x, phase_dummy, x * phase_dummy])
    y = np.asarray(y, dtype=float).ravel()
    # Least squares with standard errors
    XtX_inv = np.linalg.pinv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    resid = y - X @ coef
    dof = max(1, n - X.shape[1])
    sigma2 = float((resid ** 2).sum()) / dof
    cov = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov))
    beta_base = float(coef[1])                    # slope in phase 0
    beta_inter = float(coef[3])                   # slope in phase 1 = beta_base + beta_inter
    z = 1.959964
    return {
        "beta_A": beta_base,
        "beta_B": beta_base + beta_inter,
        "beta_interaction": beta_inter,
        "beta_interaction_se": float(se[3]),
        "beta_interaction_ci_lo": beta_inter - z * float(se[3]),
        "beta_interaction_ci_hi": beta_inter + z * float(se[3]),
        "beta_combined": beta_base + 0.5 * beta_inter,  # symmetric pooled
    }


# ─────────────────────────────────────────────────────────────────────
# β_within — per-cell seed-level OLS
# ─────────────────────────────────────────────────────────────────────

def beta_within(
    runs: Sequence[Run], benchmark: str = "biored_f1", kb: str = "kb_hit_A",
    rng: np.random.Generator | None = None, n_boot: int = 5000,
) -> dict[str, Any]:
    """Per-cell OLS slopes; cell-count-weighted average; cluster-bootstrap CI."""
    rng = rng or np.random.default_rng(20260416)

    cells: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        cells[r.cell_key].append(r)

    per_cell: list[dict[str, Any]] = []
    for ck, cell_runs in sorted(cells.items()):
        if len(cell_runs) < 3:
            continue
        x = np.array([getattr(r, benchmark) for r in cell_runs])
        y = np.array([getattr(r, kb) for r in cell_runs])
        _, b, _ = _ols_slope(x, y)
        per_cell.append({"cell": ck, "n": len(cell_runs),
                          "phase": cell_runs[0].phase, "beta": b,
                          "x_sd": float(x.std(ddof=1)),
                          "y_sd": float(y.std(ddof=1))})
    if not per_cell:
        return {"n_cells": 0, "estimate": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "ci_width": float("inf"),
                "per_cell": []}
    # weighted mean (by n_c)
    ws = np.array([p["n"] for p in per_cell], dtype=float)
    bs = np.array([p["beta"] for p in per_cell])
    w_mean = float((ws * bs).sum() / ws.sum())
    # Cluster bootstrap over whole cells
    boot_means: list[float] = []
    cell_ids = np.array([p["cell"] for p in per_cell])
    for _ in range(n_boot):
        idx = rng.integers(0, len(per_cell), size=len(per_cell))
        bws = ws[idx]; bbs = bs[idx]
        boot_means.append(float((bws * bbs).sum() / bws.sum()))
    boot_means.sort()
    lo = boot_means[int(0.025 * n_boot)]
    hi = boot_means[int(0.975 * n_boot)]
    return {
        "n_cells": len(per_cell),
        "estimate": w_mean,
        "ci_lo": float(lo), "ci_hi": float(hi),
        "ci_width": float(hi - lo),
        "per_cell": per_cell,
        "method": "cell-count-weighted mean; cluster bootstrap (5000)",
    }


# ─────────────────────────────────────────────────────────────────────
# β_schema — Phase A between-schema at fixed encoder
# β_encoder — Phase A between-encoder at fixed schema
# ─────────────────────────────────────────────────────────────────────

def _cell_means_A(runs: Sequence[Run]) -> dict[tuple[str, str], tuple[float, float, int]]:
    agg: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for r in runs:
        if r.phase != "A":
            continue
        agg[(r.encoder, r.schema)].append(r)
    out: dict[tuple[str, str], tuple[float, float, int]] = {}
    for k, rs in agg.items():
        x = np.array([r.biored_f1 for r in rs])
        y = np.array([r.kb_hit_A for r in rs])
        out[k] = (float(x.mean()), float(y.mean()), len(rs))
    return out


def beta_schema(
    runs: Sequence[Run], rng: np.random.Generator | None = None,
    n_boot: int = 5000,
) -> dict[str, Any]:
    """Between-schema slope at fixed encoder; inverse-variance-weighted
    pool across encoders; cluster-bootstrap over encoder clusters."""
    rng = rng or np.random.default_rng(20260416)
    cell_means = _cell_means_A(runs)
    encoders = sorted({e for (e, _) in cell_means})
    per_encoder: list[dict[str, Any]] = []
    for e in encoders:
        points = [(cell_means[(e, s)][0], cell_means[(e, s)][1])
                  for s in ("Sflat", "Spair", "Smech") if (e, s) in cell_means]
        if len(points) < 2:
            continue
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        b, se, lo, hi = _ols_slope_with_wald_ci(x, y)
        per_encoder.append({"encoder": e, "n_schemas": len(points),
                             "beta": b, "se": se, "wald_ci": [lo, hi],
                             "x": x.tolist(), "y": y.tolist()})
    if not per_encoder:
        return {"n_encoders": 0, "estimate": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "ci_width": float("inf"), "per_encoder": []}
    # Inverse-variance weighted pool (weights = 1 / se^2; se == inf → weight 0)
    ses = np.array([p["se"] for p in per_encoder])
    bs = np.array([p["beta"] for p in per_encoder])
    w = np.where(np.isfinite(ses) & (ses > 0), 1.0 / (ses ** 2), 0.0)
    if w.sum() > 0:
        pooled = float((w * bs).sum() / w.sum())
    else:
        pooled = float(bs.mean())
    # Cluster-bootstrap over encoders: resample *encoders* with replacement,
    # each carrying its (beta, se) pair; pool as above.
    boot_estimates: list[float] = []
    n = len(per_encoder)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bws = np.where(np.isfinite(ses[idx]) & (ses[idx] > 0),
                        1.0 / (ses[idx] ** 2), 0.0)
        bbs = bs[idx]
        if bws.sum() > 0:
            boot_estimates.append(float((bws * bbs).sum() / bws.sum()))
        else:
            boot_estimates.append(float(bbs.mean()))
    boot_estimates.sort()
    lo = boot_estimates[int(0.025 * n_boot)]
    hi = boot_estimates[int(0.975 * n_boot)]
    return {
        "n_encoders": len(per_encoder),
        "estimate": pooled, "ci_lo": float(lo), "ci_hi": float(hi),
        "ci_width": float(hi - lo),
        "per_encoder": per_encoder,
        "method": "inverse-variance-weighted pool; cluster bootstrap over encoders (5000)",
    }


def beta_encoder(
    runs: Sequence[Run], rng: np.random.Generator | None = None,
    n_boot: int = 5000,
) -> dict[str, Any]:
    """Between-encoder slope at fixed schema; pool across schemas."""
    rng = rng or np.random.default_rng(20260416)
    cell_means = _cell_means_A(runs)
    schemas = sorted({s for (_, s) in cell_means})
    per_schema: list[dict[str, Any]] = []
    for s in schemas:
        points = [(cell_means[(e, s)][0], cell_means[(e, s)][1])
                  for e in ("RB", "PB", "BL", "PL") if (e, s) in cell_means]
        if len(points) < 2:
            continue
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        b, se, lo, hi = _ols_slope_with_wald_ci(x, y)
        per_schema.append({"schema": s, "n_encoders": len(points),
                            "beta": b, "se": se, "wald_ci": [lo, hi],
                            "x": x.tolist(), "y": y.tolist()})
    if not per_schema:
        return {"n_schemas": 0, "estimate": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "ci_width": float("inf"), "per_schema": []}
    ses = np.array([p["se"] for p in per_schema])
    bs = np.array([p["beta"] for p in per_schema])
    w = np.where(np.isfinite(ses) & (ses > 0), 1.0 / (ses ** 2), 0.0)
    pooled = float((w * bs).sum() / w.sum()) if w.sum() > 0 else float(bs.mean())
    boot_estimates: list[float] = []
    n = len(per_schema)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bws = np.where(np.isfinite(ses[idx]) & (ses[idx] > 0),
                        1.0 / (ses[idx] ** 2), 0.0)
        bbs = bs[idx]
        boot_estimates.append(
            float((bws * bbs).sum() / bws.sum()) if bws.sum() > 0 else float(bbs.mean())
        )
    boot_estimates.sort()
    lo = boot_estimates[int(0.025 * n_boot)]
    hi = boot_estimates[int(0.975 * n_boot)]
    return {
        "n_schemas": len(per_schema), "estimate": pooled,
        "ci_lo": float(lo), "ci_hi": float(hi), "ci_width": float(hi - lo),
        "per_schema": per_schema,
        "method": "inverse-variance-weighted pool; cluster bootstrap over schemas (5000)",
    }


# ─────────────────────────────────────────────────────────────────────
# β_config — Phase B between-config slope (S_pair only)
# ─────────────────────────────────────────────────────────────────────

def _cell_means_B(runs: Sequence[Run]) -> dict[str, tuple[float, float, int]]:
    agg: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        if r.phase != "B":
            continue
        if r.encoder not in PHASE_B_MAIN_ENCODERS:
            continue
        agg[r.cell_key].append(r)
    out: dict[str, tuple[float, float, int]] = {}
    for k, rs in agg.items():
        x = np.array([r.biored_f1 for r in rs])
        y = np.array([r.kb_hit_A for r in rs])
        out[k] = (float(x.mean()), float(y.mean()), len(rs))
    return out


def beta_config(
    runs: Sequence[Run], rng: np.random.Generator | None = None,
    n_boot: int = 5000,
) -> dict[str, Any] | None:
    """OLS across 9 realised Phase B FT main cell means (post-B.24).

    RB is a descriptive reference cell and is excluded from H6, matching
    §7.3/§7.4 and the H7 implementation in `analyze_phase_b.py`.
    """
    rng = rng or np.random.default_rng(20260416)
    cell_means = _cell_means_B(runs)
    if not cell_means:
        return None
    points = list(cell_means.items())
    x = np.array([p[1][0] for p in points])
    y = np.array([p[1][1] for p in points])
    b, se, lo, hi = _ols_slope_with_wald_ci(x, y)
    # Cluster bootstrap: resample whole cells (each with its 20-seed
    # composition under the amended factorial; since we operate on cell
    # means the composition affects only the mean's sampling variance,
    # already captured in se).  The cell-resampling bootstrap is the right
    # object for the between-cell slope.
    cell_runs = defaultdict(list)
    for r in runs:
        if r.phase == "B":
            if r.encoder not in PHASE_B_MAIN_ENCODERS:
                continue
            cell_runs[r.cell_key].append(r)
    cell_keys = list(cell_runs)
    boot: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(cell_keys), size=len(cell_keys))
        boot_means: list[tuple[float, float]] = []
        for j in idx:
            rs = cell_runs[cell_keys[j]]
            bx = np.array([r.biored_f1 for r in rs])
            by = np.array([r.kb_hit_A for r in rs])
            boot_means.append((float(bx.mean()), float(by.mean())))
        bx_arr = np.array([p[0] for p in boot_means])
        by_arr = np.array([p[1] for p in boot_means])
        _, b_boot, _ = _ols_slope(bx_arr, by_arr)
        boot.append(b_boot)
    boot.sort()
    boot_lo = boot[int(0.025 * n_boot)]
    boot_hi = boot[int(0.975 * n_boot)]
    return {
        "n_cells": len(points),
        "estimate": float(b), "wald_se": float(se),
        "wald_ci_lo": float(lo), "wald_ci_hi": float(hi),
        "bootstrap_ci_lo": float(boot_lo), "bootstrap_ci_hi": float(boot_hi),
        "ci_lo": float(boot_lo), "ci_hi": float(boot_hi),
        "ci_width": float(boot_hi - boot_lo),
        "method": "OLS on realised Phase B FT main cell means (9 cells post-B.24; RB excluded); Wald + cluster bootstrap (5000)",
    }


# ─────────────────────────────────────────────────────────────────────
# β_combined_cell — pooled across phases with phase dummy
# ─────────────────────────────────────────────────────────────────────

def beta_combined_cell(runs: Sequence[Run]) -> dict[str, Any]:
    cells_A = _cell_means_A(runs)
    cells_B = _cell_means_B(runs)
    xs: list[float] = []; ys: list[float] = []; ph: list[float] = []
    cell_keys: list[str] = []
    for (e, s), (mx, my, _n) in sorted(cells_A.items()):
        xs.append(mx); ys.append(my); ph.append(0.0)
        cell_keys.append(f"PA_{e}_{s}")
    for k, (mx, my, _n) in sorted(cells_B.items()):
        xs.append(mx); ys.append(my); ph.append(1.0)
        cell_keys.append(k)
    if len(xs) < 3:
        return {"n_cells": len(xs), "estimate": float("nan"),
                "phase_A_included": bool(cells_A),
                "phase_B_included": bool(cells_B)}
    x = np.array(xs); y = np.array(ys); p = np.array(ph)
    if len(cells_B) == 0:
        # Phase-A-only fallback: plain OLS, no phase dummy.
        b, se, lo, hi = _ols_slope_with_wald_ci(x, y)
        return {
            "n_cells": len(xs),
            "estimate": float(b),
            "wald_se": float(se),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "ci_width": float(hi - lo),
            "phase_A_included": True, "phase_B_included": False,
            "phase_interaction": None,
            "method": "Phase-A-only OLS (no phase dummy; β_combined_cell "
                      "requires Phase B data for the phase-interaction test)",
        }
    fit = _ols_with_phase_dummy(x, y, p)
    # Decision tree per §9.4 (e):
    inter_lo = fit["beta_interaction_ci_lo"]
    inter_hi = fit["beta_interaction_ci_hi"]
    decision: str
    if inter_lo > 0 or inter_hi < 0:
        decision = "excludes_zero"
    elif abs(inter_lo) < 0.5 and abs(inter_hi) < 0.5:
        decision = "cleanly_includes_zero"
    else:
        decision = "inconclusive"
    return {
        "n_cells": len(xs),
        "estimate": fit["beta_combined"],
        "beta_A": fit["beta_A"],
        "beta_B": fit["beta_B"],
        "phase_interaction": {
            "estimate": fit["beta_interaction"],
            "se": fit["beta_interaction_se"],
            "ci_lo": inter_lo, "ci_hi": inter_hi,
            "decision": decision,
        },
        "ci_lo": None, "ci_hi": None,  # combined CI from interaction model
        "phase_A_included": True, "phase_B_included": True,
        "method": "OLS y = a + b x + c phase + d (x × phase); phase dummy A=0, B=1",
    }


# ─────────────────────────────────────────────────────────────────────
# Spearman ρ with Fisher-z CI  (descriptive, Phase A / Phase B only)
# ─────────────────────────────────────────────────────────────────────

def spearman_rho(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    n = len(x)
    if n < 4:
        return {"n": n, "rho": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan")}
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    rx = _rankdata(x); ry = _rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    z = math.atanh(max(min(rho, 0.999999), -0.999999))
    se = 1.0 / math.sqrt(n - 3)
    lo = math.tanh(z - 1.959964 * se)
    hi = math.tanh(z + 1.959964 * se)
    return {"n": n, "rho": rho, "ci_lo": float(lo), "ci_hi": float(hi)}


def _rankdata(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(v))
    # Average-rank tie handling
    unique, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    avg = np.zeros_like(unique, dtype=float)
    idx = 0
    for u, c in zip(unique, counts):
        avg[np.where(unique == u)[0][0]] = idx + (c - 1) / 2.0
        idx += c
    return avg[inv]


# ─────────────────────────────────────────────────────────────────────
# Three-bin labelling  (§7.2 H6 + §9.3 CI-width gate)
# ─────────────────────────────────────────────────────────────────────

def label_slope(beta: float, ci_lo: float, ci_hi: float,
                 ci_width_threshold: float = 0.30) -> dict[str, Any]:
    w = ci_hi - ci_lo if (ci_hi is not None and ci_lo is not None) else float("inf")
    if not math.isfinite(w) or w > ci_width_threshold:
        label = "inconclusive"
    else:
        a = abs(beta)
        if a < 0.3:
            label = "weak"
        elif a < 1.0:
            label = "moderate"
        else:
            label = "strong"
        label = f"{'positive' if beta >= 0 else 'negative'}-{label}"
    return {"label": label, "ci_width": float(w) if math.isfinite(w) else None}


# ─────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────

def fit_h6_slopes(
    runs: Sequence[Run], seed: int = 20260416, n_boot: int = 5000,
    ci_width_threshold: float = 0.30,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_A = sum(1 for r in runs if r.phase == "A")
    n_B = sum(1 for r in runs if r.phase == "B")
    res: dict[str, Any] = {
        "meta": {
            "n_runs": len(runs), "n_phase_A": n_A, "n_phase_B": n_B,
            "bootstrap_seed": seed, "n_boot": n_boot,
            "ci_width_threshold": ci_width_threshold,
        },
    }
    bw = beta_within(runs, rng=rng, n_boot=n_boot)
    bw["label"] = label_slope(bw["estimate"], bw["ci_lo"], bw["ci_hi"],
                                ci_width_threshold)["label"]
    res["beta_within"] = bw
    bs = beta_schema(runs, rng=rng, n_boot=n_boot)
    bs["label"] = label_slope(bs["estimate"], bs["ci_lo"], bs["ci_hi"],
                                ci_width_threshold)["label"]
    res["beta_schema"] = bs
    be = beta_encoder(runs, rng=rng, n_boot=n_boot)
    be["label"] = label_slope(be["estimate"], be["ci_lo"], be["ci_hi"],
                                ci_width_threshold)["label"]
    res["beta_encoder"] = be
    bc = beta_config(runs, rng=rng, n_boot=n_boot)
    if bc is not None:
        bc["label"] = label_slope(bc["estimate"], bc["ci_lo"], bc["ci_hi"],
                                    ci_width_threshold)["label"]
    res["beta_config"] = bc
    bcomb = beta_combined_cell(runs)
    if bcomb.get("ci_lo") is not None:
        bcomb["label"] = label_slope(bcomb["estimate"], bcomb["ci_lo"],
                                       bcomb["ci_hi"], ci_width_threshold)["label"]
    res["beta_combined_cell"] = bcomb

    # Spearman ρ on cell-level (A) and cell-level (B)
    cells_A = _cell_means_A(runs)
    cells_B = _cell_means_B(runs)
    res["spearman_rho_A_cell"] = spearman_rho(
        [mx for mx, _my, _n in cells_A.values()],
        [my for _mx, my, _n in cells_A.values()],
    )
    res["spearman_rho_B_cell"] = spearman_rho(
        [mx for mx, _my, _n in cells_B.values()],
        [my for _mx, my, _n in cells_B.values()],
    )
    # Seed-level Spearman (descriptive)
    A_x = [r.biored_f1 for r in runs if r.phase == "A"]
    A_y = [r.kb_hit_A for r in runs if r.phase == "A"]
    B_x = [r.biored_f1 for r in runs if r.phase == "B"]
    B_y = [r.kb_hit_A for r in runs if r.phase == "B"]
    res["spearman_rho_A_seed"] = spearman_rho(A_x, A_y)
    res["spearman_rho_B_seed"] = spearman_rho(B_x, B_y)
    return res


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _load_runs(args: argparse.Namespace) -> list[Run]:
    runs: list[Run] = []
    if args.input:
        runs.extend(runs_from_csv(args.input))
    if args.input_dirs:
        for d in args.input_dirs:
            runs.extend(runs_from_eval_dir(d))
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None,
                    help="Aggregate CSV (Phase A or Phase B)")
    ap.add_argument("--input-dirs", type=Path, nargs="*", default=None,
                    help="Directories containing PA_*/PB_*/eval/phase_a_eval.json")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSON path")
    ap.add_argument("--seed", type=int, default=20260416)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--ci-width", type=float, default=0.30)
    args = ap.parse_args()

    runs = _load_runs(args)
    if not runs:
        print("ERROR: no runs loaded; provide --input or --input-dirs",
              file=sys.stderr)
        return 2
    print(f"loaded {len(runs)} runs "
          f"(A={sum(1 for r in runs if r.phase=='A')}, "
          f"B={sum(1 for r in runs if r.phase=='B')})")
    result = fit_h6_slopes(runs, seed=args.seed, n_boot=args.n_boot,
                             ci_width_threshold=args.ci_width)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {args.out}")
    # Condensed human-readable summary
    print("\n== H6 mechanism-stratified slopes ==")
    for name in ("beta_within", "beta_schema", "beta_encoder",
                   "beta_config", "beta_combined_cell"):
        d = result.get(name)
        if d is None:
            print(f"  {name:<22} [None — phase not present in data]")
            continue
        est = d.get("estimate")
        lo = d.get("ci_lo"); hi = d.get("ci_hi")
        lab = d.get("label", "?")
        est_s = f"{est:+.3f}" if isinstance(est, (int, float)) and math.isfinite(est) else str(est)
        lo_s = f"{lo:+.3f}" if isinstance(lo, (int, float)) and math.isfinite(lo) else str(lo)
        hi_s = f"{hi:+.3f}" if isinstance(hi, (int, float)) and math.isfinite(hi) else str(hi)
        print(f"  {name:<22} {est_s}  [{lo_s}, {hi_s}]  label={lab}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
