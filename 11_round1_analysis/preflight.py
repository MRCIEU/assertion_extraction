#!/usr/bin/env python3
"""Pre-flight checks for folder 11 (no full KB scoring)."""

from __future__ import annotations

import importlib
import py_compile
import sys
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from importlib import import_module

from shared.constants import TRAIN_SEEDS
from shared.models import MODELS
from shared.pool_loader import load_primary_candidates

from importlib import import_module as im

cfg = im("11_round1_analysis.config")
score_runs = im("11_round1_analysis.score_runs")


def _check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg, flush=True)
    return ok


def main() -> int:
    print("=== Folder 11 pre-flight (no full scoring) ===\n", flush=True)
    all_ok = True

    # 1. Matrix markers + best checkpoints
    n_markers = 0
    n_best = 0
    missing: list[str] = []
    for spec in MODELS:
        for seed in TRAIN_SEEDS:
            marker = score_runs.matrix_marker(spec.model_id, seed)
            best = score_runs.matrix_best_ckpt(spec.model_id, seed)
            if marker.exists():
                n_markers += 1
            else:
                missing.append(f"marker {spec.model_id}/seed_{seed}")
            if best.exists():
                n_best += 1
            else:
                missing.append(f"best ckpt {spec.model_id}/seed_{seed}")
    expected = len(MODELS) * len(TRAIN_SEEDS)
    all_ok &= _check(
        "72 matrix completion markers",
        n_markers == expected,
        f"{n_markers}/{expected}" + (f"; missing e.g. {missing[0]}" if missing else ""),
    )
    all_ok &= _check(
        "72 best checkpoints",
        n_best == expected,
        f"{n_best}/{expected}",
    )

    # 2. Load one checkpoint (CPU smoke test)
    sample = score_runs.matrix_best_ckpt(MODELS[0].model_id, TRAIN_SEEDS[0])
    try:
        tok = AutoTokenizer.from_pretrained(sample)
        mdl = AutoModelForSequenceClassification.from_pretrained(sample)
        n_params = sum(p.numel() for p in mdl.parameters())
        del mdl, tok
        all_ok &= _check("Sample checkpoint loadable", True, f"{sample.name} ({n_params:,} params)")
    except Exception as exc:
        all_ok &= _check("Sample checkpoint loadable", False, str(exc))

    # 3. Frozen pool
    try:
        pool = load_primary_candidates()
        n_abs = pool["pmid"].nunique()
        n_cand = len(pool)
        pair_types = sorted(pool["pair_type"].unique().tolist())
        all_ok &= _check(
            "Frozen CIViC pool loads",
            n_cand > 0 and n_abs > 0,
            f"{n_cand} candidates, {n_abs} abstracts, pair types {pair_types}",
        )
        variant_in_pool = any(str(pt).startswith("variant") for pt in pair_types)
        all_ok &= _check("Variant pairs excluded from pool", not variant_in_pool, f"pair types={pair_types}")
    except Exception as exc:
        all_ok &= _check("Frozen CIViC pool loads", False, str(exc))

    # 4. Step-02 eval targets + step-01 PMID list
    try:
        from _paths import OUTPUT_ROOT

        ranking_targets = OUTPUT_ROOT / "outputs" / "02_evaluation_protocol" / "ranking_targets.csv"
        rt_ok = ranking_targets.exists()
        all_ok &= _check("Step-02 ranking targets", rt_ok, str(ranking_targets))
        from shared.paths import upstream_paths

        excl = upstream_paths()["excluded_pmids_json"]
        all_ok &= _check("Step-01 excluded PMIDs list", excl.exists(), str(excl))
    except Exception as exc:
        all_ok &= _check("Evaluation upstream artifacts", False, str(exc))

    # 5. Shared inference imports
    try:
        import_module("shared.inference")
        import_module("shared.metrics_ranking")
        import_module("shared.metrics_calibration")
        import_module("shared.distance_analysis")
        import_module("shared.pool_stats")
        all_ok &= _check("Shared inference/metrics imports", True)
    except Exception as exc:
        all_ok &= _check("Shared inference/metrics imports", False, str(exc))

    # 6. py_compile folder-11 modules
    step_dir = REPO / "11_round1_analysis"
    py_files = sorted(step_dir.glob("*.py"))
    compile_ok = True
    compile_err = ""
    for pf in py_files:
        try:
            py_compile.compile(str(pf), doraise=True)
        except py_compile.PyCompileError as exc:
            compile_ok = False
            compile_err = f"{pf.name}: {exc}"
            break
    all_ok &= _check("py_compile folder-11 modules", compile_ok, compile_err or f"{len(py_files)} files")

    # 7. Scoring path trace (imports + marker helpers)
    try:
        assert hasattr(score_runs, "is_scored")
        assert hasattr(score_runs, "count_scored_runs")
        n_scored = score_runs.count_scored_runs()
        all_ok &= _check(
            "Scoring path trace",
            True,
            f"markers on disk {n_scored}/{expected} (0 expected before stage 1)",
        )
    except Exception as exc:
        all_ok &= _check("Scoring path trace", False, str(exc))

    # 8. Analysis path trace
    try:
        import_module("11_round1_analysis.run_analysis")
        import_module("11_round1_analysis.build_auxiliary")
        import_module("11_round1_analysis.figures")
        import_module("11_round1_analysis.report")
        all_ok &= _check("Analysis path trace", True)
    except Exception as exc:
        all_ok &= _check("Analysis path trace", False, str(exc))

    print("\n=== Pre-flight summary ===", flush=True)
    if all_ok:
        print("ALL CHECKS PASSED", flush=True)
        return 0
    print("ONE OR MORE CHECKS FAILED — do not submit until fixed", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
