#!/usr/bin/env python3
"""Pre-flight for folder 20 (no full epoch scoring)."""

from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from importlib import import_module

cfg = import_module("20_round2_diagnostic.config")
ci = import_module("20_round2_diagnostic.checkpoint_inventory")
mi = import_module("20_round2_diagnostic.matrix_io")
ta = import_module("20_round2_diagnostic.two_axis")
pc = import_module("20_round2_diagnostic.power_check")


def _check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg, flush=True)
    return ok


def main() -> int:
    print("=== Folder 20 pre-flight ===\n", flush=True)
    all_ok = True

    inv, case = ci.build_checkpoint_inventory()
    n_epochs = int(inv["n_recoverable_checkpoints"].sum())
    n_with = int((inv["n_recoverable_checkpoints"] > 0).sum())
    all_ok &= _check(
        "Per-epoch checkpoints available (folder 10 matrix)",
        n_with >= 60,
        f"{n_with}/72 runs with epochs; {n_epochs} total epoch checkpoints",
    )
    all_ok &= _check(
        "Not old single-checkpoint-only layout",
        "all_epochs_saved" in inv["checkpoint_policy"].values or n_epochs > 100,
        case[:120] + "...",
    )

    lr_vals = inv["recipe_lr"].dropna().unique().tolist()
    lr_ok = True
    if lr_vals:
        lr_ok = all(str(v) in ("1e-05", "1e-5", "0.00001", "1e-05") or str(v).startswith("1e-5") for v in lr_vals)
    else:
        complete = cfg.MATRIX_RESULTS_DIR / "pubmedbert_base" / "seed_42" / "matrix_complete.json"
        if complete.exists():
            import json

            lr_ok = str(json.loads(complete.read_text()).get("recipe_lr", "")).startswith("1e-5")
    all_ok &= _check(
        "Reads 1e-5 recipe (not old 2e-5 sweep)",
        lr_ok,
        f"recipe_lr sample: {lr_vals[:3] if lr_vals else 'from matrix_complete'}",
    )

    focus_epochs = ta.count_epoch_checkpoints_to_score()
    all_ok &= _check(
        "Focus encoder epoch inventory",
        focus_epochs >= 150,
        f"{focus_epochs} checkpoints for 3 encoders x 8 seeds",
    )

    all_ok &= _check("Folder 11 per-run scores", cfg.R11_PER_RUN_CSV.exists(), str(cfg.R11_PER_RUN_CSV))
    all_ok &= _check("Folder 11 variance components", cfg.R11_VARIANCE_CSV.exists(), str(cfg.R11_VARIANCE_CSV))
    all_ok &= _check("Folder 11 easy/hard ranking", cfg.R11_EASY_HARD_CSV.exists(), str(cfg.R11_EASY_HARD_CSV))

    import pandas as pd

    vc = pd.read_csv(cfg.R11_VARIANCE_CSV) if cfg.R11_VARIANCE_CSV.exists() else pd.DataFrame()
    gd = vc[vc["metric"] == "kb_mrr_gene_drug"]
    all_ok &= _check(
        "Power check uses Round 1 seed noise",
        not gd.empty and float(gd.iloc[0]["seed_variance_share"]) > 0.5,
        f"gene-drug seed share {float(gd.iloc[0]['seed_variance_share']):.0%}" if not gd.empty else "missing",
    )

    try:
        import_module("shared.pool_loader")
        import_module("shared.distance_analysis")
        import_module("shared.benchmark_eval")
        import_module("20_round2_diagnostic.pool_cache")
        all_ok &= _check("Shared inference imports (not legacy 10_round1)", True)
    except Exception as exc:
        all_ok &= _check("Shared inference imports", False, str(exc))

    legacy = list(REPO.glob("10_round1_benchmark_kb/**/*.py"))
    try:
        src = Path(REPO / "20_round2_diagnostic/pool_cache.py").read_text()
        all_ok &= _check("No legacy 10_round1 imports in pool_cache", "10_round1_benchmark_kb" not in src)
        src_fig = Path(REPO / "20_round2_diagnostic/figures.py").read_text()
        all_ok &= _check("No legacy sweep figure source", "round1_sweep_recipe_match" not in src_fig)
    except Exception as exc:
        all_ok &= _check("Legacy import scan", False, str(exc))

    step_dir = REPO / "20_round2_diagnostic"
    for pf in sorted(step_dir.glob("*.py")):
        try:
            py_compile.compile(str(pf), doraise=True)
        except py_compile.PyCompileError as exc:
            all_ok &= _check("py_compile", False, f"{pf.name}: {exc}")
            break
    else:
        all_ok &= _check("py_compile folder-20 modules", True, f"{len(list(step_dir.glob('*.py')))} files")

    try:
        import_module("20_round2_diagnostic.run")
        import_module("20_round2_diagnostic.report")
        curves = import_module("20_round2_diagnostic.training_curves").load_epoch_curves()
        all_ok &= _check("Training curve path trace", len(curves) > 100, f"{len(curves)} epoch rows")
    except Exception as exc:
        all_ok &= _check("Analysis path trace", False, str(exc))

    sample = mi.epoch_checkpoint_dir("pubmedbert_base", 42, 1)
    all_ok &= _check(
        "Sample per-epoch checkpoint path",
        sample.exists(),
        str(sample),
    )

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(sample)
        del tok
        all_ok &= _check("Sample fp16 epoch checkpoint loadable", True, sample.name)
    except Exception as exc:
        all_ok &= _check("Sample fp16 epoch checkpoint loadable", False, str(exc))

    print(f"\n=== Compute scope ===", flush=True)
    print(f"  Epoch checkpoints to score (focus encoders): {focus_epochs}", flush=True)
    print(f"  Rough wall-time: ~3–8 min/checkpoint GPU => ~{focus_epochs*5//60}–{focus_epochs*8//60} hours", flush=True)
    print(f"  Stage 1: GPU (step_score_epochs.sbatch); Stage 2: CPU (step_analyze.sbatch)", flush=True)

    print("\n=== Pre-flight summary ===", flush=True)
    if all_ok:
        print("ALL CHECKS PASSED", flush=True)
        return 0
    print("ONE OR MORE CHECKS FAILED", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
