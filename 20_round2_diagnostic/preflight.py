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
es = import_module("20_round2_diagnostic.epoch_scoring")
mi = import_module("20_round2_diagnostic.matrix_io")


def _check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg, flush=True)
    return ok


def main() -> int:
    print("=== Folder 20 pre-flight (5e-6 per-epoch checkpoints) ===\n", flush=True)
    all_ok = True

    inv, case = ci.build_checkpoint_inventory()
    n_epochs = int(inv["n_recoverable_checkpoints"].sum())
    n_with = int((inv["n_recoverable_checkpoints"] > 0).sum())
    all_ok &= _check(
        "Per-epoch checkpoints (72 runs)",
        n_with == 72,
        f"{n_with}/72 runs with epochs; {n_epochs} total epoch checkpoints",
    )

    lr_vals = inv["recipe_lr"].dropna().unique()
    lr_ok = len(lr_vals) > 0 and all(abs(float(v) - cfg.EXPECTED_RECIPE_LR) < 1e-10 for v in lr_vals)
    all_ok &= _check(
        "Recipe is 5e-6/none (clean matrix)",
        lr_ok,
        f"recipe_lr values: {[f'{float(v):.0e}' for v in lr_vals[:3]]}",
    )

    try:
        from shared.pool_loader import load_primary_candidates

        pool = load_primary_candidates()
        n_cand = len(pool)
        n_abs = pool["pmid"].nunique()
        all_ok &= _check(
            "Frozen CIViC pool loads",
            n_cand > 18000,
            f"{n_cand} candidates, {n_abs} abstracts",
        )
    except Exception as exc:
        all_ok &= _check("Frozen CIViC pool loads", False, str(exc))

    try:
        from shared.benchmark_eval import build_biored_test_examples

        ex = build_biored_test_examples()
        all_ok &= _check("BioRED benchmark test examples", len(ex) > 0, f"{len(ex)} examples")
    except Exception as exc:
        all_ok &= _check("BioRED benchmark test examples", False, str(exc))

    try:
        import_module("shared.inference")
        import_module("shared.distance_analysis")
        import_module("shared.benchmark_eval")
        import_module("20_round2_diagnostic.mundane_explanations")
        import_module("20_round2_diagnostic.encoder_correlation")
        import_module("20_round2_diagnostic.qualitative_errors")
        all_ok &= _check("Shared imports + folder-20 modules", True)
    except Exception as exc:
        all_ok &= _check("Shared imports", False, str(exc))

    step_dir = REPO / "20_round2_diagnostic"
    compile_ok = True
    for pf in sorted(step_dir.glob("*.py")):
        try:
            py_compile.compile(str(pf), doraise=True)
        except py_compile.PyCompileError as exc:
            compile_ok = False
            all_ok &= _check("py_compile", False, f"{pf.name}: {exc}")
            break
    else:
        all_ok &= _check("py_compile folder-20 modules", True, f"{len(list(step_dir.glob('*.py')))} files")

    expected = es.count_expected_epochs()
    scored = es.count_scored_epochs()
    all_ok &= _check(
        "Scoring path trace",
        expected > 400,
        f"expected {expected} epoch scores; on disk {scored} (0 before stage 1)",
    )

    try:
        import_module("20_round2_diagnostic.run")
        import_module("20_round2_diagnostic.report")
        import_module("20_round2_diagnostic.figures")
        all_ok &= _check("Analysis path trace", True)
    except Exception as exc:
        all_ok &= _check("Analysis path trace", False, str(exc))

    sample_mid = "pubmedbert_base"
    meta = mi.load_training_meta(sample_mid, 42)
    epochs = mi.list_recoverable_epochs(sample_mid, 42, meta)
    sample = mi.epoch_checkpoint_dir(sample_mid, 42, epochs[0] if epochs else 1)
    all_ok &= _check("Sample per-epoch checkpoint path", sample.exists(), str(sample))

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(sample)
        del tok
        all_ok &= _check("Sample fp16 epoch checkpoint loadable", True, sample.name)
    except Exception as exc:
        all_ok &= _check("Sample fp16 epoch checkpoint loadable", False, str(exc))

    if cfg.R11_VARIANCE_CSV.exists():
        all_ok &= _check("Folder 11 variance (power context)", True, str(cfg.R11_VARIANCE_CSV))
    else:
        all_ok &= _check("Folder 11 variance (power context)", False, "missing")

    r11_sample = cfg.R11_SCORES_DIR / "pubmedbert_base" / "seed_42.jsonl"
    all_ok &= _check(
        "Folder 11 CIViC scores (best-val jsonl)",
        r11_sample.exists(),
        str(r11_sample),
    )
    all_ok &= _check(
        "Folder 03 pool size by abstract",
        cfg.POOL_SIZE_BY_ABSTRACT_CSV.exists(),
        str(cfg.POOL_SIZE_BY_ABSTRACT_CSV),
    )
    import json

    ep_samples = list(cfg.SCORES_DIR.glob("*/*/epoch_01.json"))
    cross_ok = False
    if ep_samples:
        sample_ep = json.loads(ep_samples[0].read_text(encoding="utf-8"))
        cross_ok = "kb_mrr_gene_disease_hard" in sample_ep
    all_ok &= _check("Cross-subset metrics in epoch scores", cross_ok)

    print(f"\n=== Compute scope ===", flush=True)
    print(f"  Epoch checkpoints to score: {expected}", flush=True)
    print(f"  Encoders: 9 x up to 8 seeds (all with per-epoch saves)", flush=True)
    print(f"  Stage 1: GPU (parallel submit recommended); Stage 2: CPU", flush=True)
    print(f"\n{case}", flush=True)

    print("\n=== Pre-flight summary ===", flush=True)
    if all_ok:
        print("ALL CHECKS PASSED", flush=True)
        return 0
    print("ONE OR MORE CHECKS FAILED", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
