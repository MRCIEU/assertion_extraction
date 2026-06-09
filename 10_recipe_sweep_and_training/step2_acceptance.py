"""Post-training acceptance gate for step-2 matrix (DeBERTa stability is a hard gate)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shared.constants import DEGENERATE_BENCHMARK_F1_MAX, TRAIN_SEEDS
from shared.models import MODELS, MODEL_BY_ID

from .config import (
    MATRIX_CKPT_DIR,
    MATRIX_COMPLETE,
    MATRIX_DATA,
    SWEEP_OUTPUT_DIR,
    SWEEP_REPORT_PATH,
    matrix_result_path,
    matrix_run_root,
    require_chosen_recipe,
)

# Plain-language thresholds (stated in report, not hidden weights).
DEBERTA_COLLAPSE_F1 = 0.05
DEBERTA_SUPPRESSION_GAP = 0.05  # seed mean far below peer encoders at same recipe
DEBERTA_MODEL_ID = "deberta_base"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_marker(model_id: str, seed: int) -> dict | None:
    path = matrix_result_path(model_id, seed)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _footprint_gib(root: Path) -> float:
    if not root.exists():
        return 0.0
    total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    return total / (1024**3)


def _deberta_verdict(deberta_f1s: dict[int, float], peer_means: dict[str, float]) -> tuple[str, list[str]]:
    """Return (verdict_label, reasons). verdict_label is PASS or FAIL."""
    reasons: list[str] = []
    seeds_sorted = sorted(deberta_f1s)
    values = [deberta_f1s[s] for s in seeds_sorted]

    collapsed = [s for s, f1 in deberta_f1s.items() if f1 <= DEBERTA_COLLAPSE_F1]
    if collapsed:
        reasons.append(
            f"DeBERTa collapsed (benchmark F1 <= {DEBERTA_COLLAPSE_F1}) on seeds {collapsed}"
        )

    deb_mean = float(np.mean(values)) if values else 0.0
    peer_mean_of_means = float(np.mean(list(peer_means.values()))) if peer_means else deb_mean
    gap = peer_mean_of_means - deb_mean
    if gap > DEBERTA_SUPPRESSION_GAP and deb_mean < peer_mean_of_means:
        reasons.append(
            f"DeBERTa seed mean {deb_mean:.3f} is {gap:.3f} below peer-encoder mean "
            f"{peer_mean_of_means:.3f} (systematic suppression threshold {DEBERTA_SUPPRESSION_GAP})"
        )

    if reasons:
        return "FAIL", reasons
    return "PASS", [
        f"DeBERTa benchmark F1 on all 8 seeds above {DEBERTA_COLLAPSE_F1} "
        f"(mean {deb_mean:.3f}, peer mean {peer_mean_of_means:.3f})"
    ]


def run_acceptance_gate() -> int:
    recipe = require_chosen_recipe()
    expected = len(MODELS) * len(TRAIN_SEEDS)

    _log("\n=== Step-2 matrix acceptance gate ===")
    _log(f"Expected recipe: lr={recipe.lr}, warmup={recipe.warmup_label}")
    _log(f"Systematic suppression: DeBERTa seed mean > {DEBERTA_SUPPRESSION_GAP} below peer mean, "
         f"or any seed benchmark F1 <= {DEBERTA_COLLAPSE_F1} (collapse).\n")

    # 1. Completion count
    rows: list[dict] = []
    missing: list[str] = []
    for spec in MODELS:
        for seed in TRAIN_SEEDS:
            marker = _load_marker(spec.model_id, seed)
            if marker is None:
                missing.append(f"{spec.model_id}/seed_{seed}")
                continue
            rows.append(
                {
                    "model_id": spec.model_id,
                    "short_name": spec.short_name,
                    "seed": seed,
                    "benchmark_f1": float(marker.get("benchmark_f1", 0)),
                    "best_epoch": int(marker.get("best_epoch_val_f1", 0)),
                    "recipe_lr": float(marker.get("recipe_lr", 0)),
                    "degenerate": bool(marker.get("degenerate", False)),
                }
            )

    n_done = len(rows)
    _log(f"[{'PASS' if n_done == expected else 'FAIL'}] matrix_complete.json: {n_done}/{expected}")
    if missing:
        _log(f"  Missing: {missing[:5]}" + (" ..." if len(missing) > 5 else ""))

    wrong_recipe = [r for r in rows if abs(r["recipe_lr"] - recipe.lr) > 1e-12]
    if wrong_recipe:
        _log(f"[FAIL] Recipe mismatch in {len(wrong_recipe)} markers (expected lr={recipe.lr})")

    # 2. Collapsed runs (all encoders)
    collapsed = [r for r in rows if r["benchmark_f1"] <= DEGENERATE_BENCHMARK_F1_MAX or r["degenerate"]]
    _log(
        f"[{'PASS' if not collapsed else 'FAIL'}] No collapsed runs: "
        f"{len(collapsed)} with F1 <= {DEGENERATE_BENCHMARK_F1_MAX}"
    )
    for r in collapsed:
        _log(f"  collapsed: {r['short_name']} seed={r['seed']} F1={r['benchmark_f1']:.4f}")

    # 3. DeBERTa per-seed (decisive gate)
    deberta_f1s: dict[int, float] = {}
    _log("\n=== DeBERTa benchmark F1 by seed (hard gate) ===")
    for seed in TRAIN_SEEDS:
        m = _load_marker(DEBERTA_MODEL_ID, seed)
        if m is None:
            _log(f"  seed {seed}: MISSING")
            deberta_f1s[seed] = 0.0
        else:
            f1 = float(m.get("benchmark_f1", 0))
            deberta_f1s[seed] = f1
            flag = "COLLAPSE" if f1 <= DEBERTA_COLLAPSE_F1 else "ok"
            _log(f"  seed {seed}: {f1:.4f}  [{flag}]")

    # 4. Per-encoder means and spread
    by_model: dict[str, list[float]] = {}
    for r in rows:
        by_model.setdefault(r["short_name"], []).append(r["benchmark_f1"])
    _log("\n=== Per-encoder mean benchmark F1 (seed 42-49) ===")
    peer_means: dict[str, float] = {}
    means = []
    for spec in MODELS:
        vals = by_model.get(spec.short_name, [])
        if not vals:
            continue
        mu = float(np.mean(vals))
        peer_means[spec.short_name] = mu
        means.append(mu)
        _log(f"  {spec.short_name}: mean={mu:.4f} (n={len(vals)})")
    overall_spread = max(means) - min(means) if means else 0.0
    _log(f"  Overall encoder spread (mean F1 max-min): {overall_spread:.4f}")

    # 5. Checkpoint sample
    _log("\n=== Checkpoint completeness sample ===")
    sample_specs = [MODELS[0], MODEL_BY_ID[DEBERTA_MODEL_ID]]
    for spec in sample_specs:
        for seed in (TRAIN_SEEDS[0], TRAIN_SEEDS[-1]):
            root = matrix_run_root(spec.model_id, seed)
            log_p = root / "training_log.json"
            best_p = root / "best"
            epochs = sorted((root / "epochs").glob("epoch_*")) if (root / "epochs").exists() else []
            _log(
                f"  {spec.model_id}/seed_{seed}: log={log_p.exists()} best={best_p.exists()} "
                f"epochs={len(epochs)}"
            )

    # 6. Loadability
    _log("\n=== Checkpoint loadability ===")
    sample_best = matrix_run_root(MODELS[0].model_id, TRAIN_SEEDS[0]) / "best"
    sample_epoch = matrix_run_root(MODELS[0].model_id, TRAIN_SEEDS[0]) / "epochs"
    epoch_dirs = sorted(sample_epoch.glob("epoch_*")) if sample_epoch.exists() else []
    load_ok = True
    for label, ckpt in [("best fp32", sample_best), ("epoch fp16", epoch_dirs[-1] if epoch_dirs else None)]:
        if ckpt is None or not ckpt.exists():
            _log(f"  [{label}] skip (not found yet)")
            continue
        try:
            tok = AutoTokenizer.from_pretrained(ckpt)
            mdl = AutoModelForSequenceClassification.from_pretrained(ckpt)
            del tok, mdl
            _log(f"  [PASS] {label}: {ckpt}")
        except Exception as exc:
            load_ok = False
            _log(f"  [FAIL] {label}: {exc}")

    # 7. Footprint
    gib = _footprint_gib(MATRIX_DATA)
    _log(f"\n=== Actual matrix footprint ===\n  {MATRIX_DATA}: {gib:.2f} GiB")

    # 8. Step-1 sweep intact
    sweep_ok = (SWEEP_OUTPUT_DIR / "recipe_decision_table.csv").exists() and SWEEP_REPORT_PATH.exists()
    _log(
        f"\n[{'PASS' if sweep_ok else 'FAIL'}] Step-1 sweep outputs intact: "
        f"decision table={ (SWEEP_OUTPUT_DIR / 'recipe_decision_table.csv').exists() }, "
        f"report={SWEEP_REPORT_PATH.exists()}"
    )

    # Verdict
    deb_verdict, deb_reasons = _deberta_verdict(
        deberta_f1s,
        {k: v for k, v in peer_means.items() if k != MODEL_BY_ID[DEBERTA_MODEL_ID].short_name},
    )
    all_complete = n_done == expected and not wrong_recipe
    no_collapse = not collapsed
    gate_pass = all_complete and no_collapse and deb_verdict == "PASS"

    _log("\n=== ACCEPTANCE VERDICT ===")
    for line in deb_reasons:
        _log(f"  {line}")

    if gate_pass:
        _log(
            "\nDeBERTa stable across all 8 seeds (no collapse, no systematic suppression) "
            "-> 5e-6/none confirmed, cleared to proceed to folder 11"
        )
        _log("(You decide whether to proceed; this gate does not auto-start folder 11.)")
        return 0

    _log(
        "\nDeBERTa collapses/suppressed on one or more seeds -> 5e-6/none also unstable; "
        "DO NOT proceed; stop and report, because if even the lowest learning rate cannot "
        "stabilise DeBERTa across seeds, the issue is DeBERTa's cross-seed fragility on this "
        "task, which we must handle deliberately (e.g. treat DeBERTa's seed fragility as a "
        "finding, or handle it separately) rather than by trying another recipe."
    )
    if not all_complete:
        _log(f"  Also: matrix incomplete ({n_done}/{expected}) or recipe mismatch.")
    if not no_collapse:
        _log(f"  Also: {len(collapsed)} collapsed run(s) across encoders.")
    _log("(You decide the fallback; this gate does not auto-rerun step 2.)")
    return 1


if __name__ == "__main__":
    raise SystemExit(run_acceptance_gate())
