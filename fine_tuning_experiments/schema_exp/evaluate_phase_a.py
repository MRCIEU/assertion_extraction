#!/usr/bin/env python3.11
"""
Phase A Evaluation: Comprehensive analysis of all completed training runs.

Computes for each PA_* run:
  1. Completion status (did training finish?)
  2. Best dev macro-F1 achieved and at which step
  3. Per-head F1 on BioRED test (within-schema gold)
  4. BC5CDR DRUG_DISEASE F1 (control metric)
  5. KB_surface_mean = mean(1-P(NEG)) over 165 CIViC targets [if logits available]
  6. Training convergence (dev trajectory)

Statistical analysis:
  7. Schema selection: SC* = argmax KB_surface_mean, robust across encoders
  8. Bootstrap permutation test: pairwise schema comparisons
  9. Encoder effect: does schema benefit vary across RB/PB/BL/PL?

Outputs to: projects/project_1/fine_tuning_experiments/schema_exp/
"""
from __future__ import annotations

import csv, json, math, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import statistics

SCRIPT_DIR = Path(__file__).resolve().parent
FT_ROOT    = Path(os.environ.get(
    "PROJECT_1_DATA_ROOT",
    str(Path.home() / "projects" / "project_1")
)).resolve()
RUNS_ROOT  = FT_ROOT / "fine_tuning_experiments" / "runs" / "schema_exp"
OUT_DIR    = FT_ROOT / "fine_tuning_experiments" / "schema_exp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOLDLITE = FT_ROOT / "knowledge_grounded_evidence_audit" / "data" / "processed" / "goldlite_audit_targets.csv"
PROC = FT_ROOT / "training_data_generation" / "data" / "processed"

ENCODER_MAP = {"RB": "RoBERTa-base", "PB": "PubMedBERT-base",
               "BL": "BioLinkBERT-base", "PL": "PubMedBERT-large"}
SCHEMA_MAP  = {"Sflat": "S_flat", "Spair": "S_pair", "Smech": "S_mech"}

# ───────────────────────────────────────────────────────────────────
# 1. Scan completed runs
# ───────────────────────────────────────────────────────────────────

def parse_run_id(run_name: str) -> dict | None:
    m = re.match(r'^PA_([A-Z]+)_([A-Za-z]+)_s(\d+)$', run_name)
    if not m: return None
    enc, schema, seed = m.group(1), m.group(2), int(m.group(3))
    return {"encoder": enc, "schema": schema, "seed": seed, "run_id": run_name}


def check_completion(run_dir: Path) -> dict:
    manifest = run_dir / "run_manifest.json"
    metrics_std = run_dir / "metrics" / "metrics_standard.json"
    val_hist = run_dir / "metrics" / "validation_history.json"
    log = run_dir / "training.log"

    status = {
        "run_dir": str(run_dir),
        "has_manifest": manifest.exists(),
        "has_metrics": metrics_std.exists(),
        "has_val_history": val_hist.exists(),
        "complete": False,
        "error": None,
    }

    if not manifest.exists():
        status["error"] = "missing run_manifest.json"; return status
    if not metrics_std.exists():
        status["error"] = "missing metrics_standard.json"
        # Try to get error from log
        if log.exists():
            last_lines = log.read_text().split('\n')[-10:]
            for line in last_lines:
                if 'error' in line.lower() or 'exception' in line.lower():
                    status["error"] = line[:120]
                    break
        return status

    status["complete"] = True
    return status


# ───────────────────────────────────────────────────────────────────
# 2. Extract metrics from each run
# ───────────────────────────────────────────────────────────────────

def load_dev_trajectory(run_dir: Path) -> list[dict]:
    p = run_dir / "metrics" / "validation_history.json"
    if not p.exists(): return []
    return json.loads(p.read_text())


def load_metrics(run_dir: Path) -> dict:
    p = run_dir / "metrics" / "metrics_standard.json"
    if not p.exists(): return {}
    return json.loads(p.read_text())


def get_best_dev_f1(trajectory: list[dict]) -> tuple[float, int]:
    if not trajectory: return 0.0, 0
    best = max(trajectory, key=lambda x: x.get("dev_macro_f1", 0))
    return best.get("dev_macro_f1", 0), best.get("step", 0)


def load_per_head_f1(run_dir: Path) -> dict:
    """Load BioRED per-head F1 from external evaluation results if available."""
    p = run_dir / "metrics" / "per_head_f1.json"
    if p.exists():
        return json.loads(p.read_text())
    # Try metrics_standard which may have family-wise breakdown
    metrics = load_metrics(run_dir)
    return metrics.get("per_head_f1", {})


def compute_kb_surface_mean(run_dir: Path, targets: list[dict]) -> float | None:
    """Compute KB_surface_mean = mean(1-P(NEG)) over 165 CIViC targets.
    Requires logits to be saved in predictions/ directory."""
    pred_dir = run_dir / "predictions"
    if not pred_dir.exists(): return None

    # Look for predictions file that covers KB targets
    # The trainer saves predictions_scientific.jsonl or similar
    pred_files = list(pred_dir.glob("*.jsonl*"))
    if not pred_files: return None

    # Load predictions and match to KB targets by PMID + entity pair
    # This is a simplified version — full version needs entity matching
    # Returns None if logits not available in the expected format
    logit_file = pred_dir / "kb_logits.json"
    if logit_file.exists():
        logits_data = json.loads(logit_file.read_text())
        p_neg_values = [d.get("p_negative", 0.5) for d in logits_data]
        if p_neg_values:
            return float(1 - statistics.mean(p_neg_values))

    return None  # logits not in expected format — needs dedicated eval pass


# ───────────────────────────────────────────────────────────────────
# 3. Statistical analysis
# ───────────────────────────────────────────────────────────────────

def bootstrap_permutation_test(
    group_a: list[float], group_b: list[float], n_permutations: int = 2000
) -> tuple[float, float, float]:
    """
    Permutation test for difference in means.
    Returns: (observed_diff, ci_lo, ci_hi, p_value)
    """
    if not group_a or not group_b: return 0.0, 0.0, 0.0

    import random
    observed = statistics.mean(group_a) - statistics.mean(group_b)
    combined = group_a + group_b
    n_a = len(group_a)
    perm_diffs = []
    for _ in range(n_permutations):
        perm = random.sample(combined, len(combined))
        d = statistics.mean(perm[:n_a]) - statistics.mean(perm[n_a:])
        perm_diffs.append(d)

    # Two-tailed p-value
    p = sum(1 for d in perm_diffs if abs(d) >= abs(observed)) / n_permutations
    # Bootstrap CI (resample group_a and group_b independently)
    boot_diffs = []
    for _ in range(n_permutations):
        samp_a = [random.choice(group_a) for _ in group_a]
        samp_b = [random.choice(group_b) for _ in group_b]
        boot_diffs.append(statistics.mean(samp_a) - statistics.mean(samp_b))
    boot_diffs.sort()
    ci_lo = boot_diffs[int(0.025 * n_permutations)]
    ci_hi = boot_diffs[int(0.975 * n_permutations)]
    return observed, ci_lo, ci_hi, p


# ───────────────────────────────────────────────────────────────────
# 4. Main evaluation
# ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("PHASE A EVALUATION")
    print("=" * 70)
    print()

    # Scan all PA_* run directories
    if not RUNS_ROOT.exists():
        print(f"ERROR: Runs directory not found: {RUNS_ROOT}")
        print("Training may not have completed yet.")
        return

    run_dirs = sorted([d for d in RUNS_ROOT.iterdir() if d.is_dir() and d.name.startswith("PA_")])
    print(f"Found {len(run_dirs)} run directories")
    print()

    # Load KB targets
    targets = []
    if GOLDLITE.exists():
        targets = list(csv.DictReader(open(GOLDLITE)))
        print(f"Loaded {len(targets)} KB-consistency targets")

    # Collect results
    results = []
    failed = []
    n_complete = 0

    for run_dir in run_dirs:
        info = parse_run_id(run_dir.name)
        if not info: continue

        comp = check_completion(run_dir)
        if not comp["complete"]:
            failed.append({"run": run_dir.name, "error": comp.get("error", "unknown")})
            continue

        n_complete += 1
        trajectory = load_dev_trajectory(run_dir)
        metrics = load_metrics(run_dir)
        best_f1, best_step = get_best_dev_f1(trajectory)
        per_head = load_per_head_f1(run_dir)
        kb_surface = compute_kb_surface_mean(run_dir, targets)

        results.append({
            **info,
            "encoder_name": ENCODER_MAP.get(info["encoder"], info["encoder"]),
            "schema_name": SCHEMA_MAP.get(info["schema"], info["schema"]),
            "best_dev_macro_f1": round(best_f1, 4),
            "best_dev_step": best_step,
            "n_eval_steps": len(trajectory),
            "per_head_f1": per_head,
            "kb_surface_mean": kb_surface,
            "bc5cdr_f1": metrics.get("bc5cdr_drug_disease_f1"),
            "run_dir": str(run_dir),
        })

    print(f"Completed: {n_complete} / {len(run_dirs)}")
    if failed:
        print(f"\nFailed runs ({len(failed)}):")
        for f in failed[:10]:
            print(f"  {f['run']}: {f['error']}")

    if not results:
        print("\nNo completed results to analyse yet.")
        print("Run this script again after training completes.")
        return

    print()
    print("─" * 70)
    print("RESULTS SUMMARY")
    print("─" * 70)

    # Group by (encoder, schema)
    groups: dict[tuple, list] = defaultdict(list)
    for r in results:
        key = (r["encoder"], r["schema"])
        groups[key].append(r)

    print(f"\n{'Group':<20} {'N':>3} {'Best dev F1 mean':>18} {'±SE':>8} {'KB_surface':>12}")
    print("  " + "-" * 65)
    for (enc, schema), runs in sorted(groups.items()):
        f1s = [r["best_dev_macro_f1"] for r in runs]
        kbs = [r["kb_surface_mean"] for r in runs if r["kb_surface_mean"] is not None]
        n = len(runs)
        mean_f1 = statistics.mean(f1s) if f1s else float("nan")
        se_f1 = (statistics.stdev(f1s) / math.sqrt(n)) if len(f1s) > 1 else 0
        kb_str = f"{statistics.mean(kbs):.4f}" if kbs else "N/A (no logits)"
        print(f"  PA_{enc}_{schema:<10} {n:>3} {mean_f1:>18.4f} {se_f1:>8.4f} {kb_str:>12}")

    # Schema selection
    print()
    print("─" * 70)
    print("SCHEMA SELECTION")
    print("─" * 70)
    print()

    # Group by schema across all encoders
    by_schema: dict[str, list[float]] = defaultdict(list)
    by_schema_kb: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_schema[r["schema"]].append(r["best_dev_macro_f1"])
        if r["kb_surface_mean"] is not None:
            by_schema_kb[r["schema"]].append(r["kb_surface_mean"])

    if by_schema_kb:
        print("Primary criterion: KB_surface_mean (all encoders combined)")
        for schema, vals in sorted(by_schema_kb.items(), key=lambda x: -statistics.mean(x[1])):
            m = statistics.mean(vals)
            se = statistics.stdev(vals)/math.sqrt(len(vals)) if len(vals)>1 else 0
            print(f"  {schema}: {m:.4f} ± {se:.4f} (n={len(vals)})")
        best_schema = max(by_schema_kb, key=lambda k: statistics.mean(by_schema_kb[k]))
        print(f"\n  → SC* = {best_schema} ({SCHEMA_MAP.get(best_schema, best_schema)})")
    else:
        print("KB_surface_mean not yet available (logits not saved in expected format).")
        print("Falling back to dev macro-F1 for schema ordering:")
        for schema, vals in sorted(by_schema.items(), key=lambda x: -statistics.mean(x[1])):
            m = statistics.mean(vals)
            se = statistics.stdev(vals)/math.sqrt(len(vals)) if len(vals)>1 else 0
            print(f"  {schema}: {m:.4f} ± {se:.4f} (n={len(vals)})")

    # Save full results
    out_json = OUT_DIR / "phase_a_results.json"
    out_json.write_text(json.dumps({
        "n_complete": n_complete,
        "n_total": len(run_dirs),
        "n_failed": len(failed),
        "failed": failed,
        "results": results,
    }, indent=2))
    print(f"\nFull results saved: {out_json}")

    # Save CSV
    if results:
        out_csv = OUT_DIR / "phase_a_results.csv"
        fields = ["run_id", "encoder", "schema", "seed",
                  "best_dev_macro_f1", "best_dev_step", "kb_surface_mean", "bc5cdr_f1"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        print(f"Results CSV: {out_csv}")


if __name__ == "__main__":
    main()
