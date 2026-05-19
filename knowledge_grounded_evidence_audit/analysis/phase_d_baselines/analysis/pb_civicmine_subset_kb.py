#!/usr/bin/env python3.11
"""PB KB argmax accuracy on the CIViCmine strictly-covered subset (Phase 2A).

Reads ``eval/kb_surface_targets.jsonl`` from Phase B run directories under
``--runs-root`` (default Isambard Lustre mirror). Aggregates Method~A
``hit_A_sv`` restricted to targets listed in ``civicmine_baseline_case_c.json``
and ``evaluable==True``.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO_DEFAULT = Path("/home/b5ac/freddieyu.b5ac/project_1")
DEFAULT_RUN_ROOT = Path(
    "/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b"
)


def _subset_mean_from_jsonl(path: Path, wanted: set[str]) -> tuple[float | None, int, list[str]]:
    if not path.is_file():
        return None, 0, [f"missing:{path}"]
    records: dict[str, int] = {}
    issues: list[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        tid = str(r.get("target_id"))
        if tid not in wanted:
            continue
        if not r.get("evaluable"):
            issues.append(f"not_evaluable:{tid}")
            continue
        records[tid] = int(r["hit_A_sv"])
    if len(records) != len(wanted):
        missing = sorted(wanted - set(records))
        issues.append(f"n_found={len(records)} expected={len(wanted)} missing={missing[:6]}...")
    if not records:
        return None, 0, issues
    return sum(records.values()) / len(records), len(records), issues


def aggregate_schedule_over_seeds(
    *,
    runs_root: Path,
    schedule_key: str,
    wanted: set[str],
    n_seeds: int,
) -> dict[str, object]:
    per_seed_acc: list[float] = []
    issues_all: list[str] = []
    for s in range(1, n_seeds + 1):
        rid = f"PB_PB_FT_{schedule_key}_s{s:02d}"
        jp = runs_root / rid / "eval" / "kb_surface_targets.jsonl"
        acc, nf, issues = _subset_mean_from_jsonl(jp, wanted)
        issues_all.extend(issues)
        if acc is None:
            continue
        per_seed_acc.append(acc)
    return {
        "run_prefix": "PB_PB_FT",
        "encoder_schedule": schedule_key,
        "n_targets_subset": len(wanted),
        "n_seeds_found": len(per_seed_acc),
        "subset_kb_hit_mean_seed_mean": statistics.mean(per_seed_acc) if per_seed_acc else None,
        "subset_kb_hit_seed_sd": statistics.pstdev(per_seed_acc) if len(per_seed_acc) > 2 else None,
        "per_seed_subset_accuracy_sample": per_seed_acc[:8],
        "issues_sample": issues_all[:24],
    }


def civicmine_extended162_sensitivities(
    *,
    eval_targets: list[dict[str, object]],
    covered_ids: set[str],
    n_correct_on_covered_strict: int,
) -> dict[str, object]:
    """Imputation sensitivities pushing CIViCmine-equivalent contrasts to Denom=162.

    Covered strict targets use observed CIViCmine-derived hits ``n_correct_on_covered_strict``.
    Targets outside the strict subset have no deterministic CIViCmine prediction; we apply:

    - ``assign_negative``: hypothetical argmax ``__NEGATIVE__`` everywhere outside strict.
    - ``assign_always_wrong``: hit 0 on every uncovered target.
    - ``random_label_expected``: analytic expected hit probability if draws are IID uniform over
       the eight S_pair logits as an uninformative comparator (probability ``|expected_set_sv| / 8``).
    """
    from fine_tuning_experiments.schema_exp.eval.schema_expected_label import schema_expected_label_set

    n162 = len(eval_targets)
    n_unc = n162 - len(covered_ids)
    neg_hits = 0
    rnd_e = 0.0
    for row in eval_targets:
        tid = row["target_id"]
        civic = {
            "expected_pairing_family": row.get("pairing_family"),
            "heuristic_gold_s2_label": row.get("expected_label"),
        }
        exp_set, _ = schema_expected_label_set(civic, "S_pair", "primary", "set_valued")
        if tid in covered_ids:
            continue
        neg_hits += int("__NEGATIVE__" in exp_set)
        rnd_e += len(exp_set) / 8.0
    denom = float(n162)
    acc_wrong = n_correct_on_covered_strict / denom
    acc_neg = (n_correct_on_covered_strict + neg_hits) / denom
    acc_rnd = (n_correct_on_covered_strict + rnd_e) / denom
    return {
        "definition": ("CIViCmine has deterministic argmax predictions only under strict subset; "
                      "outside that subset we summarise three imputation hypotheses."),
        "n_strict_covered_correct": int(n_correct_on_covered_strict),
        "n_uncovered_targets": int(n_unc),
        "accuracy_162_exclude_uncovered_predictions": acc_wrong,
        "accuracy_162_neg_surrogate_on_uncovered": acc_neg,
        "accuracy_162_random_label_expected_iid_uniform8": acc_rnd,
        "hits_from_negative_imputed": int(n_correct_on_covered_strict + neg_hits),
        "expected_hits_from_random_imputed": float(n_correct_on_covered_strict + rnd_e),
    }


def pb_subset_kb_block_from_covered_targets(
    covered_targets: list[dict[str, object]],
    *,
    runs_root: Path,
) -> dict[str, object]:
    """Variant that does not touch disk (used by ``run_civicmine_baseline.py`` pre-write)."""
    wanted = {str(c["target_id"]) for c in covered_targets}
    n_correct = sum(int(c["hit_A_sv_argmax"]) for c in covered_targets)
    kb_path = Path(__file__).resolve().parents[4] / (
        "fine_tuning_experiments/schema_exp/eval/inputs/kb_surface_pairs.jsonl"
    )
    eval_rows = [json.loads(l) for l in kb_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    eval_rows = [r for r in eval_rows if r.get("expected_label") != "VARIANT_GENE"]
    sens = civicmine_extended162_sensitivities(
        eval_targets=eval_rows,
        covered_ids=wanted,
        n_correct_on_covered_strict=n_correct,
    )
    return {
        "runs_root_observed": str(runs_root),
        "methods_note": (
            "Subset means use Method A hit_A_sv on kb_surface_targets.jsonl aggregated over the "
            "strict CIViCmine-covered target identifiers (deterministic PMID + tuple match)."
        ),
        "PB_T2_on_strict41": aggregate_schedule_over_seeds(runs_root=runs_root, schedule_key="T2", wanted=wanted, n_seeds=20),
        "PB_T1F_2048_on_strict41": aggregate_schedule_over_seeds(runs_root=runs_root, schedule_key="T1F", wanted=wanted, n_seeds=20),
        "PB_T1B_on_strict41": aggregate_schedule_over_seeds(runs_root=runs_root, schedule_key="T1B", wanted=wanted, n_seeds=20),
        "PB_T1F_4096_on_strict41_interim_placeholder": aggregate_schedule_over_seeds(
            runs_root=runs_root,
            schedule_key="T1F4096",
            wanted=wanted,
            n_seeds=20,
        ),
        "civicmine_162_denominator_heuristics_for_external_system_reporting": sens,
    }


def pb_subset_kb_block(*, civicmine_json_path: Path, runs_root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    data = json.loads(civicmine_json_path.read_text(encoding="utf-8"))
    cov = data["covered_targets"]
    blk = pb_subset_kb_block_from_covered_targets(cov, runs_root=runs_root)
    kb_path = Path(__file__).resolve().parents[4] / (
        "fine_tuning_experiments/schema_exp/eval/inputs/kb_surface_pairs.jsonl")
    eval_rows = [json.loads(l) for l in kb_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    eval_rows = [r for r in eval_rows if r.get("expected_label") != "VARIANT_GENE"]
    return blk, eval_rows
