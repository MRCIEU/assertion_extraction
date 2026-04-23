"""Verify the schema of every existing Phase A `phase_a_eval.json`.

The §7.8 checklist requires that every downstream-consumed field is present
and well-typed so the aggregator and the H6 coupling-slope script never
trip on a silent key rename or type drift.

Required top-level keys:
    run_id, encoder_key, schema_key, schema_id, seed,
    label2id, labels_ordered, biored_test, bc5cdr_test, kb_surface
Required biored_test keys:
    per_label, macro_f1, macro_f1_excluding_negative, n, by_source
Required bc5cdr_test keys:
    per_label, macro_f1, drug_disease_f1, drug_disease_support
Required kb_surface keys:
    schema, n_targets_total, n_targets_evaluable,
    kb_surface_mean, kb_surface_50, kb_nonneg_rate,
    kb_hit_A_setvalued, kb_hit_A_singlelabel,
    kb_pmass_B_setvalued, kb_pmass_B_singlelabel,
    kb_auc_C_setvalued, kb_auc_C_singlelabel, per_family

`eval_version` is OPTIONAL on pre-stamp (legacy v1.0) records but REQUIRED
on records produced after 2026-04-22 (see §7.8 aggregator read gate).

Run directly:
    python3.11 -m fine_tuning_experiments.phase_b.trainer.tests.test_eval_fields
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DATA_ROOT = Path(os.environ.get(
    "PROJECT_1_DATA_ROOT", "/lus/lfs1aip2/projects/b5ac/project_1",
))
RUNS_ROOT = DATA_ROOT / "fine_tuning_experiments" / "runs" / "schema_exp"

TOP_KEYS = {
    "run_id", "encoder_key", "schema_key", "schema_id", "seed",
    "label2id", "labels_ordered", "biored_test", "bc5cdr_test", "kb_surface",
}
BIORED_KEYS = {"per_label", "macro_f1", "macro_f1_excluding_negative", "n", "by_source"}
BC5CDR_KEYS = {"per_label", "macro_f1", "drug_disease_f1", "drug_disease_support"}
KB_KEYS = {
    "schema", "n_targets_total", "n_targets_evaluable",
    "kb_surface_mean", "kb_surface_50", "kb_nonneg_rate",
    "kb_hit_A_setvalued", "kb_hit_A_singlelabel",
    "kb_pmass_B_setvalued", "kb_pmass_B_singlelabel",
    "kb_auc_C_setvalued", "kb_auc_C_singlelabel", "per_family",
}


def check_one(path: Path) -> list[str]:
    errs: list[str] = []
    try:
        d = json.loads(path.read_text())
    except Exception as exc:
        return [f"{path}: JSON parse error: {exc}"]
    missing = TOP_KEYS - set(d)
    if missing:
        errs.append(f"{path.parent.parent.name}: missing top keys {sorted(missing)}")
    for sub, keys in [("biored_test", BIORED_KEYS),
                       ("bc5cdr_test", BC5CDR_KEYS),
                       ("kb_surface", KB_KEYS)]:
        if sub in d:
            m = keys - set(d[sub])
            if m:
                errs.append(f"{path.parent.parent.name}: {sub} missing {sorted(m)}")
    # Type sanity
    if "biored_test" in d and not isinstance(d["biored_test"].get("macro_f1"), (int, float)):
        errs.append(f"{path.parent.parent.name}: biored macro_f1 not numeric")
    kb = d.get("kb_surface") or {}
    if "kb_hit_A_setvalued" in kb and not isinstance(kb["kb_hit_A_setvalued"], (int, float)):
        errs.append(f"{path.parent.parent.name}: kb_hit_A_setvalued not numeric")
    return errs


def run() -> int:
    if not RUNS_ROOT.is_dir():
        print(f"SKIP: RUNS_ROOT not found: {RUNS_ROOT}")
        return 0
    eval_paths = sorted(RUNS_ROOT.glob("PA_*/eval/phase_a_eval.json"))
    if not eval_paths:
        print(f"SKIP: no PA_*/eval/phase_a_eval.json under {RUNS_ROOT}")
        return 0
    all_errs: list[str] = []
    versions_seen: dict[str, int] = {}
    for p in eval_paths:
        all_errs.extend(check_one(p))
        try:
            d = json.loads(p.read_text())
            versions_seen[d.get("eval_version", "<legacy/missing>")] = \
                versions_seen.get(d.get("eval_version", "<legacy/missing>"), 0) + 1
        except Exception:
            pass
    print(f"checked {len(eval_paths)} eval JSONs; eval_version counts: {versions_seen}")
    if all_errs:
        print("\nFAIL:")
        for e in all_errs[:50]:
            print(f"  - {e}")
        if len(all_errs) > 50:
            print(f"  ... and {len(all_errs) - 50} more")
        return 1
    print(f"PASS: all {len(eval_paths)} eval JSONs have required schema.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
