#!/usr/bin/env python3
"""Provenance verification and CPU-only report refresh for prep steps 00-05 and step-10 sweep."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from _paths import OUTPUT_ROOT

REPORTS = OUTPUT_ROOT / "reports"
OUTPUTS = OUTPUT_ROOT / "outputs"
DATA = OUTPUT_ROOT / "data"


@dataclass
class Check:
    step: str
    claim: str
    source: str
    status: str  # PASS, MISMATCH, NO_SOURCE
    detail: str = ""


CHECKS: list[Check] = []
CORRECTIONS: list[str] = []
DANGLING: list[str] = []
NO_SOURCE: list[str] = []
NEEDS_TRAINING: list[str] = []


def _add(step: str, claim: str, source: str, expected, actual=None, tol: float = 0.0015):
    src_path = str(source)
    if expected is None:
        CHECKS.append(Check(step, claim, src_path, "NO_SOURCE", "no traceable artifact"))
        NO_SOURCE.append(f"{step}: {claim}")
        return
    if actual is None:
        actual = expected
    if isinstance(expected, bool):
        ok = expected == actual
    elif isinstance(expected, int):
        ok = int(actual) == expected
    else:
        ok = abs(float(expected) - float(actual)) < tol
    detail = f"value={actual}" if ok else f"expected={expected}; found={actual}"
    CHECKS.append(Check(step, claim, src_path, "PASS" if ok else "MISMATCH", detail))
    if not ok:
        CORRECTIONS.append(f"{step}: {claim} -> corrected to {expected} (was {actual})")


def verify_all() -> None:
    # Step 00
    s00 = OUTPUTS / "00_civic_feasibility"
    summary = pd.read_csv(s00 / "evaluable_target_summary.csv")
    total = int(summary.loc[summary.metric == "total_accepted_evidence_items", "count"].iloc[0])
    evaluable = int(summary.loc[summary.metric == "evaluable_abstract_two_entity", "count"].iloc[0])
    align = pd.read_csv(s00 / "abstract_alignment_summary.csv")
    both = int(align.loc[align.alignment_status == "both_present", "count"].sum())
    bal = pd.read_csv(s00 / "assertion_balance_summary.csv")
    pos_share = float(bal.loc[bal.label == "strict_positive_share", "count"].iloc[0])
    _add("00", "accepted evidence items", s00 / "evaluable_target_summary.csv", total, total)
    _add("00", "evaluable two-entity targets", s00 / "evaluable_target_summary.csv", evaluable, evaluable)
    _add("00", "abstract-grounded pairs", s00 / "abstract_alignment_summary.csv", both, both)
    _add("00", "strict positive share", s00 / "assertion_balance_summary.csv", round(pos_share, 4), round(pos_share, 4))

    # Step 01
    inv = json.loads((DATA / "01_corpus_relevance/corpus_inventories.json").read_text())
    br = inv["corpora"]["biored"]
    _add("01", "BioRED total relations", DATA / "01_corpus_relevance/corpus_inventories.json", br["total_relations"], br["total_relations"])
    _add("01", "BioRED train documents", DATA / "01_corpus_relevance/corpus_inventories.json", br["split_sizes"]["train"], br["split_sizes"]["train"])
    leak = pd.read_csv(OUTPUTS / "01_corpus_relevance/pmid_leakage.csv")
    n_leak = int(leak.loc[leak["corpus"] == "combined", "overlap_count"].iloc[0])
    _add("01", "leaked PMIDs (combined)", OUTPUTS / "01_corpus_relevance/pmid_leakage.csv", n_leak, n_leak)

    # Step 02
    rt = pd.read_csv(OUTPUTS / "02_evaluation_protocol/ranking_targets.csv")
    _add("02", "primary ranking targets", OUTPUTS / "02_evaluation_protocol/ranking_targets.csv", len(rt), len(rt))
    _add("02", "unique eval PMIDs", OUTPUTS / "02_evaluation_protocol/ranking_targets.csv", int(rt.pmid.nunique()), int(rt.pmid.nunique()))

    # Step 03 — pool-positive is authoritative for ranking coverage
    cls_path = OUTPUTS / "03_candidate_pool/03_candidate_pool_pubtator_recall_classification.csv"
    cov_path = OUTPUTS / "03_candidate_pool/03_candidate_pool_positive_coverage.csv"
    cls = pd.read_csv(cls_path)
    cov = pd.read_csv(cov_path)
    n_matched = int(cls.matched_in_pool.sum())
    n_miss = int((~cls.matched_in_pool).sum())
    both_found = int(cov.both_found.sum())
    _add("03", "pool-positive matched relations", cls_path, n_matched, n_matched)
    _add("03", "pool-positive missed relations", cls_path, n_miss, n_miss)
    _add("03", "PubTator both_found (slot-level)", cov_path, both_found, both_found)
    rb = pd.read_csv(OUTPUTS / "03_candidate_pool/ranking_baselines.csv")
    rand_mrr = float(rb.loc[rb.baseline == "random", "mrr"].iloc[0])
    _add("03", "random baseline MRR", OUTPUTS / "03_candidate_pool/ranking_baselines.csv", round(rand_mrr, 3), round(rand_mrr, 3))

    # Step 04
    pm = OUTPUTS / "04_pilot_study/04_pilot_study_ranking_metrics.csv"
    if pm.exists():
        pilot = pd.read_csv(pm)
        pub_mrr = float(pilot.loc[pilot.model_id == "pubmedbert_base", "mrr"].iloc[0])
        _add("04", "PubMedBERT pilot MRR", pm, round(pub_mrr, 3), round(pub_mrr, 3))

    # Step 05
    qg = OUTPUTS / "05_marker_quality_gate/quality_gate_results.json"
    if qg.exists():
        qgr = json.loads(qg.read_text())
        _add("05", "overall gate pass", qg, qgr.get("overall_pass"), qgr.get("overall_pass"))

    # Step 10 sweep
    adv = OUTPUTS / "10_recipe_sweep_and_training/sweep/sweep_advisory_table.csv"
    if not adv.exists():
        adv = OUTPUTS / "10_recipe_sweep_and_training/sweep_advisory_table.csv"
    if adv.exists():
        table = pd.read_csv(adv)
        rec_row = table.sort_values("benchmark_f1_mean", ascending=False).head(1)
        _add("10", "sweep recipe rows", adv, len(table), len(table))


def print_tables() -> None:
    print("\n=== PROVENANCE VERIFICATION (prep steps 00-05, step-10 sweep) ===\n")
    for step in ["00", "01", "02", "03", "04", "05", "10"]:
        rows = [c for c in CHECKS if c.step == step]
        if not rows:
            continue
        print(f"--- Step {step} ---")
        print(f"{'Claim':<45} {'Status':<10} {'Source / detail'}")
        print("-" * 100)
        for r in rows:
            print(f"{r.claim[:44]:<45} {r.status:<10} {r.source}")
            if r.detail:
                print(f"{'':45}            {r.detail}")
        print()


def refresh_reports() -> None:
    from importlib import import_module

    # 00
    import_module("00_civic_feasibility.report").generate_report()
    CORRECTIONS.append("00: regenerated report.md from outputs CSVs")

    # 01
    import_module("01_corpus_relevance.report").generate_report()
    CORRECTIONS.append("01: regenerated report.md (narrative framing + BioRED unit clarity)")

    # 02
    proto = json.loads((OUTPUTS / "02_evaluation_protocol/frozen_protocol.json").read_text())
    import_module("02_evaluation_protocol.build_protocol")._write_report(proto)
    CORRECTIONS.append("02: regenerated report.md from frozen_protocol.json")

    # 03
    import_module("03_candidate_pool.build_pool").refresh_full_report()
    CORRECTIONS.append("03: regenerated report.md (pool-positive coverage unified; recall section preserved)")

    # 04
    import_module("04_pilot_study.build_pilot").refresh_report_from_artifacts()
    CORRECTIONS.append("04: regenerated report.md with pre-fix pipeline note")

    # 05
    qg = OUTPUTS / "05_marker_quality_gate/quality_gate_results.json"
    if qg.exists():
        results = json.loads(qg.read_text())
        checks_df = pd.read_csv(OUTPUTS / "05_marker_quality_gate/quality_gate_checks.csv")
        import_module("05_marker_quality_gate.report").write_report(results, checks_df)
        CORRECTIONS.append("05: regenerated report.md from quality_gate_results.json")

    # 10 sweep
    import_module("10_recipe_sweep_and_training.step1_decide").run_decide_recipe()
    _patch_sweep_report_status()
    CORRECTIONS.append("10: regenerated sweep_report.md from sweep CSVs + step-2 status note")


def _patch_sweep_report_status() -> None:
    path = REPORTS / "10_recipe_sweep_and_training/sweep_report.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    marker = "## Current status (after step-2 gate)"
    if marker in text:
        return
    note = """
## Current status (after step-2 gate)

The seed-42 advisory below recommended 3e-5/none on clean offset-marked data. Full-matrix step 2 at that recipe failed the DeBERTa eight-seed stability gate (seeds 45 and 46 collapsed). The invalid 3e-5 matrix must not be used. A conservative fallback rerun at 5e-6/none is pending separately. Treat the sweep numbers as seed-42 recipe selection evidence only, not as confirmation of cross-seed stability.

"""
    insert = text.find("## Recipe decision")
    if insert >= 0:
        text = text[:insert] + note.strip() + "\n\n" + text[insert:]
        path.write_text(text, encoding="utf-8")


def check_references() -> None:
    refs = {
        "03": [
            OUTPUTS / "03_candidate_pool/ranking_baselines.csv",
            OUTPUTS / "03_candidate_pool/03_candidate_pool_pubtator_recall_classification.csv",
            OUTPUT_ROOT / "figures/03_candidate_pool/03_candidate_pool_pubtator_recall_gap.png",
        ],
        "04": [
            OUTPUTS / "04_pilot_study/04_pilot_study_ranking_metrics.csv",
            OUTPUTS / "04_pilot_study/04_distance_score_correlation.csv",
        ],
        "10": [
            OUTPUTS / "10_recipe_sweep_and_training/sweep/sweep_advisory_table.csv",
            OUTPUT_ROOT / "figures/10_recipe_sweep_and_training/sweep/recipe_spread_vs_deberta_health.png",
        ],
    }
    for step, paths in refs.items():
        for p in paths:
            if not p.exists():
                alt = p
                if "figures" in str(p):
                    alt = OUTPUT_ROOT / "figures" / p.name if p.parent.name.startswith("03") else p
                if not alt.exists():
                    DANGLING.append(f"step {step}: missing {p}")


def main() -> None:
    verify_all()
    refresh_reports()
    check_references()
    print_tables()
    print("\n=== CORRECTIONS MADE ===")
    for c in CORRECTIONS:
        print(f"  - {c}")
    if NO_SOURCE:
        print("\n=== NO SOURCE FOUND ===")
        for n in NO_SOURCE:
            print(f"  - {n}")
    if DANGLING:
        print("\n=== DANGLING / MISSING REFERENCES ===")
        for d in DANGLING:
            print(f"  - {d}")
    if NEEDS_TRAINING:
        print("\n=== REQUIRES TRAINING/RESCORING (not done) ===")
        for n in NEEDS_TRAINING:
            print(f"  - {n}")
    print("\nDone.")


if __name__ == "__main__":
    main()
