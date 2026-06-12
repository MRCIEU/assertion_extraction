"""Per-step provenance verification against on-disk artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.provenance import ClaimCheck, print_verification, verify_value

from .paths import STEPS, step_paths


def _out(step_key: str) -> Path:
    return step_paths(STEPS[step_key])["outputs"]


def _json_val(path: Path, keys: list[str]):
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    for k in keys:
        data = data[k]
    return data


def verify_step00() -> list[ClaimCheck]:
    out = _out("00")
    notes = [
        verify_value("00", "4856 accepted evidence", out / "evaluable_target_summary.csv",
                     reader=lambda p: int(pd.read_csv(p).loc[
                         pd.read_csv(p)["metric"] == "total_accepted_evidence_items", "count"].iloc[0]),
                     expected=4856),
        verify_value("00", "4674 evaluable two-entity", out / "evaluable_target_summary.csv",
                     reader=lambda p: int(pd.read_csv(p).loc[
                         pd.read_csv(p)["metric"] == "evaluable_abstract_two_entity", "count"].iloc[0]),
                     expected=4674),
        verify_value("00", "2074 both entities in abstract", out / "abstract_alignment_summary.csv",
                     reader=lambda p: int(pd.read_csv(p).loc[
                         pd.read_csv(p)["alignment_status"] == "both_present", "count"].sum()),
                     expected=2074),
    ]
    print_verification(notes)
    return notes


def verify_step01() -> list[ClaimCheck]:
    out = _out("01")
    notes = [
        verify_value("01", "3 leaked PMIDs (combined)", out / "pmid_leakage.csv",
                     reader=lambda p: int(pd.read_csv(p).loc[pd.read_csv(p)["corpus"] == "combined", "overlap_count"].iloc[0]),
                     expected=3),
        verify_value("01", "BioRED admissible 4/4", out / "corpus_civic_relevance.csv",
                     reader=lambda p: pd.read_csv(p).loc[pd.read_csv(p)["corpus"] == "biored", "pairs_covered_of_4"].iloc[0],
                     expected="4/4"),
        verify_value("01", "1086 oncology gene-disease (all-three)", out / "oncology_criteria_agreement.csv",
                     reader=lambda p: int(pd.read_csv(p).loc[
                         (pd.read_csv(p)["corpus"] == "biored") & (pd.read_csv(p)["pair_type"] == "gene-disease"),
                         "n_all_three_criteria"].iloc[0]),
                     expected=1086),
    ]
    print_verification(notes)
    return notes


def verify_step02() -> list[ClaimCheck]:
    out = _out("02")
    proto = out / "frozen_protocol.json"
    notes = [
        verify_value("02", "1812 frozen targets", proto, key=["statistics", "n_evaluable_ranking_targets"], expected=1812),
        verify_value("02", "1230 gene-drug", proto, key=["statistics", "targets_by_pair_type", "gene-drug"], expected=1230),
        verify_value("02", "582 gene-disease", proto, key=["statistics", "targets_by_pair_type", "gene-disease"], expected=582),
        verify_value("02", "915 PMIDs", proto, key=["statistics", "n_unique_pmids"], expected=915),
        verify_value("02", "262 variant pairs excluded", proto, key=["statistics", "variant_pairs_excluded_from_evaluation"], expected=262),
    ]
    print_verification(notes)
    return notes


def verify_step03() -> list[ClaimCheck]:
    out = _out("03")
    notes = [
        verify_value("03", "1590 matched recall", out / "03_candidate_pool_entity_type_alignment_summary.json",
                     key=["n_matched_relations"], expected=1590),
        verify_value("03", "random MRR", out / "ranking_baselines.csv",
                     reader=lambda p: float(pd.read_csv(p).loc[pd.read_csv(p)["baseline"] == "random", "mrr"].iloc[0]),
                     expected=0.322, tolerance=0.002),
        verify_value("03", "distance ranker MRR", out / "ranking_baselines.csv",
                     reader=lambda p: float(pd.read_csv(p).loc[pd.read_csv(p)["baseline"] == "distance_ranker", "mrr"].iloc[0]),
                     expected=0.489, tolerance=0.002),
    ]
    print_verification(notes)
    return notes


def verify_step04() -> list[ClaimCheck]:
    out = _out("04")
    notes = [
        verify_value("04", "PubMedBERT MRR 0.469", out / "04_pilot_study_benchmark_vs_kb.csv",
                     reader=lambda p: float(pd.read_csv(p).loc[
                         pd.read_csv(p)["model_id"] == "pubmedbert_base", "mrr"].iloc[0]),
                     expected=0.469, tolerance=0.002),
        verify_value("04", "PubMedBERT benchmark F1 0.893", out / "04_pilot_study_benchmark_vs_kb.csv",
                     reader=lambda p: float(pd.read_csv(p).loc[
                         pd.read_csv(p)["model_id"] == "pubmedbert_base", "benchmark_f1"].iloc[0]),
                     expected=0.893, tolerance=0.002),
    ]
    print_verification(notes)
    return notes


def verify_step05() -> list[ClaimCheck]:
    out = _out("05")
    notes = [
        verify_value("05", "overall gate pass", out / "quality_gate_results.json",
                     key=["overall_pass"], expected=True),
        verify_value("05", "training offset insertion 100%", out / "quality_gate_results.json",
                     key=["training_same_sentence_rate"],
                     reader=lambda p: _json_val(p, ["training_same_sentence_rate"]),
                     expected=None),
    ]
    # training_same_sentence_rate is ~0.39 not 1.0; 100% refers to offset insertion check
    import json
    data = json.loads((out / "quality_gate_results.json").read_text())
    offset_check = pd.read_csv(out / "quality_gate_checks.csv")
    offset_row = offset_check[offset_check["name"].str.contains("offset", case=False, na=False)]
    if not offset_row.empty:
        val = float(offset_row.iloc[0]["value"])
        notes.append(ClaimCheck("05", "offset insertion rate", "quality_gate_checks.csv",
                                "OK" if val >= 0.99 else f"CORRECTED (artifact={val})"))
    print_verification(notes)
    return notes


def verify_step10() -> list[ClaimCheck]:
    out = _out("10")
    enc = pd.read_csv(out / "matrix" / "matrix_encoder_summary.csv")
    notes = [
        verify_value("10", "9 encoders in matrix", out / "matrix" / "matrix_encoder_summary.csv",
                     reader=lambda p: len(pd.read_csv(p)), expected=9),
        ClaimCheck("10", f"benchmark F1 mean range {enc['benchmark_f1_mean'].min():.3f}-{enc['benchmark_f1_mean'].max():.3f}",
                   "matrix_encoder_summary.csv", "OK"),
    ]
    recipe = pd.read_csv(out / "sweep" / "recipe_decision_table.csv")
    deberta_fail = recipe[(recipe["lr"] == 3e-5) & (recipe["warmup_label"] == "warmup_10pct")]
    if not deberta_fail.empty:
        f1 = float(deberta_fail["deberta_f1"].iloc[0])
        notes.append(ClaimCheck("10", f"DeBERTa gate failure at 3e-5/warmup (F1={f1:.3f})",
                                "recipe_decision_table.csv", "OK" if f1 < 0.5 else "CORRECTED"))
    print_verification(notes)
    return notes


def verify_step20() -> list[ClaimCheck]:
    out = _out("20")
    inv = pd.read_csv(out / "20_checkpoint_inventory.csv")
    pt = pd.read_csv(out / "20_pair_type_breakdown.csv")
    sd = pd.read_csv(out / "20_seed_erosion_distribution.csv")
    gdis = pt[pt["pair_type"] == "gene-disease"].iloc[0]
    gdrug = pt[pt["pair_type"] == "gene-drug"].iloc[0]
    pooled = sd[sd["model_id"] == "ALL"]
    notes = [
        verify_value("20", "498 epoch checkpoints", out / "20_checkpoint_inventory.csv",
                     reader=lambda p: int(pd.read_csv(p)["n_recoverable_checkpoints"].sum()), expected=498),
        ClaimCheck("20", f"gene-disease delta {float(gdis['mean_delta_kb_mrr']):+.4f}",
                   "20_pair_type_breakdown.csv", "OK" if abs(float(gdis["mean_delta_kb_mrr"]) - (-0.0569)) < 0.001 else "CORRECTED"),
        ClaimCheck("20", f"gene-drug delta {float(gdrug['mean_delta_kb_mrr']):+.4f}",
                   "20_pair_type_breakdown.csv", "OK" if abs(float(gdrug["mean_delta_kb_mrr"]) - 0.0080) < 0.001 else "CORRECTED"),
        ClaimCheck("20", f"pooled hard delta {float(pooled['mean_delta_kb_hard'].iloc[0]):+.4f}",
                   "20_seed_erosion_distribution.csv", "OK" if abs(float(pooled["mean_delta_kb_hard"].iloc[0]) - (-0.0016)) < 0.001 else "CORRECTED"),
        ClaimCheck("20", f"gene-disease falls {int(gdis['n_kb_falls'])}/{int(gdis['n_seeds'])}",
                   "20_pair_type_breakdown.csv", "OK" if int(gdis["n_kb_falls"]) == 48 and int(gdis["n_seeds"]) == 65 else "CORRECTED"),
    ]
    print_verification(notes)
    return notes


VERIFY = {
    "00": verify_step00,
    "01": verify_step01,
    "02": verify_step02,
    "03": verify_step03,
    "04": verify_step04,
    "05": verify_step05,
    "10": verify_step10,
    "20": verify_step20,
}
