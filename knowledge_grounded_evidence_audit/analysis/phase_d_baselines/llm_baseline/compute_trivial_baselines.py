#!/usr/bin/env python3.11
"""Emit ``outputs/llm_baseline/trivial_baselines.json`` for Phase~2D framing.

Computes IID-uniform and always-DGR trivial baselines on the 162-set and the
strict CIViCmine-covered 41-set (same ``target_id``s as Case~C).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[4]
OUT_DEFAULT = (
    REPO / "knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/llm_baseline/trivial_baselines.json"
)
KB_PATH = REPO / "fine_tuning_experiments/schema_exp/eval/inputs/kb_surface_pairs.jsonl"
CIV_PATH = REPO / "knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/civicmine_baseline_case_c.json"

sys.path.insert(0, str(REPO))
from fine_tuning_experiments.schema_exp.eval.schema_expected_label import (  # noqa: E402
    schema_expected_label_set,
)


def expected_sv(r: dict) -> set[str]:
    civic = {
        "expected_pairing_family": r["pairing_family"],
        "heuristic_gold_s2_label": r["expected_label"],
    }
    exp, _ = schema_expected_label_set(civic, "S_pair", "primary", "set_valued")
    return set(exp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    rows = [json.loads(l) for l in KB_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    kb162 = [r for r in rows if r.get("expected_label") != "VARIANT_GENE"]
    if len(kb162) != 162:
        raise SystemExit(f"expected 162 evaluable rows, got {len(kb162)}")

    cov = json.loads(CIV_PATH.read_text(encoding="utf-8"))
    wanted = {c["target_id"] for c in cov["covered_targets"]}
    kb41 = [r for r in kb162 if r["target_id"] in wanted]
    if len(kb41) != 41:
        raise SystemExit(f"expected 41 covered targets, got {len(kb41)}")

    def iid_mean(targets: list[dict]) -> float:
        return mean(len(expected_sv(r)) / 8.0 for r in targets)

    def always_dgr(targets: list[dict]) -> dict[str, float | int]:
        hits = sum(1 for r in targets if "DRUG_GENE_REGULATION" in expected_sv(r))
        return {"n_hits": hits, "n_targets": len(targets), "accuracy": hits / len(targets)}

    d162 = always_dgr(kb162)
    d41 = always_dgr(kb41)

    out = {
        "schema_projection": "S_pair / primary / set_valued",
        "sources": {
            "kb_surface_pairs": str(KB_PATH.resolve()),
            "civicmine_case_c": str(CIV_PATH.resolve()),
        },
        "IID_uniform_mean_P_hit": {
            "definition": "Per target, draw label uniformly from 8 S_pair heads; P(hit)=|expected_set_sv|/8. Report mean.",
            "162_set": iid_mean(kb162),
            "41_set": iid_mean(kb41),
        },
        "always_predict_DRUG_GENE_REGULATION": {
            "definition": "Predict DRUG_GENE_REGULATION for every target; hit iff label ∈ expected_set_sv.",
            "162_set": d162,
            "41_set": d41,
        },
        "strict41_targets_with_DGR_in_expected_set": {
            "count": d41["n_hits"],
            "denominator": 41,
            "fraction": d41["accuracy"],
            "note": "Same as always-DGR hits on 41-set (singleton sets: hit iff S={DGR}).",
        },
        "modal_primary_expected_label_on_162": [
            {"label": a, "count": b} for a, b in Counter(r["expected_label"] for r in kb162).most_common(5)
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
