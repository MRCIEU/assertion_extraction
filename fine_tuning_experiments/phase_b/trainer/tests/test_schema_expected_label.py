"""Test the (family × heuristic gold × schema × projection mode) lookup in
`schema_expected_label.py` against its documented mapping.

Pins the full table so any silent change to the mapping (projection, family
resolution, VARIANT_GENE exclusion, AG fallback) is caught by CI.

Run directly:
    python3.11 -m fine_tuning_experiments.phase_b.trainer.tests.test_schema_expected_label
"""
from __future__ import annotations

import sys

from fine_tuning_experiments.schema_exp.eval.schema_expected_label import (
    resolve_family,
    schema_expected_label_set,
)

# Each case: (expected_pairing_family, heuristic_gold_s2_label, schema,
#             projection_mode, expected_set, expected_confidence)
CASES = [
    # ── DGR family, set_valued ─────────────────────────────────────────
    ("gene_drug", "DRUG_GENE_REGULATION", "S_flat", "set_valued",
     frozenset({"DRUG_GENE_REGULATION"}), "high"),
    ("gene_drug", "DRUG_GENE_REGULATION", "S_pair", "set_valued",
     frozenset({"DRUG_GENE_REGULATION"}), "high"),
    ("gene_drug", "DRUG_GENE_REGULATION", "S_mech", "set_valued",
     frozenset({"DRUG_GENE_REGULATION",
                "DGR_ACTIVATE", "DGR_INHIBIT", "DGR_METABOLIC",
                "DGR_REGULATE", "DGR_STRUCTURAL"}), "high"),
    # ── DGR family, single_label ───────────────────────────────────────
    ("gene_drug", "DRUG_GENE_REGULATION", "S_mech", "single_label",
     frozenset({"DRUG_GENE_REGULATION"}), "high"),
    # ── Gene_drug + AG heuristic under "primary" still maps to DGR ─────
    ("gene_drug", "ASSOCIATION_GENERAL", "S_pair", "set_valued",
     frozenset({"DRUG_GENE_REGULATION"}), "medium"),
    # ── Variant_disease family, set_valued ─────────────────────────────
    ("variant_disease", "VARIANT_DISEASE", "S_flat", "set_valued",
     frozenset({"ASSOCIATION_GENERAL"}), "medium"),
    ("variant_disease", "VARIANT_DISEASE", "S_pair", "set_valued",
     frozenset({"VARIANT_DISEASE"}), "medium"),
    ("variant_disease", "VARIANT_DISEASE", "S_mech", "set_valued",
     frozenset({"VARIANT_DISEASE"}), "medium"),
    ("variant_disease", "ASSOCIATION_GENERAL", "S_flat", "set_valued",
     frozenset({"ASSOCIATION_GENERAL"}), "high"),
    ("variant_disease", "ASSOCIATION_GENERAL", "S_pair", "set_valued",
     frozenset({"VARIANT_DISEASE"}), "high"),
    # ── VARIANT_GENE heuristic under variant_disease → unmapped ────────
    ("variant_disease", "VARIANT_GENE", "S_flat", "set_valued",
     frozenset(), "unmapped"),
    ("variant_disease", "VARIANT_GENE", "S_pair", "set_valued",
     frozenset(), "unmapped"),
    ("variant_disease", "VARIANT_GENE", "S_mech", "set_valued",
     frozenset(), "unmapped"),
    # ── Unknown family → unmapped ──────────────────────────────────────
    ("gene_gene", "GENE_GENE_ASSOC", "S_pair", "set_valued",
     frozenset(), "unmapped"),
]


def run() -> int:
    failures: list[str] = []
    for pf, gold, schema, mode, exp_set, exp_conf in CASES:
        target = {"expected_pairing_family": pf, "heuristic_gold_s2_label": gold}
        got_set, got_conf = schema_expected_label_set(target, schema, "primary", mode)
        if got_set != exp_set or got_conf != exp_conf:
            failures.append(
                f"pf={pf!r} gold={gold!r} schema={schema} mode={mode}: "
                f"expected ({sorted(exp_set)}, {exp_conf!r}) got "
                f"({sorted(got_set)}, {got_conf!r})"
            )
        else:
            print(f"[ok] {pf}/{gold}/{schema}/{mode} -> "
                  f"{sorted(got_set)} ({got_conf})")

    # Spot-check resolve_family under sensitivity strategy
    fam, conf = resolve_family(
        {"expected_pairing_family": "gene_drug",
         "heuristic_gold_s2_label": "ASSOCIATION_GENERAL"},
        strategy="sensitivity_trust_heuristic",
    )
    if fam != "AG_FAMILY" or conf != "medium":
        failures.append(
            f"sensitivity_trust_heuristic: expected (AG_FAMILY, medium) got ({fam!r}, {conf!r})"
        )
    else:
        print("[ok] sensitivity_trust_heuristic: gene_drug+AG -> AG_FAMILY")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\nPASS: all {len(CASES)} mapping cases + sensitivity fallback.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
