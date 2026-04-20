"""
Formal schema comparison framework.

Documents the evaluation protocol for SC0 vs SC1 vs SC3, including:
  - Evaluation metrics (KB_surface_mean, per-head F1, BC5CDR control)
  - Formal selection rule
  - Expected gold label distributions per schema on BioRED test

Outputs:
  - reports/schema_comparison_framework.json
  - reports/tables/biored_test_gold_by_schema.csv
  - reports/tables/schema_feasibility_summary.csv
"""
from __future__ import annotations

import csv, json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import PROC, REPORTS, TABLES, ensure_dirs


def load_test_gold(path: Path, filter_split: str | None = "test") -> Counter:
    """Load relation family counts, optionally filtered to a specific source_split."""
    c: Counter = Counter()
    if not path.exists(): return c
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            if filter_split and rec.get("source_split") != filter_split:
                continue
            for rel in rec.get("relations", []):
                c[rel.get("relation_family", "?")] += 1
    return c


def run() -> None:
    ensure_dirs()
    print("=== Schema Comparison Framework ===\n")

    # BioRED test gold distributions per schema
    # SC0: filter t1_biored.jsonl to source_split='test'
    # SC1/SC3: pre-filtered test-only files
    test_golds = {
        "S_flat": load_test_gold(PROC / "t1_biored_test_Sflat.jsonl", filter_split=None),
        "S_pair": load_test_gold(PROC / "t1_biored_test_Spair.jsonl",  filter_split=None),
        "S_mech": load_test_gold(PROC / "t1_biored_test_Smech.jsonl", filter_split=None),
    }
    # Note: SC1 and SC3 BioRED test gold are identical (by design):
    # SC3 differs from SC1 ONLY in DrugProt mechanism split.
    # BioRED does not have DrugProt mechanism labels → BioRED packages reused from SC1.

    print("BioRED test gold distributions per schema (990 total gold relations):\n")
    rows = []
    for schema, gold in test_golds.items():
        n_classes = len(gold) + 1   # +1 for NEGATIVE
        random_mf1 = round(1 / n_classes, 4)
        print(f"  {schema} ({n_classes} classes, random macro-F1 = {random_mf1}):")
        for head, cnt in sorted(gold.items(), key=lambda x: -x[1]):
            pct = 100 * cnt / 990
            print(f"    {head:<30} {cnt:>4} ({pct:.1f}%)")
            rows.append({"schema": schema, "head": head, "count": cnt,
                         "fraction": round(cnt/990,4),
                         "n_classes_incl_neg": n_classes,
                         "random_macro_f1": random_mf1})
        print()

    with open(TABLES / "biored_test_gold_by_schema.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["schema","head","count","fraction","n_classes_incl_neg","random_macro_f1"])
        w.writeheader(); w.writerows(rows)

    # Feasibility summary
    feasibility = {
        "S_flat": {"n_heads": 3, "min_support": 4827, "dead_heads": 0, "trainable": True,
                "axes": "E0×M0", "description": "Partially entity-aware baseline (original S2_current)"},
        "S_pair": {"n_heads": 6, "min_support": 410,  "dead_heads": 0, "trainable": True,
                "axes": "E1×M0", "description": "Fully entity-pair-type-aware; core oncology pairs"},
        "S_mech": {"n_heads": 10,"min_support": 687,  "dead_heads": 0, "trainable": True,
                "axes": "E1×M1", "description": "Entity-pair-type + DrugProt 5-group mechanism split"},
    }
    feasibility_rows = [{"schema": k, **v} for k, v in feasibility.items()]
    with open(TABLES / "schema_feasibility_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["schema","axes","n_heads","min_support","dead_heads","trainable","description"])
        w.writeheader(); w.writerows(feasibility_rows)

    # Evaluation protocol document
    framework = {
        "research_question": (
            "Given heterogeneous biomedical RE corpora with incompatible native label "
            "ontologies, what relation schema granularity maximises oncology assertion "
            "surfacing utility (KB_surface_mean) while maintaining trainability "
            "(all heads F1 > 0.05) on available public data?"
        ),
        "schema_candidates": list(feasibility.keys()),
        "primary_metric": {
            "name": "KB_surface_mean",
            "definition": "mean(1 - P(NEGATIVE)) across 165 CIViC targets",
            "why": (
                "Threshold-free; removes softmax-size confound between schemas "
                "(SC0:4-class, SC1:6-class, SC3:10-class). NEG class always present "
                "→ 1-P(NEG) directly comparable across all schemas."
            ),
            "scale": "[0, 1]",
            "interpretation": "Higher = model assigns more surfacing probability to CIViC oncology targets",
        },
        "secondary_metrics": {
            "per_head_F1": "Within-schema discriminability; NOT comparable across schemas",
            "BC5CDR_F1": "Control — DRUG_DISEASE unchanged across schemas; expected stable",
            "KB_surface_matched": "mean(P(expected_label)) — measures semantic label quality",
        },
        "trainability_criterion": "All heads F1 > 0.05 on BioRED test (within-schema gold)",
        "selection_rule": "SC* = argmax KB_surface_mean subject to trainability constraint",
        "statistical_test": "Bootstrap permutation test (1000 permutations) for SC* vs SC0",
        "evaluation_note": (
            "BioRED macro-F1 is NOT comparable across schemas: "
            "SC0 random=0.25, SC1 random=0.167, SC3 random=0.10. "
            "Report per-head F1 and use only KB_surface_mean for cross-schema comparison."
        ),
        "2d_pareto_figure": {
            "x_axis": "geometric mean of per-head F1 (overall discriminability)",
            "y_axis": "KB_surface_mean (downstream utility)",
            "interpretation": "SC* is on the Pareto frontier",
        },
        "sc0_correct_description": (
            "SC0 is NOT 'corpus membership only': it already maps BioRED "
            "Bind/Cotreatment/Drug_Interaction to DRUG_GENE_REGULATION "
            "(entity-type-aware for biochemical interactions) but maps all other "
            "BioRED association-type relations to ASSOCIATION_GENERAL regardless of "
            "entity pair type. SC1 extends this to all four core oncology entity-pair types."
        ),
        "sc1_design_tradeoff": (
            "SC1 collapses relation polarity within entity-pair types "
            "(BioRED GENE_DISEASE: Association=1133, Neg_Corr=62, Pos_Corr=50 → all GENE_DISEASE) "
            "to gain between-type discriminability. No public span-supervised oncology corpus "
            "provides directional gene-disease labels."
        ),
        "sc3_grouping_citation": (
            "DrugProt 13-type → 5-group mapping follows annotation guidelines "
            "(Miranda-Escalada et al., BioCreative VII 2021). "
            "PART-OF is DGR_STRUCTURAL (compositional), not DGR_METABOLIC (enzymatic)."
        ),
    }
    (REPORTS / "schema_comparison_framework.json").write_text(json.dumps(framework, indent=2))
    print(f"  Outputs: schema_comparison_framework.json, biored_test_gold_by_schema.csv, schema_feasibility_summary.csv")
