#!/usr/bin/env python3.11
"""
Oncology Projection — Main Entry Point

Steps:
  1. Document and validate the recovered cancer surface lexicon
  2. Run keyword-based T2 projection (original method, for reference)
  3. Run MeSH C04 T2 projection (preferred, reproducible method)
  4. Compare both projection methods

Outputs → ~/projects/project_1/oncology_projection/

Usage:
  python3.11 run_projection.py [--skip-mesh]   # skip-mesh if no NCBI key
"""
from __future__ import annotations
import sys, json, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import OUT_ROOT, REPORTS, TABLES, DATA_OUT, ensure_dirs
from pipeline.lexicon import CANCER_REGEX, CANCER_TERMS, KNOWN_GAPS, PAPER_DESCRIPTION
from pipeline.keyword_projection import run as run_keyword
from pipeline.mesh_projection import run as run_mesh
from pipeline.projection_comparison import run as run_comparison

SKIP_MESH = "--skip-mesh" in sys.argv


def save_lexicon_record() -> None:
    """Save the lexicon documentation."""
    ensure_dirs()
    record = {
        "name": "english_cancer_surface_regex_recovered",
        "source": "Recovered from compiled .pyc of oncology_projection/utils/lexicon.py (2026-04-15). "
                  "Original source file no longer exists; pyc was created under Python 3.10.",
        "regex": CANCER_REGEX,
        "terms": CANCER_TERMS,
        "n_terms": len(CANCER_TERMS),
        "flags": "re.IGNORECASE",
        "match_type": "substring (re.search)",
        "known_gaps": KNOWN_GAPS,
        "paper_description": PAPER_DESCRIPTION,
        "validated_precision": 1.0,
        "validated_recall": 0.923,
        "validation_note": "Spot-check on 13 cancer + 8 non-cancer examples",
    }
    (DATA_OUT / "cancer_lexicon.json").write_text(json.dumps(record, indent=2))
    print(f"  Lexicon documented: {len(CANCER_TERMS)} terms → cancer_lexicon.json")


def main() -> None:
    ensure_dirs()
    print("=" * 60)
    print("ONCOLOGY PROJECTION — Full Pipeline")
    print(f"Output root: {OUT_ROOT}")
    print("=" * 60)
    print()

    print("STEP 1: Cancer Surface Lexicon")
    print("-" * 40)
    save_lexicon_record()
    print(f"  Terms ({len(CANCER_TERMS)}): {', '.join(CANCER_TERMS[:8])}...")
    print()

    print("STEP 2: Keyword-Based T2 Projection")
    print("-" * 40)
    run_keyword()
    print()

    if not SKIP_MESH:
        print("STEP 3: MeSH C04 T2 Projection (Preferred)")
        print("-" * 40)
        run_mesh()
        print()

        print("STEP 4: Method Comparison")
        print("-" * 40)
        run_comparison()
        print()
    else:
        print("STEP 3: MeSH projection SKIPPED (--skip-mesh flag)")
        print("  Run without --skip-mesh and with NCBI_API_KEY set")
        print()

    # Final summary report
    summary = {
        "subproject": "oncology_projection",
        "output_root": str(OUT_ROOT),
        "lexicon_terms": len(CANCER_TERMS),
        "methods": ["keyword_cancer_lexicon", "mesh_c04_neoplasms"],
        "preferred_method": "mesh_c04_neoplasms",
        "paper_justification": (
            "MeSH C04 filtering is reproducible (standard NCBI taxonomy) and citable. "
            "Keyword lexicon is retained for documentation of original methodology."
        ),
        "reports": [
            "data/processed/cancer_lexicon.json",
            "reports/keyword_projection_summary.json",
            "reports/mesh_projection_summary.json",
            "reports/projection_comparison.json",
        ],
        "tables": [
            "reports/tables/keyword_projection_stats.csv",
            "reports/tables/mesh_projection_stats.csv",
            "reports/tables/projection_comparison_stats.csv",
        ],
    }
    (REPORTS / "projection_index.json").write_text(json.dumps(summary, indent=2))
    print("=" * 60)
    print("ONCOLOGY PROJECTION COMPLETE")
    print(f"Index: {REPORTS / 'projection_index.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
