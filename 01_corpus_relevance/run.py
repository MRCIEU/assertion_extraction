#!/usr/bin/env python3
"""Step 01 entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from importlib import import_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 01: corpus relevance")
    parser.add_argument("--force", action="store_true", help="Re-pull BigBio corpora")
    parser.add_argument("--force-oncology-mesh", action="store_true", help="Re-fetch PubMed MeSH index")
    args = parser.parse_args()

    print(f"=== Step 01 start {__import__('datetime').datetime.now().isoformat()} ===")
    inventories_mod = import_module("01_corpus_relevance.inventories")
    alignment_mod = import_module("01_corpus_relevance.alignment")
    granularity_mod = import_module("01_corpus_relevance.granularity")
    volume_mod = import_module("01_corpus_relevance.volume")
    mapping_mod = import_module("01_corpus_relevance.drugprot_mapping")
    pmid_mod = import_module("01_corpus_relevance.pmid_diagnostics")
    figures_mod = import_module("01_corpus_relevance.figures")
    report_mod = import_module("01_corpus_relevance.report")

    inventories = inventories_mod.build_inventories(force=args.force)
    alignment_mod.run_alignment(inventories)
    gran_df = granularity_mod.run_granularity(inventories)
    figures_mod.plot_granularity_ladder(gran_df)
    figures_mod.plot_corpus_overview()
    volume_mod.run_volume()
    mapping_mod.run_drugprot_mapping()
    diag = pmid_mod.run_pmid_diagnostics()
    figures_mod.plot_pmid_overlap(diag["overlap"])
    figures_mod.plot_conflict_summary(diag["conflict"])
    figures_mod.plot_leakage(diag["leakage"])
    oncology_mod = import_module("01_corpus_relevance.oncology_subset")
    oncology = oncology_mod.run_oncology_subset(force_mesh=args.force_oncology_mesh)
    figures_mod.plot_oncology_fractions(oncology["fractions"])
    figures_mod.plot_oncology_agreement(oncology["agreement"])
    report_mod.generate_report()
    print("=== Step 01 complete ===")


if __name__ == "__main__":
    main()
