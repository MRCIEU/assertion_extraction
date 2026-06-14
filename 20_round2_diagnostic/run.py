#!/usr/bin/env python3
"""Round 2 training-dynamics diagnostic (inference only, 5e-6 per-epoch checkpoints)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from importlib import import_module

import pandas as pd


def run_score_epochs(*, force: bool = False, model_ids: list[str] | None = None) -> None:
    es = import_module("20_round2_diagnostic.epoch_scoring")
    es.score_all_epochs(model_ids=model_ids, force=force)


def run_supplement_cross(*, force: bool = False, model_ids: list[str] | None = None) -> None:
    scm = import_module("20_round2_diagnostic.supplement_cross_metrics")
    scm.supplement_all_cross_metrics(model_ids=model_ids, force=force)


def run_stratum_epoch1_cache(*, model_ids: list[str] | None = None) -> None:
    cfg = import_module("20_round2_diagnostic.config")
    mundane = import_module("20_round2_diagnostic.mundane_explanations")

    paired = pd.read_csv(cfg.PAIRED_CHANGES_CSV)
    if model_ids:
        paired = paired[paired["model_id"].isin(model_ids)]
    mundane.build_epoch1_stratum_cache(paired)


def run_analysis(*, allow_partial: bool = False, skip_stratum_inference: bool = False) -> None:
    cfg = import_module("20_round2_diagnostic.config")
    ci = import_module("20_round2_diagnostic.checkpoint_inventory")
    es = import_module("20_round2_diagnostic.epoch_scoring")
    adj = import_module("20_round2_diagnostic.adjudication")
    fig = import_module("20_round2_diagnostic.figures")
    rep = import_module("20_round2_diagnostic.report")

    expected = es.count_expected_epochs()
    scored = es.count_scored_epochs()
    if not allow_partial and scored < expected:
        raise SystemExit(
            f"Epoch scoring incomplete: {scored}/{expected}. "
            "Run --score-epochs-only first or use --allow-partial-analysis."
        )

    inv, inv_case = ci.build_checkpoint_inventory()
    inv.to_csv(cfg.INVENTORY_CSV, index=False)
    ci.print_inventory_summary(inv, inv_case)

    results = adj.run_adjudication_analysis()

    mundane_mod = import_module("20_round2_diagnostic.mundane_explanations")
    mundane = mundane_mod.run_mundane_explanations(
        results["trajectory"],
        results["paired"],
        skip_stratum_inference=skip_stratum_inference,
    )

    enc_mod = import_module("20_round2_diagnostic.encoder_correlation")
    gd_enc = results.get("gene_disease", {}).get("encoder")
    encoder_corr = enc_mod.run_encoder_correlation(
        gd_enc if gd_enc is not None else pd.DataFrame(),
        results["trajectory"],
    )

    qual_mod = import_module("20_round2_diagnostic.qualitative_errors")
    qual = qual_mod.run_qualitative_errors()

    gd = results.get("gene_disease", {})
    fig.generate_all_figures(
        results["trajectory"],
        results["paired"],
        results["hard_easy"],
        results["pair_type"],
        results["robustness"],
        gd.get("pair_subset"),
        mundane.get("timing_summary"),
        mundane.get("stratum_summary"),
        encoder_corr.get("table"),
        qual.get("patterns"),
        qual.get("summary"),
    )

    rep.write_report(
        inventory_case=inv_case,
        inventory=inv,
        verdict=results["verdict"],
        gene_disease_verdict=gd.get("verdict", {}),
        seed_dist=results["seed_dist"],
        hard_easy=results["hard_easy"],
        pair_type=results["pair_type"],
        robustness=results["robustness"],
        paired=results["paired"],
        gd_subset=gd.get("subset"),
        gd_robustness=gd.get("robustness"),
        gd_encoder=gd.get("encoder"),
        gd_seed=gd.get("seed_dist"),
        mundane=mundane,
        encoder_corr=encoder_corr,
    )
    rep.write_qualitative_report(qual)
    rep.write_readme(
        verdict=results["verdict"],
        gene_disease_verdict=gd.get("verdict", {}),
        seed_dist=results["seed_dist"],
        inventory=inv,
        gd_subset=gd.get("subset"),
        qual_summary=qual.get("summary"),
        mundane=mundane,
    )
    print("\n=== Round 2 diagnostic analysis complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 2 training-dynamics diagnostic")
    parser.add_argument(
        "--score-epochs-only",
        action="store_true",
        help="Score benchmark + KB at every saved epoch checkpoint (GPU)",
    )
    parser.add_argument("--analyze-only", action="store_true", help="CPU analysis from stored scores")
    parser.add_argument(
        "--supplement-cross-metrics-only",
        action="store_true",
        help="GPU: add pair×subset cross metrics to existing epoch score JSON",
    )
    parser.add_argument("--force-rescore", action="store_true", help="Overwrite existing epoch score JSON")
    parser.add_argument(
        "--model-id",
        action="append",
        default=None,
        help="Limit scoring to encoder(s)",
    )
    parser.add_argument(
        "--stratum-epoch1-only",
        action="store_true",
        help="GPU/CPU: build epoch-1 stratum score cache for pool-size analysis",
    )
    parser.add_argument(
        "--skip-stratum-inference",
        action="store_true",
        help="Skip epoch-1 inference during analyze (use existing stratum cache)",
    )
    parser.add_argument(
        "--allow-partial-analysis",
        action="store_true",
        help="Run analysis even if epoch scoring is incomplete",
    )
    args = parser.parse_args()

    if args.score_epochs_only:
        run_score_epochs(force=args.force_rescore, model_ids=args.model_id)
        return
    if args.supplement_cross_metrics_only:
        run_supplement_cross(force=args.force_rescore, model_ids=args.model_id)
        return
    if args.stratum_epoch1_only:
        run_stratum_epoch1_cache(model_ids=args.model_id)
        return
    if args.analyze_only:
        run_analysis(
            allow_partial=args.allow_partial_analysis,
            skip_stratum_inference=args.skip_stratum_inference,
        )
        return
    raise SystemExit(
        "Use --score-epochs-only, --supplement-cross-metrics-only, "
        "--stratum-epoch1-only, or --analyze-only"
    )


if __name__ == "__main__":
    main()
