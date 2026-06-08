#!/usr/bin/env python3
"""Round 2 diagnostic: training dynamics and power (inference only)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from importlib import import_module


def _write_scoring_complete(cfg, expected: int, scored: int) -> None:
    payload = {
        "expected_epoch_checkpoints": expected,
        "scored_epochs": scored,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "complete": scored >= expected,
    }
    cfg.EPOCH_SCORE_COMPLETE.parent.mkdir(parents=True, exist_ok=True)
    cfg.EPOCH_SCORE_COMPLETE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_score_epochs(*, force: bool = False) -> None:
    cfg = import_module("20_round2_diagnostic.config")
    ta = import_module("20_round2_diagnostic.two_axis")
    expected = ta.count_epoch_checkpoints_to_score()
    print(f"=== Epoch scoring (GPU): {expected} checkpoints across focus encoders ===")
    ta.trajectory_from_training_logs(rescore_epochs=True)
    scored = ta.count_scored_epochs()
    _write_scoring_complete(cfg, expected, scored)
    print(f"Scored epochs on disk: {scored}/{expected}")


def run_analysis(*, allow_partial: bool = False) -> None:
    cfg = import_module("20_round2_diagnostic.config")
    ci = import_module("20_round2_diagnostic.checkpoint_inventory")
    tc = import_module("20_round2_diagnostic.training_curves")
    ta = import_module("20_round2_diagnostic.two_axis")
    pc = import_module("20_round2_diagnostic.power_check")
    fig = import_module("20_round2_diagnostic.figures")
    rep = import_module("20_round2_diagnostic.report")

    expected = ta.count_epoch_checkpoints_to_score()
    scored = ta.count_scored_epochs()
    if not allow_partial and scored < expected:
        raise SystemExit(
            f"Epoch scoring incomplete: {scored}/{expected}. "
            "Run --score-epochs-only first or use --allow-partial-analysis."
        )

    t_all = time.perf_counter()
    print("=== Round 2 diagnostic analysis (CPU) ===")

    inv, inv_case = ci.build_checkpoint_inventory()
    inv.to_csv(cfg.OUTPUT_DIR / "checkpoint_inventory.csv", index=False)
    ci.print_inventory_summary(inv, inv_case)

    curves = tc.load_epoch_curves()
    mean_curves = tc.encoder_mean_curves(curves)
    tsum = tc.training_curve_summary(curves)
    tsum.to_csv(cfg.OUTPUT_DIR / "training_curve_summary.csv", index=False)
    shape = tc.describe_curve_shape(tsum)
    print(f"\n=== Training-curve shape ===\n{shape}")

    traj = ta.build_two_axis_trajectory(rescore_epochs=False)
    traj.to_csv(cfg.OUTPUT_DIR / "two_axis_trajectory.csv", index=False)
    timing = ta.summarize_timing(traj)
    print(f"\n=== Two-axis trajectory ===\n{timing['narrative']}")

    main_traj = traj[traj["source"] == "r11_best"] if not traj.empty else traj
    pw = pc.build_power_check(main_traj, traj)
    pw.to_csv(cfg.OUTPUT_DIR / "power_check.csv", index=False)
    print("\n=== Power check ===")
    for _, r in pw.iterrows():
        print(
            f"  {r.get('short_name')}: effect={r.get('estimated_training_effect_hard', float('nan')):.3f}, "
            f"hard SD={r.get('kb_mrr_hard_sd_at_val_f1_ckpt', float('nan')):.3f}, "
            f"R1 pool SD={r.get('r1_mean_within_encoder_sd_gene_drug', float('nan')):.3f}"
        )

    fig.figure1_training_curves(mean_curves)
    fig.figure2_two_axis_timing(traj)
    fig.figure3_power(pw)

    rep.write_report(
        inventory_case=inv_case,
        curve_shape=shape,
        timing_notes=timing["narrative"],
        training_summary=tsum,
        power_df=pw,
        two_axis=traj,
    )
    print(f"\n=== Round 2 diagnostic complete ({time.perf_counter() - t_all:.1f}s) ===")


def run_three_point_analysis() -> None:
    """CPU post-hoc three-point paired timing (reads epoch_kb_trajectory.csv only)."""
    tp = import_module("20_round2_diagnostic.three_point_timing")
    fig = import_module("20_round2_diagnostic.figures")
    rep = import_module("20_round2_diagnostic.report")

    three_pt, summary, overall = tp.run_three_point_timing()
    fig.figure4_three_point_paired(three_pt, summary)
    rep.append_three_point_section(three_pt=three_pt, summary=summary, overall=overall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 2 diagnostic (inference only)")
    parser.add_argument(
        "--score-epochs-only",
        action="store_true",
        help="Score benchmark F1 and KB at every saved epoch for focus encoders (GPU)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Run curves, power check, figures, report from stored epoch scores",
    )
    parser.add_argument("--force-rescore", action="store_true", help="Overwrite cached epoch scores")
    parser.add_argument(
        "--allow-partial-analysis",
        action="store_true",
        help="Run analysis even if epoch scoring is incomplete",
    )
    parser.add_argument(
        "--three-point-timing",
        action="store_true",
        help="Three-point paired timing from saved epoch_kb_trajectory.csv (CPU post-hoc)",
    )
    parser.add_argument("--dry-trace-three-point", action="store_true", help="Dry trace only")
    args = parser.parse_args()

    if args.dry_trace_three_point:
        ok = import_module("20_round2_diagnostic.three_point_timing").dry_trace()
        raise SystemExit(0 if ok else 1)

    if args.score_epochs_only:
        run_score_epochs(force=args.force_rescore)
        return
    if args.three_point_timing:
        run_three_point_analysis()
        return
    if args.analyze_only:
        run_analysis(allow_partial=args.allow_partial_analysis)
        return
    raise SystemExit("Use --score-epochs-only, --analyze-only, --three-point-timing, or --dry-trace-three-point")


if __name__ == "__main__":
    main()
