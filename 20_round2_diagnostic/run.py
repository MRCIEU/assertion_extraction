#!/usr/bin/env python3
"""Round 2 diagnostic: training curves, two-axis timing, power (Round 1 data only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from importlib import import_module

cfg = import_module("20_round2_diagnostic.config")
ci = import_module("20_round2_diagnostic.checkpoint_inventory")
tc = import_module("20_round2_diagnostic.training_curves")
ta = import_module("20_round2_diagnostic.two_axis")
pc = import_module("20_round2_diagnostic.power_check")
fig = import_module("20_round2_diagnostic.figures")
rep = import_module("20_round2_diagnostic.report")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 2 diagnostic (analysis only)")
    parser.add_argument(
        "--rescore-sweep",
        action="store_true",
        help="Inference on sweep val_loss/val_f1 checkpoints (slow; uses GPU if available)",
    )
    args = parser.parse_args()

    print("=== Round 2 diagnostic start ===")

    inv, inv_case = ci.build_checkpoint_inventory()
    inv.to_csv(cfg.OUTPUT_DIR / "checkpoint_inventory.csv", index=False)
    ci.print_inventory_summary(inv, inv_case)

    curves = tc.load_epoch_curves()
    mean_curves = tc.encoder_mean_curves(curves)
    tsum = tc.training_curve_summary(curves)
    tsum.to_csv(cfg.OUTPUT_DIR / "training_curve_summary.csv", index=False)
    shape = tc.describe_curve_shape(tsum)
    print("\n=== Step 1: Training-curve shape ===")
    print(shape)
    for _, r in tsum.iterrows():
        print(
            f"  {r['model_id']}: peak val_f1 epoch mean={r['mean_peak_val_f1_epoch']:.2f} "
            f"loss-rise epoch mean={r['mean_val_loss_rise_epoch']:.2f} "
            f"plateau width mean={r['mean_plateau_width_epochs']:.2f}"
        )

    per_run = pd.read_csv(cfg.R1_PER_RUN_CSV)
    easy_hard = pd.read_csv(cfg.R1_EASY_HARD_CSV) if cfg.R1_EASY_HARD_CSV.exists() else None
    sweep_cache = cfg.DATA_DIR / "sweep_two_point_kb.csv"

    traj = ta.build_two_axis_trajectory(
        per_run,
        pool=None,
        sweep_cache=sweep_cache,
        rescore_sweep=args.rescore_sweep,
        easy_hard=easy_hard,
    )
    traj.to_csv(cfg.OUTPUT_DIR / "two_axis_trajectory.csv", index=False)
    timing = ta.summarize_timing(traj)
    print("\n=== Step 2: Two-axis trajectory ===")
    print(timing["narrative"])
    for mid in cfg.FOCUS_MODEL_IDS:
        sub = traj[(traj["model_id"] == mid) & (traj["source"] == "round1_main")]
        if not sub.empty:
            print(
                f"  {mid} [val_f1_best main] n={len(sub)} "
                f"bench F1 mean={sub['benchmark_f1'].mean():.3f} "
                f"KB hard mean={sub['kb_mrr_hard'].mean():.3f}"
            )
    sw = traj[traj["source"] == "round1_sweep_recipe_match"]
    for _, r in sw.iterrows():
        print(
            f"  SWEEP {r['model_id']} {r['trajectory_point']} ep={int(r['epoch'])} "
            f"bench={r['benchmark_f1']:.3f} KB hard={r['kb_mrr_hard']:.3f} "
            f"KB easy={r['kb_mrr_easy']:.3f}"
        )

    main_traj = traj[traj["source"] == "round1_main"]
    pw = pc.build_power_check(main_traj, traj)
    pw.to_csv(cfg.OUTPUT_DIR / "power_check.csv", index=False)
    print("\n=== Step 3: Power check ===")
    for _, r in pw.iterrows():
        print(
            f"  {r['short_name']}: KB hard SD={r['kb_mrr_hard_sd_at_val_f1_ckpt']:.3f} "
            f"est. effect={r['estimated_training_effect_hard']:.3f} "
            f"detectable~{r['approx_detectable_effect_hard']:.3f} "
            f"clears={r['effect_clears_detectable_band']}"
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

    print("\n=== Round 2 diagnostic complete ===")
    print(f"Outputs -> {cfg.OUTPUT_DIR}")
    print(f"Figures -> {cfg.FIGURE_DIR}")
    print(f"Report -> {cfg.REPORT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
