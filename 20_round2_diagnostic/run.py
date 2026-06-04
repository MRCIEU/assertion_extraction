#!/usr/bin/env python3
"""Round 2 diagnostic: training dynamics and power (inference only)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from importlib import import_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 2 diagnostic (analysis only)")
    parser.add_argument(
        "--rescore-epochs",
        action="store_true",
        help="Score benchmark F1 and KB at every saved epoch for focus encoders (GPU; slow)",
    )
    args = parser.parse_args()

    cfg = import_module("20_round2_diagnostic.config")
    ci = import_module("20_round2_diagnostic.checkpoint_inventory")
    tc = import_module("20_round2_diagnostic.training_curves")
    ta = import_module("20_round2_diagnostic.two_axis")
    pc = import_module("20_round2_diagnostic.power_check")
    fig = import_module("20_round2_diagnostic.figures")
    rep = import_module("20_round2_diagnostic.report")

    t_all = time.perf_counter()
    print("=== Round 2 diagnostic start ===")

    t0 = time.perf_counter()
    inv, inv_case = ci.build_checkpoint_inventory()
    inv.to_csv(cfg.OUTPUT_DIR / "checkpoint_inventory.csv", index=False)
    ci.print_inventory_summary(inv, inv_case)
    print(f"  [timing] Step 0: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    curves = tc.load_epoch_curves()
    mean_curves = tc.encoder_mean_curves(curves)
    tsum = tc.training_curve_summary(curves)
    tsum.to_csv(cfg.OUTPUT_DIR / "training_curve_summary.csv", index=False)
    shape = tc.describe_curve_shape(tsum)
    print(f"\n=== Step 1: Training-curve shape ===\n{shape}")
    print(f"  [timing] Step 1: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    traj = ta.build_two_axis_trajectory(rescore_epochs=args.rescore_epochs)
    traj.to_csv(cfg.OUTPUT_DIR / "two_axis_trajectory.csv", index=False)
    timing = ta.summarize_timing(traj)
    print(f"\n=== Step 2: Two-axis trajectory ===\n{timing['narrative']}")
    print(f"  [timing] Step 2: {time.perf_counter() - t0:.1f}s")

    main_traj = traj[traj["source"] == "r11_best"] if not traj.empty else traj
    pw = pc.build_power_check(main_traj, traj)
    pw.to_csv(cfg.OUTPUT_DIR / "power_check.csv", index=False)
    print("\n=== Step 3: Power check ===")
    for _, r in pw.iterrows():
        print(
            f"  {r.get('short_name', r.get('model_id'))}: "
            f"KB hard SD={r.get('kb_mrr_hard_sd_at_val_f1_ckpt', float('nan')):.3f}"
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

    print(f"\n=== Round 2 diagnostic complete ({time.perf_counter() - t_all:.1f}s total) ===")


if __name__ == "__main__":
    main()
