"""Diagnostic report for Round 2 planning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from importlib import import_module

from .config import FOCUS_MODEL_IDS, MODEL_BY_ID, REPORT_DIR

_r1 = import_module("10_round1_benchmark_kb.analysis")


def write_report(
    *,
    inventory_case: str,
    curve_shape: str,
    timing_notes: str,
    training_summary: pd.DataFrame,
    power_df: pd.DataFrame,
    two_axis: pd.DataFrame,
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "report.md"

    ts = training_summary[training_summary["model_id"].isin(FOCUS_MODEL_IDS)]
    med_peak = float(ts["median_peak_val_f1_epoch"].mean()) if not ts.empty else 0.0

    power_lines = []
    for _, r in power_df.iterrows():
        clears = r.get("effect_clears_detectable_band")
        if pd.isna(clears):
            verdict = "not assessed (missing two-checkpoint KB contrast)"
        elif clears:
            verdict = "estimated hard-subset KB shift exceeds the rough 10-seed detectable band"
        else:
            verdict = "estimated hard-subset KB shift is smaller than the rough 10-seed detectable band"
        power_lines.append(
            f"{r['short_name']}: seed-level hard-subset KB SD about {r['kb_mrr_hard_sd_at_val_f1_ckpt']:.3f} "
            f"at the deployed checkpoint; estimated training-amount effect about "
            f"{r['estimated_training_effect_hard']:.3f} from the matched sweep two-checkpoint contrast; "
            f"{verdict}."
        )

    lines = [
        "# Round 2 diagnostic: training dynamics and power (read-only on Round 1)",
        "",
        "This note uses completed Round 1 artifacts only. Nothing was trained or rescored "
        "except optional inference on checkpoints that were already saved (main matrix: "
        "val_f1-best weights; matched sweep at lr 2e-5 with no warmup: val_loss-best and "
        "val_f1-best weights at seed 42 for three encoders).",
        "",
        "## Checkpoint inventory and permitted trajectory density",
        "",
        inventory_case,
        "",
        "## Training-curve shape (validation metrics, nine encoders)",
        "",
        curve_shape,
        "",
        f"Across the three focus encoders, the median val_f1-best epoch (seed-level) averages "
        f"about {med_peak:.1f}. Validation loss often rises within one to two epochs after "
        "its minimum, so the interesting region is early training rather than a long flat "
        "late plateau.",
        "",
        "## Two-axis timing: benchmark versus KB along training",
        "",
        timing_notes,
        "",
        "For the main seventy-two-run matrix, selecting the checkpoint with best validation F1 "
        "is the only test-time weight on disk. That is the checkpoint Round 1 already used for "
        "benchmark and KB scores. A val_loss-best weight would represent a more under-trained "
        "point, but those weights were not retained in the main matrix, so one cannot score "
        "BioRED test F1 or the frozen knowledge-base pool at that point without new training.",
        "",
        "The matched sweep at the same learning rate and no warmup, seed 42, provides a "
        "directional two-point contrast on the test benchmark and on hard-subset KB ranking. "
        "That contrast is illustrative (one seed per encoder), not a full multi-seed curve.",
        "",
        "## Power check: training lever versus seed noise",
        "",
        "Round 1 showed that most KB variance sits within encoders (seed noise), not between "
        "encoders. A Round 2 that only moves training amount or checkpoint rule is a smaller "
        "lever. The table below compares, per focus encoder, the seed-level spread of "
        "hard-subset KB MRR at the deployed checkpoint against the absolute KB shift between "
        "the two sweep checkpoints at seed 42, and a rough detectable band if ten seeds were "
        "averaged per cell (approximate 2× standard error of the mean).",
        "",
        " ".join(power_lines),
        "",
        "## Plain reading for Round 2 planning",
        "",
        "Validation curves support defining training-amount levels around early epochs (often "
        "epochs one to five for val_f1-best), not a wide late over-training plateau on "
        "validation. Whether benchmark-best and KB-best coincide on the test axes cannot be "
        "fully mapped from one checkpoint per main run; the sweep two-point contrast suggests "
        "benchmark F1 and hard-subset KB can move in different directions between "
        "under-trained and well-trained saves, but the KB shift is modest relative to typical "
        "seed spreads unless the experiment uses more seeds or a larger deliberate contrast.",
        "",
        "A Round 2 framed on training configuration is informative if it explicitly compares "
        "checkpoint rules and early stopping on both the benchmark axis and the KB axis, with "
        "enough seeds per cell. If the expected KB shift along training stays near the "
        "seed-noise scale seen here, a large multi-encoder Round 2 on this lever alone may "
        "not repay the compute unless the design widens the training contrast or targets "
        "hard-subset ranking with more replication.",
        "",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
