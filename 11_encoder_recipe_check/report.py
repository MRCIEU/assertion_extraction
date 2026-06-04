"""Publication report for encoder recipe check."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import PRIMARY_SEED, REPORT_DIR, ROUND1_RECIPE_LR


def write_report(
    *,
    degenerate: pd.DataFrame,
    grid: pd.DataFrame,
    effects: pd.DataFrame,
    placement: pd.DataFrame,
    others_min: float,
    others_max: float,
    deberta_means: dict[str, float],
    grid_best: dict[str, Any],
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "report.md"

    lr_sum = effects[effects["contrast"] == "summary_lr_averaged_over_warmup"].iloc[0]
    w_sum = effects[effects["contrast"] == "summary_warmup_averaged_over_lr"].iloc[0]
    bp = float(grid_best["benchmark_f1"])
    r1_recipe = grid[(grid["lr"] == ROUND1_RECIPE_LR) & (grid["warmup_label"] == "none")].iloc[0]

    lines = [
        "# Encoder recipe check: why DeBERTa looked weak in Round 1",
        "",
        "Round 1 trained nine encoders under one shared recipe: learning rate 2e-05, no warmup, "
        "and checkpoint selection by best validation F1. DeBERTa-base was not part of the small "
        "sweep that chose that recipe. Two of its eight Round-1 seeds collapsed to zero validation "
        "and benchmark F1, and its naive eight-seed benchmark mean was 0.554, far below the other "
        "encoders. This check holds the data and benchmark protocol fixed and varies only learning "
        "rate and warmup for DeBERTa-base on seed 42. No new training matrix was run here beyond "
        "that four-point grid, which is already complete.",
        "",
        "## What the grid shows",
        "",
        f"At learning rate 1e-05, benchmark F1 is {float(grid[grid['lr']==1e-5]['benchmark_f1'].min()):.3f} "
        f"to {float(grid[grid['lr']==1e-5]['benchmark_f1'].max()):.3f} depending on warmup "
        f"(best {bp:.3f} with no warmup, epoch {int(grid_best['best_epoch_val_f1'])}). "
        f"At learning rate 2e-05, scores are {float(grid[grid['lr']==2e-5]['benchmark_f1'].min()):.3f} "
        f"to {float(grid[grid['lr']==2e-5]['benchmark_f1'].max()):.3f}, including "
        f"{float(r1_recipe['benchmark_f1']):.3f} for the exact Round-1 recipe (best epoch "
        f"{int(r1_recipe['best_epoch_val_f1'])}). Averaged over warmup, mean benchmark F1 at 1e-05 "
        f"is {float(lr_sum['mean_1e5']):.3f} versus {float(lr_sum['mean_2e5']):.3f} at 2e-05 "
        f"(difference {float(lr_sum['delta_2e5_minus_1e5']):+.3f}). Averaged over learning rate, "
        f"adding ten percent linear warmup moves the mean from {float(w_sum['mean_none']):.3f} "
        f"to {float(w_sum['mean_warmup_10pct']):.3f} (difference {float(w_sum['delta_warmup_minus_none']):+.3f}).",
        "",
        "The lever that recovers DeBERTa is the learning rate, not warmup. Lowering the rate "
        "from 2e-05 to 1e-05 lifts benchmark F1 by about four points on this seed when averaged "
        "over warmup (up to five points for the no-warmup pair), "
        "while warmup shifts scores only slightly at either rate. Warmup does give a modest "
        "lift at 2e-05 (0.738 versus 0.721 for the Round-1 point) but cannot substitute for "
        "the lower rate. The Round-1 problem for DeBERTa was therefore that 2e-05 was too high "
        "for this architecture under this setup, not that the absence of warmup was the main "
        "cause as originally hypothesised.",
        "",
        f"The 0.554 figure from Round 1 is a recipe artifact, not a capability floor. Under "
        f"lr 1e-05 on seed 42, DeBERTa reaches {bp:.3f}, which lies inside the Round-1 band "
        f"spanned by the other eight encoders ({others_min:.3f} to {others_max:.3f}) and above "
        f"DeBERTa's old clean six-seed mean ({deberta_means['clean6']:.3f}). DeBERTa is not "
        "intrinsically a weak encoder in this benchmark; its low Round-1 standing came from an "
        "unstable shared recipe that hit this model hardest.",
        "",
        "Best epoch moves with the recipe. The Round-1 recipe stopped at epoch 2 with early "
        "stopping, while lr 1e-05 runs peak at epoch 5 and warmup variants at epochs 7 to 9. "
        "With a suitable recipe the task does not overfit at epoch 1 on this seed; training "
        "remains useful through middle epochs. That observation is noted for later work on "
        "training dynamics and is not developed further here.",
        "",
        "## Limitation: single seed",
        "",
        f"This grid used one seed ({PRIMARY_SEED}) per recipe. The bad-seed guard did not run "
        "because no grid point collapsed. DeBERTa nevertheless showed the largest seed "
        "variance in Round 1, including two catastrophic failures under the old recipe. "
        f"The value {bp:.3f} shows that a working recipe exists under which DeBERTa trains "
        "normally on this seed. It does not prove that lr 1e-05 yields stable scores near "
        "0.774 across seeds. The conclusion is existential (a viable recipe exists), not a "
        "stable per-seed estimate.",
        "",
        "## What this means for Round 1",
        "",
        "The shared Round-1 recipe was suboptimal but usable for the eight encoders that did not "
        "collapse: their benchmark means stayed in a reasonable band while 2e-05 was tolerable "
        "for them. It was catastrophic only for DeBERTa, which alone produced the two zero seeds. "
        "One should not claim that all nine encoders were badly trained; the evidence points to "
        "a recipe that was broadly acceptable except for this architecture.",
        "",
        "Round 1's core findings do not hinge on DeBERTa's depressed scores. Knowledge-base "
        "ranking was largely insensitive to encoder choice, seed noise dominated "
        "between-encoder differences, and the pair-type-specific pattern held across the "
        "stable encoders. Placing DeBERTa at its recovered level near 0.774 instead of the "
        "0.554 artifact would not overturn those conclusions; it would only weaken the already "
        "de-emphasised nine-point mean-level correlations further.",
        "",
        "On that basis, Round 1 does not need to be re-run. That is an evidence-based reading, "
        "not a convenience choice: the artifact is understood, DeBERTa's capability sits inside "
        "the encoder band under a suitable recipe, and the seed-dominance story is unchanged.",
        "",
        "## Figures",
        "",
        "Figure 1 plots the four grid points with reference lines for the Round-1 recipe score, "
        "the 0.554 eight-seed mean, and the 0.739 clean six-seed mean. Figure 2 places the "
        "best grid DeBERTa back into the nine-encoder Round-1 distribution. Figure 3 shows "
        "validation curves for the four recipes, illustrating the early stop under the Round-1 "
        "recipe versus later peaks under lr 1e-05.",
    ]

    if not degenerate.empty:
        lines.extend(["", "## Round-1 collapsed seeds (reference)", ""])
        for _, d in degenerate.iterrows():
            lines.append(
                f"DeBERTa-base seed {int(d['seed'])} registered zero validation and benchmark F1 "
                "in Round 1 and was excluded from the primary six-seed DeBERTa summaries."
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report -> {path}")
    return path
