#!/usr/bin/env python3.11
"""Phase 2C — continuous compute attribution (α̂) per matched_compute/COMMITMENT.md.

Loads seed-wise KB argmax accuracy ``kb_hit_A_setvalued`` from ``phase_b_eval.json``
for PubMedBERT × FT × (T1F@2048, T1F@4096, T2).  Point estimate and paired
bootstrap (B=5000, RNG seed 20260518) follow the signed COMMITMENT text; verdict
mapping is emitted as machine-readable labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from knowledge_grounded_evidence_audit.analysis.phase_d_baselines.analysis.phase_d_rb_extensions import (  # noqa: E501
    kb_from_phase_b_eval,
)


def _find_project_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(12):
        if (p / "fine_tuning_experiments").is_dir() and (p / "report" / "data").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError("Cannot locate project_1 root")


REPO_ROOT = _find_project_root()
RUN_ROOT_DEFAULT = Path(
    "/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b"
)

BOOT_B = 5000
BOOT_RNG = 20260518


def paired_mean_bootstrap_ci(
    diffs: np.ndarray, *, n_boot: int, rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap distribution of the sample mean of paired differences (seed resampling, paired design)."""
    n = int(diffs.shape[0])
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = float(diffs[idx].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_Y_triplet(run_root: Path, seed_int: int) -> tuple[float | None, float | None, float | None]:
    """Returns (Y_T1F2048, Y_T1F4096, Y_T2) for one seed index."""
    s = f"{seed_int:02d}"
    y2048 = kb_from_phase_b_eval(run_root / f"PB_PB_FT_T1F_s{s}" / "eval" / "phase_b_eval.json")
    y4096 = kb_from_phase_b_eval(run_root / f"PB_PB_FT_T1F4096_s{s}" / "eval" / "phase_b_eval.json")
    yt2 = kb_from_phase_b_eval(run_root / f"PB_PB_FT_T2_s{s}" / "eval" / "phase_b_eval.json")
    return y2048, y4096, yt2


def verdict(alpha_hat: float, lo: float, hi: float) -> str:
    width = hi - lo
    if alpha_hat > 1.0:
        return "unexpected_alpha_gt_1_halt_reframing"
    if width >= 0.50:
        return "mixed_attribution_uncertain"
    if alpha_hat < 0.20 and hi < 0.30:
        return "content_dominant"
    if alpha_hat > 0.80 and lo > 0.70:
        return "compute_dominant"
    if 0.20 <= alpha_hat <= 0.80 and width < 0.50:
        return "mixed_report_alpha_hat"
    return "mixed_review_manually"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=RUN_ROOT_DEFAULT)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT
        / "knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/phase_d_alpha_attribution.json",
    )
    args = ap.parse_args()
    run_root: Path = args.runs_root

    ys2048: list[float] = []
    ys4096: list[float] = []
    yst2: list[float] = []
    missing: dict[str, list[int]] = {"T1F2048": [], "T1F4096": [], "T2": []}

    for s in range(1, 21):
        a, b, c = load_Y_triplet(run_root, s)
        if a is None:
            missing["T1F2048"].append(s)
        if b is None:
            missing["T1F4096"].append(s)
        if c is None:
            missing["T2"].append(s)
        if a is not None and b is not None and c is not None:
            ys2048.append(a)
            ys4096.append(b)
            yst2.append(c)

    n_ok = len(ys2048)
    ready = n_ok == 20 and not any(missing.values())

    out: dict[str, Any] = {
        "runs_root": str(run_root.resolve()),
        "n_seeds_with_complete_triplet": n_ok,
        "missing_eval_by_arm": missing,
        "ready_for_commitment_analysis": ready,
    }

    if not ready:
        out["note"] = (
            "Incomplete: run Phase B eval for all PB_PB_FT_T1F_s*, T1F4096_s*, T2_s* "
            "(pb_ft_t1f4096_eval_array.sbatch for the 4096 cell)."
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.out} (incomplete)")
        return

    y2048 = np.asarray(ys2048, dtype=np.float64)
    y4096 = np.asarray(ys4096, dtype=np.float64)
    yt2 = np.asarray(yst2, dtype=np.float64)
    d_comp = y4096 - y2048
    d_gap = yt2 - y2048
    mean_comp = float(d_comp.mean())
    mean_gap = float(d_gap.mean())
    alpha_hat = float(mean_comp / mean_gap) if mean_gap != 0 else float("nan")

    rng = np.random.default_rng(BOOT_RNG)
    alphas: list[float] = []
    discarded = 0
    for _ in range(BOOT_B):
        idx = rng.integers(0, 20, size=20)
        mc = float(d_comp[idx].mean())
        mg = float(d_gap[idx].mean())
        if mg == 0.0:
            discarded += 1
            continue
        alphas.append(mc / mg)

    discard_rate = discarded / BOOT_B
    if not alphas:
        ci_lo = ci_hi = float("nan")
    else:
        arr = np.asarray(alphas, dtype=np.float64)
        ci_lo, ci_hi = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    v = verdict(alpha_hat, ci_lo, ci_hi)

    # --- Paired contrasts on KB argmax (162-set), seeds 1..20 (§S13.3) ---
    d_content = yt2 - y4096  # T2 minus T1F-4096: oncology staging at matched compute
    delta_content = float(d_content.mean())
    sd_content_sample = float(d_content.std(ddof=1))  # descriptive SD of paired differences
    sd_content_pstdev = float(d_content.std(ddof=0))
    # Independent RNG streams from master seed 20260518 (content bootstrap per user spec)
    _spawn = np.random.SeedSequence(BOOT_RNG).spawn(3)
    rng_content_boot = np.random.default_rng(_spawn[1])
    rng_compute_boot = np.random.default_rng(_spawn[2])
    ci_c_lo, ci_c_hi = paired_mean_bootstrap_ci(
        d_content, n_boot=BOOT_B, rng=rng_content_boot,
    )
    tt_c = stats.ttest_rel(yt2, y4096, nan_policy="omit")

    ci_comp_lo, ci_comp_hi = paired_mean_bootstrap_ci(
        d_comp, n_boot=BOOT_B, rng=rng_compute_boot,
    )
    tt_comp = stats.ttest_rel(y4096, y2048, nan_policy="omit")
    sd_comp_sample = float(d_comp.std(ddof=1))
    sd_comp_pstdev = float(d_comp.std(ddof=0))

    out.update({
        "definitions": {
            "Y_metric": "kb_hit_A_setvalued from phase_b_eval.json kb_surface",
            "d_comp": "Y_T1F4096 - Y_T1F2048",
            "d_gap": "Y_T2 - Y_T1F2048",
            "alpha_hat": "mean(d_comp) / mean(d_gap)",
        },
        "point_estimate": {
            "mean_Y_T1F2048": float(y2048.mean()),
            "mean_Y_T1F4096": float(y4096.mean()),
            "mean_Y_T2": float(yt2.mean()),
            "mean_d_comp": mean_comp,
            "mean_d_gap": mean_gap,
            "alpha_hat": alpha_hat,
        },
        "bootstrap": {
            "B": BOOT_B,
            "rng_seed": BOOT_RNG,
            "n_discarded_zero_mean_d_gap": discarded,
            "discard_rate": discard_rate,
            "alpha_ci_95_pct": {"lower": ci_lo, "upper": ci_hi, "width": ci_hi - ci_lo},
        },
        "compute_only_contrast": {
            "label": "Extra multi-corpus flat-T1 steps: T1F-4096 minus T1F-2048 (paired, 162-set KB argmax)",
            "delta_compute_mean_t1f4096_minus_t1f2048": mean_comp,
            "paired_difference_per_seed_sample": {
                "seeds_1_to_20": [int(i) for i in range(1, 21)],
                "d_compute_s": [float(x) for x in d_comp.tolist()],
            },
            "paired_bootstrap_mean_95_ci": {
                "B": BOOT_B,
                "master_seed": BOOT_RNG,
                "spawn_index": 2,
                "method": "resample seeds with replacement; mean of paired differences",
                "lower": ci_comp_lo,
                "upper": ci_comp_hi,
            },
            "paired_ttest_rel_scipy": {
                "alternative": "two-sided",
                "statistic": float(tt_comp.statistic),
                "pvalue": float(tt_comp.pvalue),
                "note": "descriptive post-hoc; compares T1F4096 vs T1F2048 per seed",
            },
            "paired_sd_of_differences_sample_ddof1": sd_comp_sample,
            "paired_sd_of_differences_population_pstdev": sd_comp_pstdev,
        },
        "content_only_contrast": {
            "label": "Oncology-projected staging vs multi-corpus at matched compute: T2 minus T1F-4096",
            "delta_content_mean_t2_minus_t1f4096": delta_content,
            "paired_difference_per_seed_sample": {
                "seeds_1_to_20": [int(i) for i in range(1, 21)],
                "d_content_s": [float(x) for x in d_content.tolist()],
            },
            "paired_bootstrap_mean_95_ci": {
                "B": BOOT_B,
                "master_seed": BOOT_RNG,
                "spawn_index": 1,
                "method": "resample seeds with replacement; mean of paired differences",
                "lower": ci_c_lo,
                "upper": ci_c_hi,
            },
            "paired_ttest_rel_scipy": {
                "alternative": "two-sided",
                "statistic": float(tt_c.statistic),
                "pvalue": float(tt_c.pvalue),
                "note": "descriptive post-hoc only",
            },
            "paired_sd_of_differences_sample_ddof1": sd_content_sample,
            "paired_sd_of_differences_population_pstdev": sd_content_pstdev,
        },
        "verdict_rule_table_reference": "matched_compute/COMMITMENT.md",
        "verdict": v,
    })

    if discard_rate > 0.10:
        out["warning"] = (
            f">{100*discard_rate:.1f}% bootstrap replicates discarded (mean d_gap* = 0); "
            "COMMITMENT requests instability halt."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"wrote {args.out} verdict={v} alpha_hat={alpha_hat:.4f} CI=[{ci_lo:.4f},{ci_hi:.4f}] "
        f"d_compute={mean_comp:.4f} d_compute_CI=[{ci_comp_lo:.4f},{ci_comp_hi:.4f}] "
        f"d_content={delta_content:.4f} d_content_CI=[{ci_c_lo:.4f},{ci_c_hi:.4f}]"
    )


if __name__ == "__main__":
    main()
