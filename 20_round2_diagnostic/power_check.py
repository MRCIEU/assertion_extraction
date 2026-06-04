"""Step 3: seed noise vs expected training-amount effect on KB (hard subset)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FOCUS_MODEL_IDS, MODEL_BY_ID


def estimate_training_effect(traj: pd.DataFrame) -> pd.DataFrame:
    """
    Effect size from per-epoch trajectory when KB columns exist:
    max minus min KB hard MRR across epochs within each seed.
    """
    rows: list[dict] = []
    per_epoch = traj[traj["source"] == "matrix_per_epoch"]
    best = traj[traj["source"] == "r11_best"]

    for model_id in FOCUS_MODEL_IDS:
        effect_hard = effect_overall = np.nan
        effect_source = "unavailable"

        pe = per_epoch[per_epoch["model_id"] == model_id]
        if not pe.empty and "kb_mrr_hard" in pe.columns and pe["kb_mrr_hard"].notna().any():
            deltas: list[float] = []
            for seed, sub in pe.groupby("seed"):
                if sub["kb_mrr_hard"].notna().sum() >= 2:
                    deltas.append(float(sub["kb_mrr_hard"].max() - sub["kb_mrr_hard"].min()))
            if deltas:
                effect_hard = float(np.mean(deltas))
                effect_source = "per_epoch_kb_trajectory"

        m = best[best["model_id"] == model_id]
        kb_hard_sd = float(m["kb_mrr_hard"].std(ddof=1)) if len(m) > 1 and "kb_mrr_hard" in m else np.nan
        if np.isnan(kb_hard_sd) and "kb_mrr_overall" in m.columns and len(m) > 1:
            kb_hard_sd = float(m["kb_mrr_overall"].std(ddof=1))

        n_seeds_hypothesis = 10
        z_approx = 2.0
        detectable_hard = (
            z_approx * kb_hard_sd / np.sqrt(n_seeds_hypothesis)
            if kb_hard_sd and not np.isnan(kb_hard_sd)
            else np.nan
        )
        clears = (
            effect_hard >= detectable_hard
            if not np.isnan(effect_hard) and not np.isnan(detectable_hard)
            else np.nan
        )

        rows.append(
            {
                "model_id": model_id,
                "short_name": MODEL_BY_ID[model_id].short_name,
                "kb_mrr_hard_sd_at_val_f1_ckpt": kb_hard_sd,
                "estimated_training_effect_hard": effect_hard,
                "effect_estimate_source": effect_source,
                "assumed_n_seeds_per_cell": n_seeds_hypothesis,
                "approx_detectable_effect_hard": detectable_hard,
                "effect_clears_detectable_band": clears,
            }
        )
    return pd.DataFrame(rows)


def build_power_check(main_traj: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    return estimate_training_effect(traj)
