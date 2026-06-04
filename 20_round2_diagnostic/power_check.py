"""Step 3: seed noise vs expected training-amount effect on KB (hard subset)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FOCUS_MODEL_IDS, MODEL_BY_ID


def seed_level_kb_sd(per_run: pd.DataFrame, model_id: str) -> dict[str, float]:
    sub = per_run[per_run["model_id"] == model_id]
    return {
        "kb_mrr_overall_sd": float(sub["kb_mrr_overall"].std(ddof=1)) if len(sub) > 1 else 0.0,
        "kb_mrr_gene_drug_sd": float(sub["kb_mrr_gene_drug"].std(ddof=1)) if len(sub) > 1 else 0.0,
        "kb_mrr_gene_disease_sd": float(sub["kb_mrr_gene_disease"].std(ddof=1))
        if len(sub) > 1
        else 0.0,
        "n_seeds": len(sub),
    }


def estimate_training_effect(traj: pd.DataFrame) -> pd.DataFrame:
    """
    Effect size: mean absolute delta between val_loss-best and val_f1-best on sweep seed 42;
    fallback: spread of main-matrix val_f1-best KB hard across seeds as upper bound on lever.
    """
    rows: list[dict] = []
    sweep = traj[traj["source"] == "round1_sweep_recipe_match"]
    main = traj[traj["source"] == "round1_main"]

    for model_id in FOCUS_MODEL_IDS:
        sw = sweep[sweep["model_id"] == model_id]
        effect_hard = effect_overall = np.nan
        effect_source = "unavailable"
        if len(sw) >= 2:
            loss = sw[sw["trajectory_point"] == "val_loss_best"]
            f1 = sw[sw["trajectory_point"] == "val_f1_best"]
            if not loss.empty and not f1.empty:
                effect_hard = abs(float(f1.iloc[0]["kb_mrr_hard"]) - float(loss.iloc[0]["kb_mrr_hard"]))
                effect_overall = abs(
                    float(f1.iloc[0]["kb_mrr_overall"]) - float(loss.iloc[0]["kb_mrr_overall"])
                )
                effect_source = "sweep_two_checkpoint_seed42"

        m = main[main["model_id"] == model_id]
        kb_hard_sd = float(m["kb_mrr_hard"].std(ddof=1)) if len(m) > 1 and "kb_mrr_hard" in m else np.nan
        if np.isnan(kb_hard_sd) and "kb_mrr_overall" in m.columns:
            kb_hard_sd = float(m["kb_mrr_overall"].std(ddof=1)) if len(m) > 1 else np.nan

        n_seeds_hypothesis = 10
        z_approx = 2.0
        detectable_hard = (
            z_approx * kb_hard_sd / np.sqrt(n_seeds_hypothesis) if kb_hard_sd and not np.isnan(kb_hard_sd) else np.nan
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
                "estimated_training_effect_overall": effect_overall,
                "effect_estimate_source": effect_source,
                "assumed_n_seeds_per_cell": n_seeds_hypothesis,
                "approx_detectable_effect_hard": detectable_hard,
                "effect_clears_detectable_band": clears,
            }
        )
    return pd.DataFrame(rows)


def build_power_check(main_traj: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    est = estimate_training_effect(traj)
    for model_id in FOCUS_MODEL_IDS:
        m = main_traj[main_traj["model_id"] == model_id]
        kb_hard_sd = float(m["kb_mrr_hard"].std(ddof=1)) if len(m) > 1 else np.nan
        er = est[est["model_id"] == model_id].iloc[0].to_dict()
        er["kb_mrr_hard_sd_at_val_f1_ckpt"] = kb_hard_sd
        rows.append(er)
    return pd.DataFrame(rows)
