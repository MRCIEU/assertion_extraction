"""Step 3: seed noise vs expected training-amount effect on KB (hard subset)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FOCUS_MODEL_IDS, MODEL_BY_ID, R11_EASY_HARD_CSV, R11_VARIANCE_CSV


def _round1_seed_noise() -> dict[str, float]:
    """Pull mean within-encoder SD from folder-11 variance-components table."""
    out = {"kb_gene_drug_sd": np.nan, "kb_gene_disease_sd": np.nan, "kb_gene_drug_seed_share": np.nan}
    if not R11_VARIANCE_CSV.exists():
        return out
    vc = pd.read_csv(R11_VARIANCE_CSV)
    gd = vc[vc["metric"] == "kb_mrr_gene_drug"]
    gdis = vc[vc["metric"] == "kb_mrr_gene_disease"]
    if not gd.empty:
        out["kb_gene_drug_sd"] = float(gd.iloc[0]["mean_within_encoder_sd"])
        out["kb_gene_drug_seed_share"] = float(gd.iloc[0]["seed_variance_share"])
    if not gdis.empty:
        out["kb_gene_disease_sd"] = float(gdis.iloc[0]["mean_within_encoder_sd"])
    return out


def _focus_hard_sd_at_best() -> dict[str, float]:
    """Seed-level hard-subset MRR SD at val_f1-best checkpoint (focus encoders, folder 11)."""
    if not R11_EASY_HARD_CSV.exists():
        return {}
    eh = pd.read_csv(R11_EASY_HARD_CSV)
    hard = eh[
        (eh["model_id"].isin(FOCUS_MODEL_IDS))
        & (eh["subset"] == "hard_cross_sentence")
        & (eh["model_id"] != "distance_ranker")
    ]
    return {mid: float(hard[hard["model_id"] == mid]["mrr"].std(ddof=1)) for mid in FOCUS_MODEL_IDS}


def estimate_training_effect(traj: pd.DataFrame) -> pd.DataFrame:
    """
    Effect size from per-epoch trajectory when KB columns exist:
    max minus min KB hard MRR across epochs within each seed.
    Seed noise from Round 1 variance components and focus-encoder hard-subset SD.
    """
    rows: list[dict] = []
    per_epoch = traj[traj["source"] == "matrix_per_epoch"]
    best = traj[traj["source"] == "r11_best"]
    r1_noise = _round1_seed_noise()
    hard_sds = _focus_hard_sd_at_best()

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
                effect_overall = float(np.mean(deltas))
                effect_source = "per_epoch_kb_trajectory"

        kb_hard_sd = hard_sds.get(model_id, np.nan)
        if np.isnan(kb_hard_sd) and model_id in best["model_id"].values:
            m = best[best["model_id"] == model_id]
            if len(m) > 1 and "kb_mrr_hard" in m.columns:
                kb_hard_sd = float(m["kb_mrr_hard"].std(ddof=1))

        n_seeds_hypothesis = 10
        z_approx = 2.0
        detectable_hard = (
            z_approx * kb_hard_sd / np.sqrt(n_seeds_hypothesis)
            if kb_hard_sd and not np.isnan(kb_hard_sd)
            else np.nan
        )
        detectable_vs_r1_pool = (
            z_approx * r1_noise["kb_gene_drug_sd"] / np.sqrt(n_seeds_hypothesis)
            if not np.isnan(r1_noise["kb_gene_drug_sd"])
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
                "r1_mean_within_encoder_sd_gene_drug": r1_noise["kb_gene_drug_sd"],
                "r1_seed_variance_share_gene_drug": r1_noise["kb_gene_drug_seed_share"],
                "estimated_training_effect_hard": effect_hard,
                "effect_estimate_source": effect_source,
                "assumed_n_seeds_per_cell": n_seeds_hypothesis,
                "approx_detectable_effect_hard": detectable_hard,
                "approx_detectable_vs_r1_pool_sd": detectable_vs_r1_pool,
                "effect_clears_detectable_band": clears,
            }
        )
    return pd.DataFrame(rows)


def build_power_check(main_traj: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    return estimate_training_effect(traj)
