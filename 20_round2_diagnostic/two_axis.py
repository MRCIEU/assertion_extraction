"""Step 2: benchmark vs KB timing along per-epoch trajectories (focus encoders only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    EPOCH_KB_CACHE,
    FOCUS_MODEL_IDS,
    MATRIX_CKPT_DIR,
    R11_EASY_HARD_CSV,
    R11_PER_RUN_CSV,
    TRAIN_SEEDS,
)
from .pool_cache import load_enriched_pool
from .scoring import benchmark_f1_at_checkpoint, kb_metrics_from_scores, score_candidates_at_checkpoint


def _epoch_ckpt(model_id: str, seed: int, epoch: int) -> Path:
    return MATRIX_CKPT_DIR / model_id / f"seed_{seed}" / "epochs" / f"epoch_{epoch:02d}"


def trajectory_from_training_logs(*, rescore_epochs: bool = False) -> pd.DataFrame:
    """
    Val metrics always from training_log.json (cheap).
    Per-epoch benchmark F1 and KB scored on demand from saved epoch checkpoints
    when rescore_epochs=True (focus encoders only).
    """
    if EPOCH_KB_CACHE.exists() and not rescore_epochs:
        cached = pd.read_csv(EPOCH_KB_CACHE)
        if not cached.empty and cached["benchmark_f1"].notna().any():
            return cached

    pool = load_enriched_pool()
    candidates = pool.drop(columns=["subset"], errors="ignore")
    rows: list[dict] = []

    for model_id in FOCUS_MODEL_IDS:
        for seed in TRAIN_SEEDS:
            log_path = MATRIX_CKPT_DIR / model_id / f"seed_{seed}" / "training_log.json"
            if not log_path.exists():
                continue
            meta = json.loads(log_path.read_text(encoding="utf-8"))
            for ep in meta.get("epoch_curve") or []:
                epoch = int(ep["epoch"])
                row = {
                    "source": "matrix_per_epoch",
                    "model_id": model_id,
                    "seed": seed,
                    "epoch": epoch,
                    "val_f1": float(ep.get("val_f1", np.nan)),
                    "val_loss": float(ep.get("val_loss", np.nan)),
                    "benchmark_f1": np.nan,
                }
                if rescore_epochs:
                    ckpt = _epoch_ckpt(model_id, seed, epoch)
                    if ckpt.exists():
                        row["benchmark_f1"] = benchmark_f1_at_checkpoint(ckpt)
                        scores = score_candidates_at_checkpoint(ckpt, candidates)
                        row.update(kb_metrics_from_scores(scores, pool))
                rows.append(row)

    traj = pd.DataFrame(rows)
    if rescore_epochs and not traj.empty:
        EPOCH_KB_CACHE.parent.mkdir(parents=True, exist_ok=True)
        traj.to_csv(EPOCH_KB_CACHE, index=False)
    return traj


def trajectory_best_point_from_r11() -> pd.DataFrame:
    """Single val_f1-best point per run from Round 1 analysis CSV."""
    if not R11_PER_RUN_CSV.exists():
        return pd.DataFrame()
    per_run = pd.read_csv(R11_PER_RUN_CSV)
    easy_hard = pd.read_csv(R11_EASY_HARD_CSV) if R11_EASY_HARD_CSV.exists() else None
    rows: list[dict] = []
    for model_id in FOCUS_MODEL_IDS:
        sub = per_run[per_run["model_id"] == model_id]
        for _, r in sub.iterrows():
            run_id = r["run_id"]
            kb_hard = kb_easy = np.nan
            if easy_hard is not None:
                eh = easy_hard[easy_hard["run_id"] == run_id]
                er = eh[eh["subset"] == "easy_co_sentence"]
                hr = eh[eh["subset"] == "hard_cross_sentence"]
                if not er.empty:
                    kb_easy = float(er.iloc[0]["mrr"])
                if not hr.empty:
                    kb_hard = float(hr.iloc[0]["mrr"])
            rows.append(
                {
                    "source": "r11_best",
                    "model_id": model_id,
                    "seed": int(r["seed"]),
                    "epoch": int(r.get("best_epoch_val_f1", 0) or 0),
                    "benchmark_f1": float(r["benchmark_f1"]),
                    "kb_mrr_overall": float(r.get("kb_mrr_overall", np.nan)),
                    "kb_mrr_hard": float(kb_hard) if not np.isnan(kb_hard) else np.nan,
                    "kb_mrr_easy": float(kb_easy) if not np.isnan(kb_easy) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_two_axis_trajectory(*, rescore_epochs: bool = False) -> pd.DataFrame:
    parts = [trajectory_best_point_from_r11()]
    per_epoch = trajectory_from_training_logs(rescore_epochs=rescore_epochs)
    if not per_epoch.empty:
        parts.append(per_epoch)
    return pd.concat([p for p in parts if not p.empty], ignore_index=True)


def summarize_timing(traj: pd.DataFrame) -> dict[str, str]:
    if traj.empty:
        return {"narrative": "No trajectory data available yet."}
    pe = traj[traj["source"] == "matrix_per_epoch"]
    if pe.empty:
        return {"narrative": "No per-epoch training logs found for focus encoders."}
    if "kb_mrr_hard" not in pe.columns or pe["kb_mrr_hard"].isna().all():
        return {
            "narrative": (
                "Validation curves are available from step-2 logs. "
                "Run with --rescore-epochs to compute per-epoch benchmark F1 and KB "
                "from saved checkpoints for PubMedBERT, RoBERTa, and DistilBERT."
            )
        }
    return {
        "narrative": (
            "Per-epoch benchmark F1 and KB trajectories are available for the three "
            "focus encoders. Compare whether benchmark and KB peak at different epochs "
            "within the same seed."
        )
    }
