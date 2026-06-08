"""Step 2: benchmark vs KB timing along per-epoch trajectories (focus encoders only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    EPOCH_KB_CACHE,
    FOCUS_MODEL_IDS,
    R11_EASY_HARD_CSV,
    R11_PER_RUN_CSV,
    TRAIN_SEEDS,
)
from .matrix_io import epoch_checkpoint_dir, list_recoverable_epochs, load_training_meta
from .pool_cache import load_enriched_pool
from .scoring import benchmark_f1_at_checkpoint, kb_metrics_from_scores, score_candidates_at_checkpoint


def _cache_key(model_id: str, seed: int, epoch: int) -> str:
    return f"{model_id}|{seed}|{epoch}"


def _load_scored_cache() -> pd.DataFrame:
    if EPOCH_KB_CACHE.exists():
        return pd.read_csv(EPOCH_KB_CACHE)
    return pd.DataFrame()


def _row_is_scored(row: pd.Series) -> bool:
    return bool(row.get("benchmark_f1_scored")) and bool(row.get("kb_scored"))


def trajectory_from_training_logs(*, rescore_epochs: bool = False) -> pd.DataFrame:
    """
    Val metrics always from training logs (cheap).
    Per-epoch benchmark F1 and KB scored on demand from saved epoch checkpoints
    when rescore_epochs=True (focus encoders only; resumable via epoch_kb_trajectory.csv).
    """
    cached = _load_scored_cache()
    cache_index: dict[str, pd.Series] = {}
    if not cached.empty:
        for _, r in cached.iterrows():
            cache_index[_cache_key(r["model_id"], int(r["seed"]), int(r["epoch"]))] = r

    pool = load_enriched_pool()
    candidates = pool.drop(columns=["subset"], errors="ignore")
    rows: list[dict] = []
    n_scored_this_run = 0
    n_skipped = 0

    for model_id in FOCUS_MODEL_IDS:
        for seed in TRAIN_SEEDS:
            meta = load_training_meta(model_id, seed)
            if not meta:
                continue
            recoverable = set(list_recoverable_epochs(model_id, seed, meta))
            for ep in meta.get("epoch_curve") or []:
                epoch = int(ep["epoch"])
                key = _cache_key(model_id, seed, epoch)
                prior = cache_index.get(key)
                row = {
                    "source": "matrix_per_epoch",
                    "model_id": model_id,
                    "seed": seed,
                    "epoch": epoch,
                    "val_f1": float(ep.get("val_f1", np.nan)),
                    "val_loss": float(ep.get("val_loss", np.nan)),
                    "benchmark_f1": np.nan,
                    "benchmark_f1_scored": False,
                    "kb_scored": False,
                }
                if prior is not None and not rescore_epochs:
                    row.update(prior.to_dict())
                elif prior is not None and _row_is_scored(prior) and not rescore_epochs:
                    row.update(prior.to_dict())
                    n_skipped += 1
                elif prior is not None and _row_is_scored(prior) and rescore_epochs:
                    row.update(prior.to_dict())
                    n_skipped += 1
                elif epoch in recoverable and rescore_epochs:
                    ckpt = epoch_checkpoint_dir(model_id, seed, epoch)
                    if ckpt.exists():
                        print(f"  SCORE epoch {model_id} seed={seed} ep={epoch}")
                        row["benchmark_f1"] = benchmark_f1_at_checkpoint(ckpt)
                        row["benchmark_f1_scored"] = True
                        scores = score_candidates_at_checkpoint(ckpt, candidates)
                        row.update(kb_metrics_from_scores(scores, pool))
                        row["kb_scored"] = True
                        n_scored_this_run += 1
                elif prior is not None:
                    row.update({k: prior[k] for k in prior.index if k in row or k.startswith("kb_")})
                    if _row_is_scored(prior):
                        n_skipped += 1

                rows.append(row)

    traj = pd.DataFrame(rows)
    if not traj.empty:
        EPOCH_KB_CACHE.parent.mkdir(parents=True, exist_ok=True)
        traj.to_csv(EPOCH_KB_CACHE, index=False)
    if rescore_epochs:
        print(f"  Epoch scoring this run: {n_scored_this_run}, skipped (cached): {n_skipped}")
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


def count_epoch_checkpoints_to_score() -> int:
    total = 0
    for model_id in FOCUS_MODEL_IDS:
        for seed in TRAIN_SEEDS:
            total += len(list_recoverable_epochs(model_id, seed))
    return total


def count_scored_epochs() -> int:
    cached = _load_scored_cache()
    if cached.empty:
        return 0
    if "kb_scored" in cached.columns:
        return int(cached["kb_scored"].fillna(False).astype(bool).sum())
    return int(cached["kb_mrr_hard"].notna().sum()) if "kb_mrr_hard" in cached.columns else 0


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
                "Run epoch scoring (--score-epochs-only) to compute per-epoch benchmark F1 "
                "and KB from saved checkpoints for PubMedBERT, RoBERTa, and DistilBERT."
            )
        }
    return {
        "narrative": (
            "Per-epoch benchmark F1 and KB trajectories are available for the three "
            "focus encoders. Compare whether benchmark and KB peak at different epochs "
            "within the same seed."
        )
    }
