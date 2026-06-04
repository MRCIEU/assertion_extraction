"""Step 1: validation training-curve shape from Round 1 metadata (read-only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    COLLAPSED_DEBERTA_SEEDS,
    MODELS,
    R1_CHECKPOINTS,
    TRAIN_SEEDS,
    TRAINING_STRATEGY,
)


def _is_clean(model_id: str, seed: int) -> bool:
    return not (model_id == "deberta_base" and seed in COLLAPSED_DEBERTA_SEEDS)


def load_epoch_curves() -> pd.DataFrame:
    rows: list[dict] = []
    for spec in MODELS:
        for seed in TRAIN_SEEDS:
            if not _is_clean(spec.model_id, seed):
                continue
            meta_path = R1_CHECKPOINTS / spec.model_id / f"seed_{seed}" / "10_train_metadata.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("training_strategy") != TRAINING_STRATEGY:
                continue
            for ep in meta.get("epoch_curve") or []:
                rows.append(
                    {
                        "model_id": spec.model_id,
                        "seed": seed,
                        "epoch": int(ep["epoch"]),
                        "val_loss": float(ep["val_loss"]),
                        "val_f1": float(ep["val_f1"]),
                        "train_loss": float(ep.get("train_loss", np.nan)),
                    }
                )
    return pd.DataFrame(rows)


def _first_val_loss_rise_epoch(curve: pd.DataFrame) -> int | None:
    """First epoch after the global val_loss minimum where loss exceeds the minimum."""
    if curve.empty:
        return None
    min_loss = curve["val_loss"].min()
    min_ep = int(curve.loc[curve["val_loss"].idxmin(), "epoch"])
    after = curve[curve["epoch"] > min_ep].sort_values("epoch")
    for _, r in after.iterrows():
        if r["val_loss"] > min_loss:
            return int(r["epoch"])
    return None


def _plateau_width_epochs(curve: pd.DataFrame, tol: float = 0.01) -> float:
    if curve.empty:
        return 0.0
    peak = curve["val_f1"].max()
    near = curve[curve["val_f1"] >= peak - tol]
    if near.empty:
        return 0.0
    return float(near["epoch"].max() - near["epoch"].min())


def training_curve_summary(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model_id, grp in curves.groupby("model_id"):
        peak_epochs: list[int] = []
        rise_epochs: list[int] = []
        plateau_widths: list[float] = []
        for seed, sub in grp.groupby("seed"):
            sub = sub.sort_values("epoch")
            peak_epochs.append(int(sub.loc[sub["val_f1"].idxmax(), "epoch"]))
            r = _first_val_loss_rise_epoch(sub)
            if r is not None:
                rise_epochs.append(r)
            plateau_widths.append(_plateau_width_epochs(sub))

        rows.append(
            {
                "model_id": model_id,
                "n_clean_seeds": grp["seed"].nunique(),
                "mean_peak_val_f1_epoch": float(np.mean(peak_epochs)),
                "median_peak_val_f1_epoch": float(np.median(peak_epochs)),
                "pct_peak_epoch_eq_1": float(np.mean(np.array(peak_epochs) == 1)),
                "mean_val_loss_rise_epoch": float(np.mean(rise_epochs)) if rise_epochs else np.nan,
                "mean_plateau_width_epochs": float(np.mean(plateau_widths)),
                "mean_val_f1_at_epoch_1": float(
                    grp[grp["epoch"] == 1].groupby("seed")["val_f1"].first().mean()
                ),
                "mean_val_f1_peak": float(grp.groupby("seed")["val_f1"].max().mean()),
            }
        )
    return pd.DataFrame(rows)


def encoder_mean_curves(curves: pd.DataFrame) -> pd.DataFrame:
    return (
        curves.groupby(["model_id", "epoch"])[["val_loss", "val_f1"]]
        .mean()
        .reset_index()
    )


def describe_curve_shape(summary: pd.DataFrame) -> str:
    peak1 = float(summary["pct_peak_epoch_eq_1"].mean())
    med_peak = float(summary["median_peak_val_f1_epoch"].median())
    mean_plateau = float(summary["mean_plateau_width_epochs"].mean())
    lines = [
        "Validation curves are logged for every epoch trained; saved weights are only at val_f1-best.",
        f"Across encoders, the median seed's val_f1 peak occurs around epoch {med_peak:.1f} "
        f"(mean plateau width near peak val_f1 about {mean_plateau:.1f} epochs).",
    ]
    if peak1 > 0.3:
        lines.append(
            f"For {peak1*100:.0f}% of encoder-seed runs (averaged per encoder), val_f1 is already "
            "highest at epoch 1, so under-trained and peak-validation can coincide for some runs."
        )
    else:
        lines.append(
            "Val_f1-best epoch is not universally epoch 1; a later-epoch well-trained region exists "
            "for many runs, though val_loss often rises shortly after the loss minimum."
        )
    if mean_plateau < 0.5:
        lines.append(
            "Plateaus near peak val_f1 are narrow on validation, so well-trained and degrading "
            "validation scores can sit close together in epoch space."
        )
    return " ".join(lines)
