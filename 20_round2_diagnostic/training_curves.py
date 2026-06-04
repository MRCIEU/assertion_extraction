"""Step 1: validation and benchmark curves from training_log.json (read-only)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import MATRIX_CKPT_DIR, MODELS, TRAIN_SEEDS


def load_epoch_curves() -> pd.DataFrame:
    rows: list[dict] = []
    for spec in MODELS:
        for seed in TRAIN_SEEDS:
            log_path = MATRIX_CKPT_DIR / spec.model_id / f"seed_{seed}" / "training_log.json"
            if not log_path.exists():
                continue
            meta = json.loads(log_path.read_text(encoding="utf-8"))
            for ep in meta.get("epoch_curve") or []:
                rows.append(
                    {
                        "model_id": spec.model_id,
                        "seed": seed,
                        "epoch": int(ep["epoch"]),
                        "val_loss": float(ep["val_loss"]),
                        "val_f1": float(ep["val_f1"]),
                        "train_loss": float(ep.get("train_loss", np.nan)),
                        "benchmark_f1": float(ep.get("benchmark_f1", np.nan)),
                    }
                )
    return pd.DataFrame(rows)


def _first_val_loss_rise_epoch(curve: pd.DataFrame) -> int | None:
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
                "n_seeds": grp["seed"].nunique(),
                "mean_peak_val_f1_epoch": float(np.mean(peak_epochs)),
                "median_peak_val_f1_epoch": float(np.median(peak_epochs)),
                "pct_peak_epoch_eq_1": float(np.mean(np.array(peak_epochs) == 1)),
                "mean_val_loss_rise_epoch": float(np.mean(rise_epochs)) if rise_epochs else np.nan,
                "mean_plateau_width_epochs": float(np.mean(plateau_widths)),
                "mean_benchmark_f1_peak": float(grp.groupby("seed")["benchmark_f1"].max().mean()),
            }
        )
    return pd.DataFrame(rows)


def encoder_mean_curves(curves: pd.DataFrame) -> pd.DataFrame:
    return (
        curves.groupby(["model_id", "epoch"])[["val_loss", "val_f1", "benchmark_f1"]]
        .mean()
        .reset_index()
    )


def describe_curve_shape(summary: pd.DataFrame) -> str:
    peak1 = float(summary["pct_peak_epoch_eq_1"].mean())
    med_peak = float(summary["median_peak_val_f1_epoch"].median())
    mean_plateau = float(summary["mean_plateau_width_epochs"].mean())
    lines = [
        "Per-epoch checkpoints and benchmark F1 are available from step-2 training logs.",
        f"Across encoders, the median seed's val_f1 peak occurs around epoch {med_peak:.1f} "
        f"(mean plateau width near peak val_f1 about {mean_plateau:.1f} epochs).",
    ]
    if peak1 > 0.3:
        lines.append(
            f"For {peak1*100:.0f}% of encoder-seed runs (averaged per encoder), val_f1 is already "
            "highest at epoch 1."
        )
    else:
        lines.append(
            "Val_f1-best epoch is not universally epoch 1; a later-epoch well-trained region exists "
            "for many runs."
        )
    return " ".join(lines)
