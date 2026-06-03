"""Calibration metrics against CIViC inclusion (is_civic_positive)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ECE_N_BINS, POSITIVE_FRACTION_PRIOR, SAMPLING_SEED


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = ECE_N_BINS) -> float:
    """ECE for binary CIViC-inclusion labels."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = ECE_N_BINS
) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not mask.any():
            rows.append(
                {
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "bin_center": (lo + hi) / 2,
                    "n": 0,
                    "mean_confidence": np.nan,
                    "empirical_rate": np.nan,
                }
            )
            continue
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "bin_center": (lo + hi) / 2,
                "n": int(mask.sum()),
                "mean_confidence": float(y_prob[mask].mean()),
                "empirical_rate": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration_for_scores(df: pd.DataFrame, label: str) -> dict[str, float]:
    y = df["label_civic_curated_positive"].astype(int).values
    p = df["score"].values.astype(float)
    return {
        "model_or_baseline": label,
        "ece": expected_calibration_error(y, p),
        "mean_score": float(p.mean()),
        "positive_rate": float(y.mean()),
        "n": len(df),
    }


def calibration_baselines(template: pd.DataFrame, seed: int = SAMPLING_SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}

    low = template.copy()
    low["score"] = 0.05
    out["constant_low"] = low

    high = template.copy()
    high["score"] = 0.95
    out["constant_high"] = high

    mid = template.copy()
    mid["score"] = POSITIVE_FRACTION_PRIOR
    out["constant_prior"] = mid

    rand = template.copy()
    rand["score"] = rng.random(len(rand))
    out["random"] = rand

    const = template.copy()
    const["score"] = 0.5
    out["constant_0.5"] = const

    return out


def evaluate_calibration(scores_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, sub in scores_df.groupby("model_id"):
        row = calibration_for_scores(sub, model_id)
        row["model_id"] = model_id
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_calibration_baselines(template: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, sub in calibration_baselines(template).items():
        rows.append(calibration_for_scores(sub, name))
    return pd.DataFrame(rows)
