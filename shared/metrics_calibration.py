"""Calibration metrics against CIViC inclusion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import ECE_N_BINS, POSITIVE_FRACTION_PRIOR, SAMPLING_SEED


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = ECE_N_BINS) -> float:
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


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = ECE_N_BINS) -> pd.DataFrame:
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


def calibration_for_scores(df: pd.DataFrame, label: str) -> dict:
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
    for name, val in [
        ("constant_low", 0.05),
        ("constant_high", 0.95),
        ("constant_prior", POSITIVE_FRACTION_PRIOR),
        ("constant_0.5", 0.5),
    ]:
        sub = template.copy()
        sub["score"] = val
        out[name] = sub
    rand = template.copy()
    rand["score"] = rng.random(len(rand))
    out["random"] = rand
    return out
