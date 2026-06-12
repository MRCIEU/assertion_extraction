"""Part 2: zero-training encoder-property correlation with gene-disease-hard erosion."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import ENCODER_CORRELATION_CSV
from .encoder_properties import PROPERTIES_BY_ID, properties_dataframe


def build_encoder_correlation_table(
    gd_encoder: pd.DataFrame,
    traj: pd.DataFrame,
) -> pd.DataFrame:
    enc = gd_encoder[
        (gd_encoder["slug"] == "gene_disease_hard")
        & (gd_encoder["well_trained_definition"] == "val_f1_best")
    ].copy()
    props = properties_dataframe()
    merged = enc.merge(props[["model_id", "params_millions", "biomedical_pretrain", "property_source"]], on="model_id", how="left")
    merged = merged.rename(columns={"short_name": "encoder_name"})

    bench = (
        traj.groupby("model_id")["benchmark_f1"]
        .mean()
        .reset_index()
        .rename(columns={"benchmark_f1": "mean_benchmark_f1"})
    )
    merged = merged.merge(bench, on="model_id", how="left")
    merged["erosion_magnitude"] = -merged["mean_delta_kb_mrr"]
    return merged


def run_encoder_correlation(gd_encoder: pd.DataFrame, traj: pd.DataFrame) -> dict[str, Any]:
    table = build_encoder_correlation_table(gd_encoder, traj)
    table.to_csv(ENCODER_CORRELATION_CSV, index=False)

    y = table["erosion_magnitude"].astype(float).to_numpy()
    results: list[dict] = []
    for prop, label in [
        ("mean_benchmark_f1", "mean benchmark F1"),
        ("biomedical_pretrain", "biomedical pretraining (binary)"),
        ("params_millions", "parameter count (millions)"),
    ]:
        x = table[prop].astype(float).to_numpy()
        if len(x) < 3:
            rho, p = np.nan, np.nan
        else:
            rho, p = spearmanr(x, y)
        results.append(
            {
                "property": prop,
                "label": label,
                "spearman_rho": float(rho) if rho == rho else np.nan,
                "p_value": float(p) if p == p else np.nan,
                "n_encoders": len(table),
            }
        )
    corr_df = pd.DataFrame(results)

    print("\n=== Part 2: Encoder-property correlation (gene-disease-hard erosion, n=9) ===")
    for _, r in corr_df.iterrows():
        print(
            f"  {r['label']}: Spearman rho={float(r['spearman_rho']):+.3f} "
            f"(p={float(r['p_value']):.3f})"
        )
    strongest = corr_df.loc[corr_df["spearman_rho"].abs().idxmax()]
    print(
        "  Note: nine encoders with correlated families; exploratory only. "
        f"Strongest monotonic association: {strongest['label']} (rho={float(strongest['spearman_rho']):+.3f})."
    )

    return {"table": table, "correlations": corr_df}
