"""Dedicated RoBERTa vs domain-specialised encoder analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shared.models import MODEL_BY_ID

from .analysis import encoder_summary_seed_bootstrap, filter_clean_runs, variance_components_table

DOMAIN_MODEL_IDS = tuple(
    m.model_id for m in MODEL_BY_ID.values() if "domain" in m.architecture.lower()
)


def roberta_gene_disease_analysis(
    per_run: pd.DataFrame,
    easy_hard: pd.DataFrame,
) -> dict[str, Any]:
    """
    If general RoBERTa ranks at or above domain encoders on KB gene-disease, characterise it.
    If not, report as recipe-dependent / non-robust.
    """
    clean = filter_clean_runs(per_run)
    enc = encoder_summary_seed_bootstrap(clean)

    gd_col = "kb_mrr_gene_disease_mean"
    roberta_row = enc[enc["model_id"] == "roberta_base"]
    domain = enc[enc["model_id"].isin(DOMAIN_MODEL_IDS)]

    if roberta_row.empty or domain.empty:
        return {"pattern_holds": False, "reason": "insufficient data"}

    roberta_gd = float(roberta_row[gd_col].iloc[0])
    domain_max = float(domain[gd_col].max())
    domain_median = float(domain[gd_col].median())
    pattern_holds = roberta_gd >= domain_max - 1e-9

    var = variance_components_table(clean)
    gd_var = var[var["metric"] == "kb_mrr_gene_disease"]
    seed_share = float(gd_var["seed_variance_share"].iloc[0]) if not gd_var.empty else np.nan

    margin_vs_median = roberta_gd - domain_median

    hard = easy_hard[
        (easy_hard["model_id"] == "roberta_base") & (easy_hard["subset"] == "hard_cross_sentence")
    ]
    hard_mrr = float(hard["mrr"].mean()) if not hard.empty else np.nan

    domain_hard = easy_hard[
        (easy_hard["subset"] == "hard_cross_sentence")
        & (easy_hard["model_id"].isin(DOMAIN_MODEL_IDS))
    ]
    domain_hard_mean = (
        float(domain_hard.groupby("model_id")["mrr"].mean().mean()) if not domain_hard.empty else np.nan
    )

    return {
        "pattern_holds": pattern_holds,
        "roberta_gene_disease_mrr": roberta_gd,
        "domain_best_gene_disease_mrr": domain_max,
        "domain_median_gene_disease_mrr": domain_median,
        "margin_vs_domain_median": margin_vs_median,
        "seed_variance_share_gene_disease": seed_share,
        "roberta_hard_mrr_mean": hard_mrr,
        "domain_hard_mrr_mean": domain_hard_mean,
        "conclusion_if_holds": (
            "RoBERTa sits at or above the best domain-specialised encoder on gene-disease KB "
            f"ranking ({roberta_gd:.3f} vs domain best {domain_max:.3f}). The margin over the "
            f"domain median is {margin_vs_median:.3f}, while seed noise accounts for "
            f"{seed_share:.0%} of variance on this metric. On the hard cross-sentence subset "
            f"RoBERTa averages {hard_mrr:.3f} MRR compared with {domain_hard_mean:.3f} across "
            "domain encoders."
        ),
        "conclusion_if_not": (
            "Under the chosen recipe, RoBERTa does not exceed the top domain-specialised "
            f"encoder on gene-disease KB ranking ({roberta_gd:.3f} vs {domain_max:.3f}). "
            "The earlier counterintuitive pattern should be treated as recipe-dependent and "
            "not robust to the new training setup."
        ),
    }


def roberta_report_paragraph(result: dict[str, Any]) -> str:
    if result.get("pattern_holds"):
        return result["conclusion_if_holds"]
    return result.get("conclusion_if_not", result.get("reason", "Analysis inconclusive."))
