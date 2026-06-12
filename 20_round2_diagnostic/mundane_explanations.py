"""Part 1: rule out mundane explanations for gene-disease erosion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.constants import TRAIN_SEEDS
from shared.inference import load_scores_jsonl
from shared.metrics_ranking import compute_mrr, per_abstract_mrr
from shared.models import MODELS

from .adjudication import WELL_DEF_VAL_F1, _bootstrap_ci, _well_epoch
from .config import (
    POOL_SIZE_BY_ABSTRACT_CSV,
    R11_SCORES_DIR,
    STRATUM_MRR_CACHE,
    TIMING_CLASSIFICATION_CSV,
    TIMING_SUMMARY_CSV,
    POOL_STRATUM_CSV,
    POOL_STRATUM_SUMMARY_CSV,
)
from .epoch_scoring import load_all_epoch_scores
from .matrix_io import epoch_checkpoint_dir, load_training_meta
from .scoring import score_candidates_at_checkpoint

TIMING_LABELS = ("before_best_val", "coincident_best_val", "after_best_val")


def _classify_peak(peak_epoch: int, best_epoch: int) -> str:
    if abs(peak_epoch - best_epoch) <= 1:
        return "coincident_best_val"
    if peak_epoch < best_epoch:
        return "before_best_val"
    return "after_best_val"


def build_kb_peak_timing(traj: pd.DataFrame) -> pd.DataFrame:
    """1a: KB peak epoch vs best-validation-F1 epoch per seed."""
    rows: list[dict] = []
    for spec in MODELS:
        for seed in TRAIN_SEEDS:
            sub = traj[(traj["model_id"] == spec.model_id) & (traj["seed"] == seed)].sort_values("epoch")
            if sub.empty:
                continue
            meta = load_training_meta(spec.model_id, seed)
            if not meta:
                continue
            best_ep = int(meta.get("best_epoch_val_f1") or sub.loc[sub["val_f1"].idxmax(), "epoch"])

            for metric, slug in [
                ("kb_mrr_gene_disease", "gene_disease"),
                ("kb_mrr_gene_disease_hard", "gene_disease_hard"),
            ]:
                if metric not in sub.columns or sub[metric].notna().sum() == 0:
                    continue
                valid = sub[sub[metric].notna()]
                peak_idx = valid[metric].idxmax()
                peak_ep = int(valid.loc[peak_idx, "epoch"])
                peak_val = float(valid.loc[peak_idx, metric])
                best_row = sub[sub["epoch"] == best_ep]
                if best_row.empty:
                    continue
                best_val = float(best_row.iloc[0][metric])
                timing = _classify_peak(peak_ep, best_ep)
                rows.append(
                    {
                        "model_id": spec.model_id,
                        "short_name": spec.short_name,
                        "seed": int(seed),
                        "metric": metric,
                        "slug": slug,
                        "best_val_epoch": best_ep,
                        "kb_peak_epoch": peak_ep,
                        "timing_class": timing,
                        "kb_mrr_at_peak": peak_val,
                        "kb_mrr_at_best_val": best_val,
                        "drop_peak_to_best_val": peak_val - best_val,
                    }
                )
    return pd.DataFrame(rows)


def summarize_timing(timing_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for slug in timing_df["slug"].unique():
        sub = timing_df[timing_df["slug"] == slug]
        n = len(sub)
        for cls in TIMING_LABELS:
            k = sub[sub["timing_class"] == cls]
            rows.append(
                {
                    "slug": slug,
                    "timing_class": cls,
                    "n_seeds": int(len(k)),
                    "frac_seeds": float(len(k) / n) if n else np.nan,
                    "mean_drop_peak_to_best_val": float(k["drop_peak_to_best_val"].mean()) if len(k) else np.nan,
                }
            )
        drops = sub["drop_peak_to_best_val"].astype(float)
        mean, lo, hi = _bootstrap_ci(drops.to_numpy())
        rows.append(
            {
                "slug": slug,
                "timing_class": "all",
                "n_seeds": n,
                "frac_seeds": 1.0,
                "mean_drop_peak_to_best_val": mean,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pd.DataFrame(rows)


def _load_pool_strata() -> dict[str, set[str]]:
    sizes = pd.read_csv(POOL_SIZE_BY_ABSTRACT_CSV)
    gd = sizes[sizes["pair_type"] == "gene-disease"].copy()
    gd = gd[gd["pool_size"] > 0]
    med = float(gd["pool_size"].median())
    gdrug = sizes[(sizes["pair_type"] == "gene-drug") & (sizes["pool_size"] > 0)]
    comparable_max = float(gdrug["pool_size"].quantile(0.75))

    return {
        "small_pool": set(gd.loc[gd["pool_size"] <= med, "pmid"].astype(str)),
        "large_pool": set(gd.loc[gd["pool_size"] > med, "pmid"].astype(str)),
        "comparable_to_gene_drug": set(
            gd.loc[gd["pool_size"] <= comparable_max, "pmid"].astype(str)
        ),
        "median_pool_size": med,
        "comparable_max_pool_size": comparable_max,
    }


def _mrr_for_stratum(scores: pd.DataFrame, pmids: set[str]) -> float:
    sub = scores[(scores["pair_type"] == "gene-disease") & (scores["pmid"].astype(str).isin(pmids))]
    return compute_mrr(sub) if len(sub) else float("nan")


def _best_val_scores(model_id: str, seed: int) -> pd.DataFrame:
    path = R11_SCORES_DIR / model_id / f"seed_{seed}.jsonl"
    return load_scores_jsonl(path)


def _epoch1_scores(model_id: str, seed: int, candidates: pd.DataFrame, cache: dict) -> pd.DataFrame:
    key = f"{model_id}|{seed}"
    if key in cache:
        return cache[key]
    ckpt = epoch_checkpoint_dir(model_id, seed, 1)
    scored = score_candidates_at_checkpoint(ckpt, candidates)
    cache[key] = scored
    return scored


def _append_cache_row(model_id: str, seed: int, df: pd.DataFrame) -> None:
    STRATUM_MRR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with STRATUM_MRR_CACHE.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"model_id": model_id, "seed": seed, "scores": df.to_dict(orient="records")},
                default=str,
            )
            + "\n"
        )


def _load_cache() -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    if not STRATUM_MRR_CACHE.exists():
        return cache
    for line in STRATUM_MRR_CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cache[f"{rec['model_id']}|{rec['seed']}"] = pd.DataFrame(rec["scores"])
    return cache


def build_epoch1_stratum_cache(paired: pd.DataFrame) -> int:
    """Score epoch-1 gene-disease candidates for pairable seeds; resumable cache."""
    from shared.pool_loader import load_primary_candidates

    candidates = load_primary_candidates()
    gd_cands = candidates[candidates["pair_type"] == "gene-disease"].copy()
    cache = _load_cache()
    pc = f"pairable_{WELL_DEF_VAL_F1}"
    pairable = paired[paired[pc]]
    n_new = 0
    for _, row in pairable.iterrows():
        mid, seed = row["model_id"], int(row["seed"])
        key = f"{mid}|{seed}"
        if key in cache:
            continue
        print(f"  epoch1 stratum cache {mid} seed={seed}", flush=True)
        ckpt = epoch_checkpoint_dir(mid, seed, 1)
        scored = score_candidates_at_checkpoint(ckpt, gd_cands)
        cache[key] = scored
        _append_cache_row(mid, seed, scored)
        n_new += 1
    print(f"  epoch1 stratum cache: {len(cache)} seeds on disk ({n_new} new)", flush=True)
    return len(cache)


def build_pool_stratum_paired_change(
    paired: pd.DataFrame,
    *,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1b: within-seed gene-disease paired change by pool-size stratum."""
    strata = _load_pool_strata()
    cache = _load_cache() if use_cache else {}
    pc = f"pairable_{WELL_DEF_VAL_F1}"
    pairable = paired[paired[pc]].copy()
    n_need = len(pairable)
    if use_cache and len(cache) < n_need:
        build_epoch1_stratum_cache(paired)
        cache = _load_cache()

    rows: list[dict] = []
    for _, row in pairable.iterrows():
        mid, seed = row["model_id"], int(row["seed"])
        key = f"{mid}|{seed}"
        try:
            best_scores = _best_val_scores(mid, seed)
            if key not in cache:
                continue
            ep1_scores = cache[key]
        except Exception as exc:
            print(f"  skip stratum {mid} seed={seed}: {exc}", flush=True)
            continue

        for stratum, pmids in [
            ("small_pool", strata["small_pool"]),
            ("large_pool", strata["large_pool"]),
            ("comparable_to_gene_drug", strata["comparable_to_gene_drug"]),
        ]:
            mrr1 = _mrr_for_stratum(ep1_scores, pmids)
            mrr_best = _mrr_for_stratum(best_scores, pmids)
            if pd.isna(mrr1) or pd.isna(mrr_best):
                continue
            rows.append(
                {
                    "model_id": mid,
                    "short_name": row["short_name"],
                    "seed": seed,
                    "stratum": stratum,
                    "mrr_epoch1": mrr1,
                    "mrr_best_val": mrr_best,
                    "delta_mrr": mrr_best - mrr1,
                    "n_pmids_in_stratum": len(pmids),
                }
            )

    detail = pd.DataFrame(rows)
    summary_rows: list[dict] = []
    for stratum in ["small_pool", "large_pool", "comparable_to_gene_drug"]:
        sub = detail[detail["stratum"] == stratum]
        if sub.empty:
            continue
        vals = sub["delta_mrr"].astype(float).to_numpy()
        mean, lo, hi = _bootstrap_ci(vals)
        summary_rows.append(
            {
                "stratum": stratum,
                "n_seeds": int(len(vals)),
                "mean_delta_mrr": mean,
                "median_delta_mrr": float(np.median(vals)),
                "ci_lo": lo,
                "ci_hi": hi,
                "n_falls": int((vals < 0).sum()),
                "frac_falls": float((vals < 0).mean()) if len(vals) else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    return detail, summary


def bootstrap_positive_sign_stability(traj: pd.DataFrame, n_boot: int = 2000) -> dict[str, float]:
    """Confirm gene-disease-hard delta sign stability via bootstrap over positives at best val."""
    from shared.metrics_ranking import _rank_within_abstracts

    rows: list[float] = []
    for spec in MODELS:
        for seed in TRAIN_SEEDS:
            try:
                scores = _best_val_scores(spec.model_id, seed)
            except Exception:
                continue
            gd = scores[scores["pair_type"] == "gene-disease"].copy()
            ep1_path = epoch_checkpoint_dir(spec.model_id, seed, 1)
            if not ep1_path.exists():
                continue
            meta = load_training_meta(spec.model_id, seed)
            if not meta:
                continue
            best_ep = int(meta["best_epoch_val_f1"])
            sub_traj = traj[
                (traj["model_id"] == spec.model_id)
                & (traj["seed"] == seed)
                & (traj["epoch"].isin([1, best_ep]))
            ]
            if len(sub_traj) < 2:
                continue
            hard_col = "kb_mrr_gene_disease_hard"
            if hard_col not in sub_traj.columns:
                continue
            e1 = float(sub_traj.loc[sub_traj["epoch"] == 1, hard_col].iloc[0])
            eb = float(sub_traj.loc[sub_traj["epoch"] == best_ep, hard_col].iloc[0])
            rows.append(eb - e1)

    if not rows:
        return {"frac_negative_bootstrap": np.nan, "mean_delta": np.nan}
    arr = np.array(rows)
    rng = np.random.default_rng(42)
    boots = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)]
    return {
        "mean_delta": float(arr.mean()),
        "frac_negative_bootstrap": float(np.mean(np.array(boots) < 0)),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
    }


def interpret_timing(summary: pd.DataFrame) -> str:
    gd = summary[(summary["slug"] == "gene_disease") & (summary["timing_class"] != "all")]
    if gd.empty:
        return "Timing data unavailable."
    before = float(gd.loc[gd["timing_class"] == "before_best_val", "frac_seeds"].iloc[0])
    coinc = float(gd.loc[gd["timing_class"] == "coincident_best_val", "frac_seeds"].iloc[0])
    after = float(gd.loc[gd["timing_class"] == "after_best_val", "frac_seeds"].iloc[0])
    if after >= 0.5:
        return (
            "Most seeds show the gene-disease ranking peak after the validation-best checkpoint, "
            "so the decline from early training to the validation-best point is largely ordinary "
            "late-training movement rather than early divergence."
        )
    if before + coinc >= 0.5:
        return (
            "Most seeds show the gene-disease ranking peak at or before the validation-best "
            "checkpoint, so ranking is already falling or flat by the time the model reaches its "
            "validation optimum. That pattern is harder to explain as simple post-optimum overfitting alone."
        )
    return "Timing classes are mixed across seeds; no single mundane timing story dominates."


def interpret_pool_strata(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "Pool stratification unavailable."
    rows = {r["stratum"]: r for _, r in summary.iterrows()}
    comp = rows.get("comparable_to_gene_drug")
    large = rows.get("large_pool")
    small = rows.get("small_pool")
    if comp is not None and float(comp["mean_delta_mrr"]) < -0.01 and float(comp["frac_falls"]) >= 0.55:
        return (
            "Gene-disease ranking still falls when restricted to abstracts whose pool size matches "
            "the gene-drug range, so the erosion is not explained by large-pool difficulty alone."
        )
    if large is not None and small is not None:
        if float(large["mean_delta_mrr"]) < float(small["mean_delta_mrr"]) - 0.02:
            return (
                "The gene-disease decline is larger in high pool-size abstracts, suggesting pool "
                "difficulty contributes; interpret the relation-type signal cautiously."
            )
    return (
        "Pool-size stratification does not fully explain the gene-disease pattern; effect sizes "
        "should be read alongside the comparable-range restriction."
    )


def run_mundane_explanations(
    traj: pd.DataFrame | None = None,
    paired: pd.DataFrame | None = None,
    *,
    skip_stratum_inference: bool = False,
) -> dict[str, Any]:
    from .config import PAIRED_CHANGES_CSV

    if traj is None:
        traj = load_all_epoch_scores()
    if paired is None:
        paired = pd.read_csv(PAIRED_CHANGES_CSV)

    timing = build_kb_peak_timing(traj)
    timing.to_csv(TIMING_CLASSIFICATION_CSV, index=False)
    timing_summary = summarize_timing(timing)
    timing_summary.to_csv(TIMING_SUMMARY_CSV, index=False)

    stratum_detail = pd.DataFrame()
    stratum_summary = pd.DataFrame()
    if not skip_stratum_inference:
        stratum_detail, stratum_summary = build_pool_stratum_paired_change(paired)
        stratum_detail.to_csv(POOL_STRATUM_CSV, index=False)
        stratum_summary.to_csv(POOL_STRATUM_SUMMARY_CSV, index=False)

    pos_boot = bootstrap_positive_sign_stability(traj)

    timing_interp = interpret_timing(timing_summary)
    pool_interp = interpret_pool_strata(stratum_summary)

    print("\n=== Part 1a: KB peak timing vs best-validation epoch ===")
    for slug in ("gene_disease", "gene_disease_hard"):
        sub = timing_summary[(timing_summary["slug"] == slug) & (timing_summary["timing_class"] != "all")]
        print(f"  {slug}:")
        for _, r in sub.iterrows():
            print(f"    {r['timing_class']}: {int(r['n_seeds'])} seeds ({float(r['frac_seeds']):.1%})")
    print(f"  Interpretation: {timing_interp}")

    print("\n=== Part 1b: Pool-size stratified gene-disease paired change ===")
    if not stratum_summary.empty:
        for _, r in stratum_summary.iterrows():
            print(
                f"  {r['stratum']}: mean delta {float(r['mean_delta_mrr']):+.4f} "
                f"[{float(r['ci_lo']):+.4f}, {float(r['ci_hi']):+.4f}], "
                f"falls {int(r['n_falls'])}/{int(r['n_seeds'])}"
            )
        print(f"  Interpretation: {pool_interp}")
    else:
        print("  (skipped stratum inference)")

    print(
        f"\n  Positive-count bootstrap (gene-disease-hard paired delta): "
        f"mean {pos_boot.get('mean_delta', float('nan')):+.4f}, "
        f"P(negative)={pos_boot.get('frac_negative_bootstrap', float('nan')):.1%}"
    )

    return {
        "timing": timing,
        "timing_summary": timing_summary,
        "timing_interpretation": timing_interp,
        "stratum_detail": stratum_detail,
        "stratum_summary": stratum_summary,
        "pool_interpretation": pool_interp,
        "positive_bootstrap": pos_boot,
    }
