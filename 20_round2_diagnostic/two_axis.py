"""Step 2: benchmark vs KB timing along available trajectory points."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from importlib import import_module

from .config import (
    COLLAPSED_DEBERTA_SEEDS,
    FOCUS_MODEL_IDS,
    MODEL_BY_ID,
    R1_CHECKPOINTS,
    R1_EASY_HARD_CSV,
    R1_PER_RUN_CSV,
    R1_SWEEP_RESULTS,
    ROUND1_RECIPE_LR,
    ROUND1_RECIPE_WARMUP_LABEL,
    TRAIN_SEEDS,
)
from .scoring import benchmark_f1_at_checkpoint, kb_metrics_from_scores, score_candidates_at_checkpoint

_r1pool = import_module("10_round1_benchmark_kb.pool_loader")
_r1dist = import_module("10_round1_benchmark_kb.distance_analysis")


def _is_clean(model_id: str, seed: int) -> bool:
    return not (model_id == "deberta_base" and seed in COLLAPSED_DEBERTA_SEEDS)


def _load_pool_enriched() -> pd.DataFrame:
    pool = _r1pool.load_primary_candidates()
    return _r1dist.enrich_with_proximity(pool)


def trajectory_from_main_matrix(
    per_run: pd.DataFrame,
    easy_hard: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One recoverable point per clean seed: val_f1-best checkpoint (Round 1 deployed)."""
    if easy_hard is None and R1_EASY_HARD_CSV.exists():
        easy_hard = pd.read_csv(R1_EASY_HARD_CSV)
    rows: list[dict] = []
    for model_id in FOCUS_MODEL_IDS:
        sub = per_run[per_run["model_id"] == model_id]
        sub = sub[sub.apply(lambda r: _is_clean(r["model_id"], int(r["seed"])), axis=1)]
        for _, r in sub.iterrows():
            seed = int(r["seed"])
            run_id = r["run_id"]
            kb_hard = kb_easy = np.nan
            if easy_hard is not None:
                eh = easy_hard[easy_hard["run_id"] == run_id]
                easy_row = eh[eh["subset"] == "easy_co_sentence"]
                hard_row = eh[eh["subset"] == "hard_cross_sentence"]
                if not easy_row.empty:
                    kb_easy = float(easy_row.iloc[0]["mrr"])
                if not hard_row.empty:
                    kb_hard = float(hard_row.iloc[0]["mrr"])
            kb_overall = float(r.get("kb_mrr_overall", np.nan))
            if np.isnan(kb_overall):
                kb_overall = float(
                    np.nanmean([r["kb_mrr_gene_drug"], r["kb_mrr_gene_disease"]])
                )
            rows.append(
                {
                    "source": "round1_main",
                    "model_id": model_id,
                    "seed": seed,
                    "trajectory_point": "val_f1_best",
                    "epoch": int(r.get("best_epoch_val_f1", 0) or 0),
                    "benchmark_f1": float(r["benchmark_f1"]),
                    "kb_mrr_overall": kb_overall,
                    "kb_mrr_gene_drug": float(r["kb_mrr_gene_drug"]),
                    "kb_mrr_gene_disease": float(r["kb_mrr_gene_disease"]),
                    "kb_mrr_easy": float(kb_easy) if not np.isnan(kb_easy) else np.nan,
                    "kb_mrr_hard": float(kb_hard) if not np.isnan(kb_hard) else np.nan,
                    "weights_available": True,
                }
            )
    return pd.DataFrame(rows)


def trajectory_from_sweep_two_point(
    pool: pd.DataFrame,
    cache_path: Path,
    rescore: bool = False,
) -> pd.DataFrame:
    """Matched sweep (lr 2e-5, no warmup, seed 42): val_loss-best vs val_f1-best checkpoints."""
    if cache_path.exists() and not rescore:
        return pd.read_csv(cache_path)

    rows: list[dict] = []
    test_examples = import_module("10_round1_benchmark_kb.benchmark_eval").build_biored_test_examples()

    for model_id in FOCUS_MODEL_IDS:
        run_id = f"{model_id}_lr2e-5_{ROUND1_RECIPE_WARMUP_LABEL}_seed42"
        res = R1_SWEEP_RESULTS / run_id / "sweep_complete.json"
        if not res.exists():
            continue
        data = json.loads(res.read_text(encoding="utf-8"))
        spec = MODEL_BY_ID[model_id]
        for point, key, ep_key in [
            ("val_loss_best", "checkpoint_val_loss", "best_epoch_by_val_loss"),
            ("val_f1_best", "checkpoint_val_f1", "best_epoch_by_val_f1"),
        ]:
            ckpt = Path(data[key])
            if not ckpt.exists():
                continue
            scores = score_candidates_at_checkpoint(ckpt, pool)
            km = kb_metrics_from_scores(scores, pool, run_id)
            bf1 = benchmark_f1_at_checkpoint(ckpt, spec, test_examples)
            rows.append(
                {
                    "source": "round1_sweep_recipe_match",
                    "model_id": model_id,
                    "seed": 42,
                    "trajectory_point": point,
                    "epoch": int(data.get(ep_key, 0)),
                    "benchmark_f1": bf1,
                    "kb_mrr_overall": km["kb_mrr_overall"],
                    "kb_mrr_gene_drug": km.get("kb_mrr_gene_drug", np.nan),
                    "kb_mrr_gene_disease": km.get("kb_mrr_gene_disease", np.nan),
                    "kb_mrr_easy": km.get("kb_mrr_easy_co_sentence", np.nan),
                    "kb_mrr_hard": km.get("kb_mrr_hard_cross_sentence", np.nan),
                    "weights_available": True,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df


def validation_epoch_timing(model_id: str) -> pd.DataFrame:
    """Epoch where val_f1 peaks vs val_loss minimizes (validation only; all clean seeds)."""
    rows: list[dict] = []
    for seed in TRAIN_SEEDS:
        if not _is_clean(model_id, seed):
            continue
        meta_path = R1_CHECKPOINTS / model_id / f"seed_{seed}" / "10_train_metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        curve = meta.get("epoch_curve") or []
        if not curve:
            continue
        cdf = pd.DataFrame(curve)
        ep_f1 = int(cdf.loc[cdf["val_f1"].idxmax(), "epoch"])
        ep_loss = int(cdf.loc[cdf["val_loss"].idxmin(), "epoch"])
        rows.append(
            {
                "model_id": model_id,
                "seed": seed,
                "epoch_val_f1_peak": ep_f1,
                "epoch_val_loss_min": ep_loss,
                "epochs_disagree": ep_f1 != ep_loss,
            }
        )
    return pd.DataFrame(rows)


def build_two_axis_trajectory(
    per_run: pd.DataFrame,
    pool: pd.DataFrame | None,
    sweep_cache: Path,
    rescore_sweep: bool = False,
    easy_hard: pd.DataFrame | None = None,
) -> pd.DataFrame:
    main = trajectory_from_main_matrix(per_run, easy_hard=easy_hard)
    sweep = pd.DataFrame()
    if rescore_sweep:
        if pool is None:
            pool = _load_pool_enriched()
        sweep = trajectory_from_sweep_two_point(pool, sweep_cache, rescore=True)
    elif sweep_cache.exists():
        sweep = pd.read_csv(sweep_cache)
    return pd.concat([main, sweep], ignore_index=True)


def summarize_timing(traj: pd.DataFrame) -> dict[str, str]:
    notes: list[str] = []
    main = traj[traj["source"] == "round1_main"]
    sweep = traj[traj["source"] == "round1_sweep_recipe_match"]

    notes.append(
        "Main Round 1 runs expose one test-time checkpoint (val_f1-best). "
        "BioRED benchmark F1 and KB scores at a val_loss-best weight are not on disk for those runs."
    )

    for mid in FOCUS_MODEL_IDS:
        val_t = validation_epoch_timing(mid)
        if not val_t.empty:
            disagree = float(val_t["epochs_disagree"].mean())
            notes.append(
                f"{MODEL_BY_ID[mid].short_name}: validation val_f1 peak and val_loss minimum "
                f"fall on different epochs in {disagree*100:.0f}% of clean seeds."
            )

    if not sweep.empty:
        for mid in FOCUS_MODEL_IDS:
            sw = sweep[sweep["model_id"] == mid]
            if len(sw) < 2:
                continue
            loss_row = sw[sw["trajectory_point"] == "val_loss_best"].iloc[0]
            f1_row = sw[sw["trajectory_point"] == "val_f1_best"].iloc[0]
            bench_delta = float(f1_row["benchmark_f1"]) - float(loss_row["benchmark_f1"])
            hard_delta = float(f1_row["kb_mrr_hard"]) - float(loss_row["kb_mrr_hard"])
            notes.append(
                f"{MODEL_BY_ID[mid].short_name} (sweep seed 42, two saved checkpoints): "
                f"benchmark F1 {loss_row['benchmark_f1']:.3f} at val_loss-best epoch "
                f"{int(loss_row['epoch'])} versus {f1_row['benchmark_f1']:.3f} at val_f1-best "
                f"epoch {int(f1_row['epoch'])} (delta {bench_delta:+.3f}); "
                f"hard-subset KB MRR {loss_row['kb_mrr_hard']:.3f} versus "
                f"{f1_row['kb_mrr_hard']:.3f} (delta {hard_delta:+.3f})."
            )

    return {"narrative": " ".join(notes)}
