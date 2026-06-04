"""Score KB pool at best checkpoint for each matrix run (inference only)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from shared.inference import score_checkpoint_path, write_scores_jsonl
from shared.metrics_calibration import calibration_for_scores
from shared.metrics_ranking import metrics_by_pair_type
from shared.models import MODELS, MODEL_BY_ID
from shared.pool_loader import load_primary_candidates

from .config import MATRIX_COMPLETE, MATRIX_CKPT_DIR, MATRIX_RESULTS_DIR, SCORES_DIR, TRAIN_SEEDS


def matrix_best_ckpt(model_id: str, seed: int) -> Path:
    return MATRIX_CKPT_DIR / model_id / f"seed_{seed}" / "best"


def matrix_marker(model_id: str, seed: int) -> Path:
    return MATRIX_RESULTS_DIR / model_id / f"seed_{seed}" / MATRIX_COMPLETE


def score_one(model_id: str, seed: int, candidates: pd.DataFrame, force: bool = False) -> dict:
    marker = matrix_marker(model_id, seed)
    if not marker.exists():
        raise FileNotFoundError(f"Missing matrix completion marker: {marker}")

    out_path = SCORES_DIR / model_id / f"seed_{seed}.jsonl"
    if out_path.exists() and not force:
        scores = pd.read_json(out_path, lines=True)
    else:
        ckpt = matrix_best_ckpt(model_id, seed)
        scores = score_checkpoint_path(
            ckpt, candidates, model_id=model_id, seed=seed, run_id=f"{model_id}_seed_{seed}"
        )
        write_scores_jsonl(scores, out_path)

    meta = json.loads(marker.read_text(encoding="utf-8"))
    spec = MODEL_BY_ID[model_id]
    kb_rows = metrics_by_pair_type(scores, f"{model_id}_seed_{seed}")
    kb_map = {row["pair_type"]: row for _, row in kb_rows.iterrows()}
    cal = calibration_for_scores(scores, f"{model_id}_seed_{seed}")

    return {
        "model_id": model_id,
        "short_name": spec.short_name,
        "seed": seed,
        "run_id": f"{model_id}_seed_{seed}",
        "training_strategy": meta.get("training_strategy"),
        "best_epoch_val_f1": meta.get("best_epoch_val_f1"),
        "best_val_f1": meta.get("best_val_f1"),
        "benchmark_f1": meta.get("benchmark_f1"),
        "kb_mrr_gene_drug": float(kb_map.get("gene-drug", {}).get("mrr", 0)),
        "kb_mrr_gene_disease": float(kb_map.get("gene-disease", {}).get("mrr", 0)),
        "kb_mrr_overall": float(kb_rows["mrr"].mean()) if not kb_rows.empty else 0.0,
        "ece": float(cal["ece"]),
    }


def score_all_runs(
    *,
    force: bool = False,
    model_ids: list[str] | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    specs = MODELS if model_ids is None else [MODEL_BY_ID[m] for m in model_ids]
    seed_list = TRAIN_SEEDS if seeds is None else seeds
    candidates = load_primary_candidates()

    rows: list[dict] = []
    for spec in specs:
        for seed in seed_list:
            try:
                row = score_one(spec.model_id, seed, candidates, force=force)
                rows.append(row)
                print(
                    f"  scored {spec.model_id} seed={seed}: "
                    f"bench={row['benchmark_f1']:.3f} KB gd={row['kb_mrr_gene_drug']:.3f}"
                )
            except FileNotFoundError as exc:
                print(f"  skip: {exc}")

    return pd.DataFrame(rows)
