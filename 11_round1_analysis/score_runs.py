"""Score KB pool at best checkpoint for each matrix run (inference only)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from shared.inference import score_checkpoint_path, write_scores_jsonl
from shared.metrics_calibration import calibration_for_scores
from shared.metrics_ranking import metrics_by_pair_type
from shared.models import MODELS, MODEL_BY_ID
from shared.pool_loader import load_primary_candidates

from .config import (
    MATRIX_COMPLETE,
    MATRIX_CKPT_DIR,
    MATRIX_RESULTS_DIR,
    PER_RUN_CSV,
    SCORE_COMPLETE,
    SCORES_DIR,
    TRAIN_SEEDS,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def matrix_best_ckpt(model_id: str, seed: int) -> Path:
    return MATRIX_CKPT_DIR / model_id / f"seed_{seed}" / "best"


def matrix_marker(model_id: str, seed: int) -> Path:
    return MATRIX_RESULTS_DIR / model_id / f"seed_{seed}" / MATRIX_COMPLETE


def score_jsonl_path(model_id: str, seed: int) -> Path:
    return SCORES_DIR / model_id / f"seed_{seed}.jsonl"


def scoring_complete_marker(model_id: str, seed: int) -> Path:
    return SCORES_DIR / model_id / f"seed_{seed}" / SCORE_COMPLETE


def is_scored(model_id: str, seed: int) -> bool:
    return scoring_complete_marker(model_id, seed).exists()


def count_scored_runs(model_ids: list[str] | None = None, seeds: list[int] | None = None) -> int:
    specs = MODELS if model_ids is None else [MODEL_BY_ID[m] for m in model_ids]
    seed_list = TRAIN_SEEDS if seeds is None else seeds
    return sum(1 for s in specs for seed in seed_list if is_scored(s.model_id, seed))


def _write_scoring_marker(model_id: str, seed: int, payload: dict) -> Path:
    path = scoring_complete_marker(model_id, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def score_one(model_id: str, seed: int, candidates: pd.DataFrame, force: bool = False) -> dict:
    train_marker = matrix_marker(model_id, seed)
    if not train_marker.exists():
        raise FileNotFoundError(f"Missing matrix completion marker: {train_marker}")

    out_path = score_jsonl_path(model_id, seed)
    marker_path = scoring_complete_marker(model_id, seed)

    if marker_path.exists() and not force and out_path.exists():
        return json.loads(marker_path.read_text(encoding="utf-8"))

    ckpt = matrix_best_ckpt(model_id, seed)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {ckpt}")

    _log(f"  SCORE START {model_id} seed={seed} checkpoint={ckpt}")
    scores = score_checkpoint_path(
        ckpt, candidates, model_id=model_id, seed=seed, run_id=f"{model_id}_seed_{seed}"
    )
    write_scores_jsonl(scores, out_path)

    meta = json.loads(train_marker.read_text(encoding="utf-8"))
    spec = MODEL_BY_ID[model_id]
    kb_rows = metrics_by_pair_type(scores, f"{model_id}_seed_{seed}")
    kb_map = {row["pair_type"]: row for _, row in kb_rows.iterrows()}
    cal = calibration_for_scores(scores, f"{model_id}_seed_{seed}")

    payload = {
        "model_id": model_id,
        "short_name": spec.short_name,
        "seed": seed,
        "run_id": f"{model_id}_seed_{seed}",
        "training_strategy": meta.get("training_strategy"),
        "recipe_lr": meta.get("recipe_lr"),
        "recipe_warmup_label": meta.get("recipe_warmup_label"),
        "best_epoch_val_f1": meta.get("best_epoch_val_f1"),
        "best_val_f1": meta.get("best_val_f1"),
        "benchmark_f1": meta.get("benchmark_f1"),
        "scores_path": str(out_path),
        "checkpoint": str(ckpt),
        "kb_mrr_gene_drug": float(kb_map.get("gene-drug", {}).get("mrr", 0)),
        "kb_mrr_gene_disease": float(kb_map.get("gene-disease", {}).get("mrr", 0)),
        "kb_mrr_overall": float(kb_rows["mrr"].mean()) if not kb_rows.empty else 0.0,
        "kb_auc_pr_gene_drug": float(kb_map.get("gene-drug", {}).get("auc_pr", 0)),
        "kb_auc_pr_gene_disease": float(kb_map.get("gene-disease", {}).get("auc_pr", 0)),
        "kb_recall1_gene_drug": float(kb_map.get("gene-drug", {}).get("recall_at_1", 0)),
        "kb_recall1_gene_disease": float(kb_map.get("gene-disease", {}).get("recall_at_1", 0)),
        "ece": float(cal["ece"]),
        "n_candidates_scored": int(len(scores)),
    }
    _write_scoring_marker(model_id, seed, payload)
    _log(
        f"  SCORE DONE {model_id} seed={seed}: bench={payload['benchmark_f1']:.3f} "
        f"KB gd={payload['kb_mrr_gene_drug']:.3f} gdis={payload['kb_mrr_gene_disease']:.3f} "
        f"marker={marker_path}"
    )
    return payload


def load_scored_summary() -> pd.DataFrame:
    rows: list[dict] = []
    for spec in MODELS:
        for seed in TRAIN_SEEDS:
            mp = scoring_complete_marker(spec.model_id, seed)
            if mp.exists():
                rows.append(json.loads(mp.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def score_all_runs(
    *,
    force: bool = False,
    model_ids: list[str] | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    specs = MODELS if model_ids is None else [MODEL_BY_ID[m] for m in model_ids]
    seed_list = TRAIN_SEEDS if seeds is None else seeds
    planned = len(specs) * len(seed_list)

    _log(
        f"\n=== Round 1 KB scoring ===\n"
        f"Encoders: {[s.model_id for s in specs]}\n"
        f"Seeds: {seed_list}\n"
        f"Planned runs this job: {planned}\n"
        f"Already scored: {count_scored_runs([s.model_id for s in specs], seed_list)}\n"
    )

    candidates = load_primary_candidates()
    _log(f"Frozen pool: {len(candidates)} candidates, {candidates['pmid'].nunique()} abstracts")

    rows: list[dict] = []
    n_skip, n_done = 0, 0
    for spec in specs:
        for seed in seed_list:
            if is_scored(spec.model_id, seed) and not force:
                row = json.loads(scoring_complete_marker(spec.model_id, seed).read_text(encoding="utf-8"))
                rows.append(row)
                n_skip += 1
                _log(f"  skip (scored): {spec.model_id} seed={seed}")
                continue
            try:
                row = score_one(spec.model_id, seed, candidates, force=force)
                rows.append(row)
                n_done += 1
            except FileNotFoundError as exc:
                _log(f"  skip: {exc}")

    df = pd.DataFrame(rows)
    if not df.empty:
        PER_RUN_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PER_RUN_CSV, index=False)

    _log(
        f"\n=== KB scoring complete (this job) ===\n"
        f"  Newly scored: {n_done}, skipped (already done): {n_skip}, "
        f"total markers on disk: {count_scored_runs()}/72\n"
    )
    return df
