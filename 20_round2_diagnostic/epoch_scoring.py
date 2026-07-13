"""Score all per-epoch checkpoints (GPU, resumable per epoch)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from shared.models import MODEL_BY_ID, MODELS

from .config import SCORES_DIR, SCORING_COMPLETE, SCORING_MODEL_IDS, TRAIN_SEEDS, resolve_checkpoint_model_id
from .matrix_io import epoch_checkpoint_dir, list_recoverable_epochs, load_training_meta
from .pool_cache import load_enriched_pool
from .scoring import score_checkpoint_full


def epoch_score_path(model_id: str, seed: int, epoch: int) -> Path:
    return SCORES_DIR / resolve_checkpoint_model_id(model_id) / f"seed_{seed}" / f"epoch_{epoch:02d}.json"


def is_epoch_scored(model_id: str, seed: int, epoch: int) -> bool:
    return epoch_score_path(model_id, seed, epoch).exists()


def count_scored_epochs(model_ids: list[str] | None = None) -> int:
    mids = model_ids or list(SCORING_MODEL_IDS)
    n = 0
    for mid in mids:
        for seed in TRAIN_SEEDS:
            d = SCORES_DIR / resolve_checkpoint_model_id(mid) / f"seed_{seed}"
            if d.exists():
                n += len(list(d.glob("epoch_*.json")))
    return n


def count_expected_epochs(model_ids: list[str] | None = None) -> int:
    mids = model_ids or list(SCORING_MODEL_IDS)
    total = 0
    for mid in mids:
        for seed in TRAIN_SEEDS:
            total += len(list_recoverable_epochs(mid, seed))
    return total


def _val_at_epoch(meta: dict, epoch: int) -> tuple[float, float]:
    for ep in meta.get("epoch_curve") or []:
        if int(ep["epoch"]) == epoch:
            return float(ep.get("val_f1", float("nan"))), float(ep.get("val_loss", float("nan")))
    return float("nan"), float("nan")


def score_one_epoch(
    model_id: str,
    seed: int,
    epoch: int,
    *,
    candidates: pd.DataFrame,
    pool: pd.DataFrame,
    test_examples: list[dict],
    meta: dict,
    force: bool = False,
) -> dict[str, Any]:
    out_path = epoch_score_path(model_id, seed, epoch)
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    ckpt = epoch_checkpoint_dir(model_id, seed, epoch)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    spec = MODEL_BY_ID[model_id]
    print(f"  SCORE {spec.short_name} seed={seed} epoch={epoch}", flush=True)
    metrics = score_checkpoint_full(ckpt, candidates, pool, test_examples)
    val_f1, val_loss = _val_at_epoch(meta, epoch)

    payload: dict[str, Any] = {
        "model_id": model_id,
        "seed": seed,
        "epoch": epoch,
        "val_f1": val_f1,
        "val_loss": val_loss,
        "recipe_lr": meta.get("recipe_lr"),
        "recipe_warmup_label": meta.get("recipe_warmup_label"),
        "best_epoch_val_f1": meta.get("best_epoch_val_f1"),
        "checkpoint": str(ckpt),
        **metrics,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def score_all_epochs(
    *,
    model_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    from shared.benchmark_eval import build_biored_test_examples

    mids = [MODEL_BY_ID[m].model_id for m in (model_ids or list(SCORING_MODEL_IDS))]
    seed_list = seeds or list(TRAIN_SEEDS)
    expected = count_expected_epochs(mids)
    already = count_scored_epochs(mids)

    print(
        f"\n=== Per-epoch scoring (5e-6/none matrix) ===\n"
        f"Encoders: {mids}\n"
        f"Seeds: {seed_list}\n"
        f"Expected epoch checkpoints: {expected}\n"
        f"Already scored: {already}\n",
        flush=True,
    )

    pool = load_enriched_pool()
    candidates = pool.drop(columns=["subset"], errors="ignore")
    test_examples = build_biored_test_examples()

    rows: list[dict] = []
    n_new, n_skip = 0, 0

    for model_id in mids:
        for seed in seed_list:
            meta = load_training_meta(model_id, seed)
            if not meta:
                print(f"  skip {model_id} seed={seed}: no training meta", flush=True)
                continue
            epochs = list_recoverable_epochs(model_id, seed, meta)
            for epoch in epochs:
                if is_epoch_scored(model_id, seed, epoch) and not force:
                    row = json.loads(epoch_score_path(model_id, seed, epoch).read_text(encoding="utf-8"))
                    rows.append(row)
                    n_skip += 1
                    continue
                try:
                    row = score_one_epoch(
                        model_id,
                        seed,
                        epoch,
                        candidates=candidates,
                        pool=pool,
                        test_examples=test_examples,
                        meta=meta,
                        force=force,
                    )
                    rows.append(row)
                    n_new += 1
                except Exception as exc:
                    print(f"  FAIL {model_id} seed={seed} ep={epoch}: {exc}", flush=True)
                    raise

    df = pd.DataFrame(rows)
    scored = count_scored_epochs(mids)
    payload = {
        "expected_epoch_checkpoints": expected,
        "scored_epochs": scored,
        "complete": scored >= expected,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "encoders": mids,
    }
    SCORING_COMPLETE.parent.mkdir(parents=True, exist_ok=True)
    SCORING_COMPLETE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"\n=== Epoch scoring batch done ===\n"
        f"  Newly scored: {n_new}, skipped: {n_skip}, on disk: {scored}/{expected}\n",
        flush=True,
    )
    return df


def load_all_epoch_scores() -> pd.DataFrame:
    rows: list[dict] = []
    for spec in MODELS:
        for seed in TRAIN_SEEDS:
            d = SCORES_DIR / resolve_checkpoint_model_id(spec.model_id) / f"seed_{seed}"
            if not d.exists():
                continue
            for p in sorted(d.glob("epoch_*.json")):
                row = json.loads(p.read_text(encoding="utf-8"))
                row["model_id"] = spec.model_id
                rows.append(row)
    return pd.DataFrame(rows)
