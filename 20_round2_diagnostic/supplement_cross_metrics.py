"""Add pair-type × subset cross metrics to existing epoch score JSON (GPU inference, resumable)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from shared.models import MODELS

from .config import SCORES_DIR, SCORING_MODEL_IDS, TRAIN_SEEDS
from .epoch_scoring import epoch_score_path
from .pool_cache import load_enriched_pool
from .scoring import kb_metrics_from_scores, score_candidates_at_checkpoint

CROSS_FIELDS = (
    "kb_mrr_gene_disease_hard",
    "kb_mrr_gene_disease_easy",
    "kb_mrr_gene_drug_hard",
    "kb_mrr_gene_drug_easy",
)


def _needs_supplement(payload: dict[str, Any], force: bool) -> bool:
    if force:
        return True
    return any(f not in payload or payload[f] is None for f in CROSS_FIELDS)


def supplement_one_epoch(
    model_id: str,
    seed: int,
    epoch: int,
    *,
    candidates: pd.DataFrame,
    pool: pd.DataFrame,
    force: bool = False,
) -> bool:
    """Return True if JSON was updated."""
    path = epoch_score_path(model_id, seed, epoch)
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not _needs_supplement(payload, force):
        return False

    ckpt = Path(payload["checkpoint"])
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint for supplement: {ckpt}")

    scores = score_candidates_at_checkpoint(ckpt, candidates)
    cross = kb_metrics_from_scores(scores, pool)
    for key in CROSS_FIELDS:
        if key in cross:
            payload[key] = cross[key]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def supplement_all_cross_metrics(
    *,
    model_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    force: bool = False,
) -> dict[str, int]:
    mids = model_ids or list(SCORING_MODEL_IDS)
    seed_list = seeds or list(TRAIN_SEEDS)
    pool = load_enriched_pool()
    candidates = pool.drop(columns=["subset"], errors="ignore")

    n_total, n_updated, n_skip = 0, 0, 0
    print(
        f"\n=== Supplement pair×subset cross metrics ===\n"
        f"Encoders: {mids}\nSeeds: {seed_list}\n",
        flush=True,
    )

    for model_id in mids:
        for seed in seed_list:
            d = SCORES_DIR / model_id / f"seed_{seed}"
            if not d.exists():
                continue
            for path in sorted(d.glob("epoch_*.json")):
                n_total += 1
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not _needs_supplement(payload, force):
                    n_skip += 1
                    continue
                epoch = int(payload["epoch"])
                print(f"  SUPPLEMENT {model_id} seed={seed} epoch={epoch}", flush=True)
                if supplement_one_epoch(
                    model_id,
                    seed,
                    epoch,
                    candidates=candidates,
                    pool=pool,
                    force=force,
                ):
                    n_updated += 1

    print(
        f"\n=== Cross-metric supplement done ===\n"
        f"  Total epoch JSON: {n_total}, updated: {n_updated}, already complete: {n_skip}\n",
        flush=True,
    )
    return {"total": n_total, "updated": n_updated, "skipped": n_skip}
