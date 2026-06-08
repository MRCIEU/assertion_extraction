"""Read folder-10 matrix training metadata and checkpoint paths."""

from __future__ import annotations

import json
from pathlib import Path

from .config import MATRIX_CKPT_DIR, MATRIX_RESULTS_DIR


def _has_model_weights(d: Path) -> bool:
    return (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()


def run_checkpoint_root(model_id: str, seed: int) -> Path:
    return MATRIX_CKPT_DIR / model_id / f"seed_{seed}"


def epoch_checkpoint_dir(model_id: str, seed: int, epoch: int) -> Path:
    return run_checkpoint_root(model_id, seed) / "epochs" / f"epoch_{epoch:02d}"


def best_checkpoint_dir(model_id: str, seed: int) -> Path:
    return run_checkpoint_root(model_id, seed) / "best"


def load_training_meta(model_id: str, seed: int) -> dict | None:
    """Load training_log.json from checkpoints, else matrix_complete.json from results."""
    ckpt_log = run_checkpoint_root(model_id, seed) / "training_log.json"
    meta: dict | None = None
    if ckpt_log.exists():
        meta = json.loads(ckpt_log.read_text(encoding="utf-8"))

    complete = MATRIX_RESULTS_DIR / model_id / f"seed_{seed}" / "matrix_complete.json"
    complete_meta = json.loads(complete.read_text(encoding="utf-8")) if complete.exists() else None

    if meta is None and complete_meta is not None:
        meta = {
            "model_id": complete_meta.get("model_id", model_id),
            "seed": complete_meta.get("seed", seed),
            "recipe_lr": complete_meta.get("recipe_lr"),
            "recipe_warmup_label": complete_meta.get("recipe_warmup_label"),
            "best_epoch_val_f1": complete_meta.get("best_epoch_val_f1"),
            "best_val_f1": complete_meta.get("best_val_f1"),
            "best_checkpoint": complete_meta.get("best_checkpoint"),
            "epoch_curve": complete_meta.get("epoch_curve") or [],
        }
    elif meta is not None and complete_meta is not None:
        for key in ("recipe_lr", "recipe_warmup_label"):
            if meta.get(key) is None and complete_meta.get(key) is not None:
                meta[key] = complete_meta[key]
    return meta


def list_recoverable_epochs(model_id: str, seed: int, meta: dict | None = None) -> list[int]:
    if meta is None:
        meta = load_training_meta(model_id, seed)
    if not meta:
        return []
    epochs: list[int] = []
    for ep in meta.get("epoch_curve") or []:
        ep_num = int(ep["epoch"])
        ep_dir = epoch_checkpoint_dir(model_id, seed, ep_num)
        if _has_model_weights(ep_dir):
            epochs.append(ep_num)
    return epochs
