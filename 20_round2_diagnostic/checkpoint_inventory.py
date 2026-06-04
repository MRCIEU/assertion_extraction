"""Step 0: inventory of per-epoch checkpoints from folder-10 matrix."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import FOCUS_MODEL_IDS, MATRIX_CKPT_DIR, MODELS, TRAIN_SEEDS


def _has_model_weights(d: Path) -> bool:
    return (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()


def inventory_main_matrix() -> pd.DataFrame:
    rows: list[dict] = []
    for spec in MODELS:
        for seed in TRAIN_SEEDS:
            run_root = MATRIX_CKPT_DIR / spec.model_id / f"seed_{seed}"
            log_path = run_root / "training_log.json"
            recoverable_epochs: list[int] = []
            policy = "none"
            best_f1_ep = None
            n_epochs_logged = 0

            if log_path.exists():
                meta = json.loads(log_path.read_text(encoding="utf-8"))
                curve = meta.get("epoch_curve") or []
                n_epochs_logged = len(curve)
                best_f1_ep = int(meta.get("best_epoch_val_f1", 0) or 0)
                epochs_dir = run_root / "epochs"
                for ep in curve:
                    ep_num = int(ep["epoch"])
                    ep_dir = epochs_dir / f"epoch_{ep_num:02d}"
                    if _has_model_weights(ep_dir):
                        recoverable_epochs.append(ep_num)
                policy = "all_epochs_saved" if recoverable_epochs else "log_only"

            rows.append(
                {
                    "source": "matrix",
                    "model_id": spec.model_id,
                    "seed": seed,
                    "checkpoint_policy": policy,
                    "recoverable_epochs": ",".join(str(e) for e in recoverable_epochs) or "",
                    "n_recoverable_checkpoints": len(recoverable_epochs),
                    "best_epoch_val_f1": best_f1_ep,
                    "n_epochs_logged": n_epochs_logged,
                    "checkpoint_path_exists": bool(recoverable_epochs),
                }
            )
    return pd.DataFrame(rows)


def build_checkpoint_inventory() -> tuple[pd.DataFrame, str]:
    inv = inventory_main_matrix()
    case = (
        "Folder-10 step-2 training saves a checkpoint at every epoch under "
        "matrix/checkpoints/{model_id}/seed_{seed}/epochs/epoch_NN/, plus a val_f1-best "
        "copy at .../best/. Each training_log.json records val_loss, val_f1, and "
        "self-measured benchmark F1 per epoch. Folder 20 can read any training point "
        "without retraining; KB scoring on non-best epochs is on-demand for focus encoders only."
    )
    return inv, case


def print_inventory_summary(inv: pd.DataFrame, case: str) -> None:
    print("\n=== Step 0: Checkpoint inventory ===")
    print(case)
    for mid in FOCUS_MODEL_IDS:
        sub = inv[(inv["model_id"] == mid) & (inv["source"] == "matrix")]
        print(f"\n{mid} (matrix, seeds 42-49):")
        for _, r in sub.iterrows():
            print(
                f"  seed {int(r['seed'])}: {int(r['n_recoverable_checkpoints'])} recoverable epochs "
                f"(logged={int(r['n_epochs_logged'])}, best val_f1 epoch={r['best_epoch_val_f1']})"
            )
