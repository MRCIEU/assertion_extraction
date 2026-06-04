"""Step 0: inventory of saved checkpoints under Round 1."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import (
    FOCUS_MODEL_IDS,
    MODELS,
    R1_CHECKPOINTS,
    R1_SWEEP_CKPT_F1,
    R1_SWEEP_CKPT_LOSS,
    R1_SWEEP_RESULTS,
    ROUND1_RECIPE_LR,
    ROUND1_RECIPE_WARMUP_LABEL,
    TRAIN_SEEDS,
    TRAINING_STRATEGY,
)


def _has_model_weights(d: Path) -> bool:
    return (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()


def inventory_main_matrix() -> pd.DataFrame:
    rows: list[dict] = []
    for spec in MODELS:
        for seed in TRAIN_SEEDS:
            ckpt_dir = R1_CHECKPOINTS / spec.model_id / f"seed_{seed}"
            meta_path = ckpt_dir / "10_train_metadata.json"
            recoverable_epochs: list[int] = []
            policy = "none"
            best_f1_ep = None
            best_loss_ep = None
            n_epochs_logged = 0

            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("training_strategy") == TRAINING_STRATEGY:
                    curve = meta.get("epoch_curve") or []
                    n_epochs_logged = len(curve)
                    if curve:
                        best_loss_ep = int(min(curve, key=lambda r: r["val_loss"])["epoch"])
                    best_f1_ep = int(meta.get("best_epoch_val_f1", 0) or 0)
                    if _has_model_weights(ckpt_dir):
                        recoverable_epochs = [best_f1_ep]
                        policy = "val_f1_best_only"

            rows.append(
                {
                    "source": "round1_main",
                    "model_id": spec.model_id,
                    "seed": seed,
                    "checkpoint_policy": policy,
                    "recoverable_epochs": ",".join(str(e) for e in recoverable_epochs) or "",
                    "n_recoverable_checkpoints": len(recoverable_epochs),
                    "best_epoch_val_f1": best_f1_ep,
                    "best_epoch_val_loss": best_loss_ep,
                    "n_epochs_logged": n_epochs_logged,
                    "checkpoint_path_exists": _has_model_weights(ckpt_dir),
                }
            )
    return pd.DataFrame(rows)


def inventory_sweep_recipe_match() -> pd.DataFrame:
    """Sweep runs at Round 1 recipe (lr 2e-5, no warmup) save val_loss and val_f1 checkpoints."""
    rows: list[dict] = []
    for model_id in FOCUS_MODEL_IDS:
        run_id = f"{model_id}_lr{ROUND1_RECIPE_LR:g}_none_seed42".replace("e-0", "e-")
        # folder names use 2e-5 not 2e-05
        run_id = f"{model_id}_lr2e-5_{ROUND1_RECIPE_WARMUP_LABEL}_seed42"
        res = R1_SWEEP_RESULTS / run_id / "sweep_complete.json"
        if not res.exists():
            rows.append(
                {
                    "source": "round1_sweep",
                    "model_id": model_id,
                    "seed": 42,
                    "checkpoint_policy": "missing",
                    "recoverable_epochs": "",
                    "n_recoverable_checkpoints": 0,
                }
            )
            continue
        data = json.loads(res.read_text(encoding="utf-8"))
        loss_ep = int(data.get("best_epoch_by_val_loss", 0))
        f1_ep = int(data.get("best_epoch_by_val_f1", 0))
        loss_dir = Path(data.get("checkpoint_val_loss", ""))
        f1_dir = Path(data.get("checkpoint_val_f1", ""))
        rec: list[int] = []
        if loss_dir.exists() and _has_model_weights(loss_dir):
            rec.append(loss_ep)
        if f1_dir.exists() and _has_model_weights(f1_dir):
            rec.append(f1_ep)
        rows.append(
            {
                "source": "round1_sweep",
                "model_id": model_id,
                "seed": 42,
                "checkpoint_policy": "val_loss_and_val_f1",
                "recoverable_epochs": ",".join(str(e) for e in sorted(set(rec))),
                "n_recoverable_checkpoints": len(rec),
                "best_epoch_val_f1": f1_ep,
                "best_epoch_val_loss": loss_ep,
                "n_epochs_logged": int(data.get("epochs_run", 0)),
                "checkpoint_path_exists": len(rec) == 2,
            }
        )
    return pd.DataFrame(rows)


def build_checkpoint_inventory() -> tuple[pd.DataFrame, str]:
    main = inventory_main_matrix()
    sweep = inventory_sweep_recipe_match()
    inv = pd.concat([main, sweep], ignore_index=True)

    main_policy = (
        "Round 1 main matrix saves a single checkpoint per run: weights at the best validation F1 "
        "epoch only (not every epoch, not a separate val_loss-best save). Per-epoch validation "
        "metrics are logged in training metadata for all epochs run."
    )
    sweep_note = (
        "The earlier sweep under the same step (lr 2e-5, no warmup, seed 42, three encoders) "
        "saved both val_loss-best and val_f1-best weights; used only as a two-point supplement "
        "because the main matrix does not retain val_loss-best checkpoints."
    )
    case = (
        f"{main_policy} {sweep_note} "
        "Trajectory density on the held-out benchmark and KB axes: one test-time point per main "
        "run (val_f1-best checkpoint); up to two test-time points per encoder from the matched "
        "sweep runs at seed 42."
    )
    return inv, case


def print_inventory_summary(inv: pd.DataFrame, case: str) -> None:
    print("\n=== Step 0: Checkpoint inventory ===")
    print(case)
    for mid in FOCUS_MODEL_IDS:
        sub = inv[(inv["model_id"] == mid) & (inv["source"] == "round1_main")]
        print(f"\n{mid} (main matrix, seeds 42-49):")
        for _, r in sub.iterrows():
            print(
                f"  seed {int(r['seed'])}: recoverable epochs [{r['recoverable_epochs']}] "
                f"(logged epochs={int(r['n_epochs_logged'])})"
            )
    print("\nMatched sweep (lr 2e-5, no warmup, seed 42):")
    sw = inv[inv["source"] == "round1_sweep"]
    for _, r in sw.iterrows():
        print(
            f"  {r['model_id']}: epochs [{r['recoverable_epochs']}] "
            f"policy={r['checkpoint_policy']}"
        )
