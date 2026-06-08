"""Step 0: inventory of per-epoch checkpoints from folder-10 matrix."""

from __future__ import annotations

import pandas as pd

from .config import FOCUS_MODEL_IDS, MODELS, TRAIN_SEEDS
from .matrix_io import list_recoverable_epochs, load_training_meta


def inventory_main_matrix() -> pd.DataFrame:
    rows: list[dict] = []
    for spec in MODELS:
        for seed in TRAIN_SEEDS:
            meta = load_training_meta(spec.model_id, seed)
            recoverable_epochs = list_recoverable_epochs(spec.model_id, seed, meta)
            curve = (meta or {}).get("epoch_curve") or []
            n_epochs_logged = len(curve)
            best_f1_ep = int((meta or {}).get("best_epoch_val_f1", 0) or 0) if meta else None
            recipe_lr = (meta or {}).get("recipe_lr")
            policy = "all_epochs_saved" if recoverable_epochs else ("log_only" if meta else "missing")

            rows.append(
                {
                    "source": "matrix",
                    "model_id": spec.model_id,
                    "seed": seed,
                    "recipe_lr": recipe_lr,
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
    n_with_epochs = int((inv["n_recoverable_checkpoints"] > 0).sum())
    n_runs = len(inv)
    focus_epochs = int(
        inv[inv["model_id"].isin(FOCUS_MODEL_IDS)]["n_recoverable_checkpoints"].sum()
    )
    case = (
        f"Folder-10 step-2 matrix (1e-5 recipe, no warmup): {n_with_epochs}/{n_runs} runs have "
        f"recoverable per-epoch fp16 checkpoints under "
        "matrix/checkpoints/{{model_id}}/seed_{{seed}}/epochs/epoch_NN/, plus fp32 best/ at "
        "val_f1-best. training_log.json (or matrix_complete.json fallback) records val_loss and "
        f"val_f1 per epoch. Focus encoders ({len(FOCUS_MODEL_IDS)} x {len(TRAIN_SEEDS)} seeds) "
        f"total {focus_epochs} epoch checkpoints available for on-demand KB/benchmark scoring."
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
                f"(logged={int(r['n_epochs_logged'])}, best val_f1 epoch={r['best_epoch_val_f1']}, "
                f"lr={r.get('recipe_lr', '?')})"
            )
