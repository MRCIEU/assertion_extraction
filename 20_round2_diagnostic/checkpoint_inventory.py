"""Step 0: inventory of per-epoch checkpoints from folder-10 matrix (5e-6/none)."""

from __future__ import annotations

import pandas as pd

from .config import MODELS, TRAIN_SEEDS
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
            recipe_wu = (meta or {}).get("recipe_warmup_label")
            policy = "all_epochs_saved" if recoverable_epochs else ("log_only" if meta else "missing")

            rows.append(
                {
                    "source": "matrix",
                    "model_id": spec.model_id,
                    "short_name": spec.short_name,
                    "seed": seed,
                    "recipe_lr": recipe_lr,
                    "recipe_warmup_label": recipe_wu,
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
    total_epochs = int(inv["n_recoverable_checkpoints"].sum())
    lr_sample = inv["recipe_lr"].dropna().unique()
    lr_str = f"{lr_sample[0]:.0e}" if len(lr_sample) else "unknown"

    enc_summary = inv.groupby("model_id").agg(
        n_seeds=("seed", "count"),
        total_epochs=("n_recoverable_checkpoints", "sum"),
        mean_epochs=("n_recoverable_checkpoints", "mean"),
    )

    case = (
        f"Folder-10 step-2 matrix at learning rate {lr_str} with no warmup: {n_with_epochs}/{n_runs} "
        f"runs have recoverable per-epoch fp16 checkpoints under "
        "matrix/checkpoints/{{model_id}}/seed_{{seed}}/epochs/epoch_NN/, plus fp32 best/ at "
        f"validation-F1 best. Total {total_epochs} epoch checkpoints across nine encoders "
        f"(eight seeds each where training completed). Per-encoder epoch counts: "
        + "; ".join(
            f"{mid} {int(r.total_epochs)} epochs ({r.n_seeds} seeds, mean {r.mean_epochs:.1f}/seed)"
            for mid, r in enc_summary.iterrows()
        )
        + "."
    )
    return inv, case


def print_inventory_summary(inv: pd.DataFrame, case: str) -> None:
    print("\n=== Checkpoint inventory (5e-6/none matrix) ===")
    print(case)
    for spec in MODELS:
        sub = inv[inv["model_id"] == spec.model_id]
        n_ep = int(sub["n_recoverable_checkpoints"].sum())
        print(f"  {spec.short_name}: {n_ep} epoch checkpoints across {len(sub)} seeds")
