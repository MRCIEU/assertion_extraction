"""run_experiment.py — Phase B entry point (clean .py source, no .pyc hacks).

Usage:
  python3.11 -m fine_tuning_experiments.phase_b.trainer.run_experiment \
      --experiment-id PB_PB_P_FT_T2_Sp_s01 \
      --config-path /path/to/config.yaml \
      --run-root /path/to/runs/phase_b

Side effects: creates `run_root / experiment_id /` and writes all trainer
artifacts there (checkpoints, metrics, predictions, run_manifest.json).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from fine_tuning_experiments.phase_b.trainer.scientific_trainer import (
    run_scientific_training,
)
from fine_tuning_experiments.phase_b.trainer.minimal_trainer import (
    run_minimal_training,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _sanity_check_cfg(cfg: dict, exp_id: str) -> None:
    assert isinstance(cfg, dict), f"cfg must be dict, got {type(cfg)}"
    scientific = cfg.get("scientific_trainer", {})
    minimal = cfg.get("minimal_trainer", {})
    if minimal.get("enabled") and scientific.get("enabled"):
        raise ValueError(
            f"{exp_id}: both minimal_trainer.enabled and scientific_trainer.enabled "
            "are True; at most one trainer may be active per run."
        )
    if not (scientific.get("enabled") or minimal.get("enabled")):
        raise ValueError(
            f"{exp_id}: neither scientific_trainer nor minimal_trainer is enabled. "
            "Set scientific_trainer.enabled = True (Phase B default)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Phase B training experiment.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    exp_id = args.experiment_id
    cfg_path = Path(args.config_path)
    run_root = Path(args.run_root)

    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(str(cfg_path))
    cfg["experiment_id"] = exp_id
    _sanity_check_cfg(cfg, exp_id)

    run_dir = run_root / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    enc = cfg.get("scientific_trainer", {}).get("model_name", "?").split("/")[-1]
    print(f"[phase_b.run_experiment] experiment_id = {exp_id}")
    print(f"[phase_b.run_experiment] schema        = {cfg.get('schema_id', '?')}")
    print(f"[phase_b.run_experiment] encoder       = {enc}")
    print(f"[phase_b.run_experiment] seed          = {cfg.get('seed', '?')}")
    print(f"[phase_b.run_experiment] run_dir       = {run_dir}")

    if cfg.get("minimal_trainer", {}).get("enabled"):
        run_minimal_training(cfg, exp_id, run_dir)
    else:
        run_scientific_training(cfg, exp_id, run_dir)

    print(f"[phase_b.run_experiment] DONE: {exp_id}")


if __name__ == "__main__":
    main()
