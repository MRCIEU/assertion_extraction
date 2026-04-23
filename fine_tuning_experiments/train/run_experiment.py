"""
run_experiment.py — Entry point for Phase A and Phase B training experiments.

Loads a YAML config, resolves the trainer (scientific_trainer), and dispatches
the training run.  The same entry point is used for Phase A (schema selection)
and Phase B (configuration factorial); the only thing that changes is the
config YAML passed in.

Usage:
    python3.11 -m fine_tuning_experiments.train.run_experiment \
        --experiment-id PA_PB_Sflat_s01 \
        --config-path /path/to/config.yaml \
        --run-root /path/to/runs/schema_exp

Trainer source: fine_tuning_experiments/phase_b/trainer/scientific_trainer.py
(clean Python source, version-controlled; replaces the earlier pyc-only path).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fine_tuning_experiments.phase_b.trainer.scientific_trainer import (
    run_scientific_training,
)


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Phase A or Phase B training experiment."
    )
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

    run_dir = run_root / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    st = cfg.get("scientific_trainer", {}) or {}
    encoder = (st.get("model_name") or "?").split("/")[-1]
    print(f"[run_experiment] experiment_id = {exp_id}")
    print(f"[run_experiment] schema        = {cfg.get('schema_id', '?')}")
    print(f"[run_experiment] encoder       = {encoder}")
    print(f"[run_experiment] seed          = {cfg.get('seed', '?')}")
    print(f"[run_experiment] run_dir       = {run_dir}")

    run_scientific_training(cfg, exp_id, run_dir)

    print(f"[run_experiment] DONE: {exp_id}")


if __name__ == "__main__":
    main()
