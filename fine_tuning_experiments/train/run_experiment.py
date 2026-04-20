"""
run_experiment.py — Entry point for Phase A and Phase B training experiments.

Registers a sys.meta_path finder for pyc-only compiled modules BEFORE
loading any training code, then invokes run_scientific_training().

Usage:
  python3.11 -m fine_tuning_experiments.train.run_experiment \
      --experiment-id PA_PB_Sflat_s01 \
      --config-path /path/to/config.yaml \
      --run-root /path/to/runs/schema_exp
"""
from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

# ── Step 1: Register pyc-only finder BEFORE any other imports ────────────
# All fine_tuning_experiments.train.* modules exist only as .pyc in __pycache__.
# Python cannot discover them without this finder.

_TRAIN_DIR = Path(__file__).resolve().parent
_CACHE_DIR = _TRAIN_DIR / "__pycache__"
_PY_VERSION = f"cpython-{sys.version_info.major}{sys.version_info.minor}"


class _PycFinder:
    """Meta-path finder for pyc-only compiled modules anywhere in fine_tuning_experiments."""

    _FT_ROOT = _TRAIN_DIR.parent   # fine_tuning_experiments/ package root
    _FT_PKG  = "fine_tuning_experiments."
    _SKIP_FULL = frozenset({
        "fine_tuning_experiments.train.run_experiment",  # has real .py
    })

    def find_spec(self, fullname, path, target=None):
        if fullname in self._SKIP_FULL:
            return None
        if not fullname.startswith(self._FT_PKG):
            return None
        # e.g.  "fine_tuning_experiments.utils.paths"
        #     -> parts = ["utils", "paths"]
        #     -> pkg_dir = FT_ROOT / "utils"
        #     -> pyc     = FT_ROOT / "utils" / "__pycache__" / "paths.cpython-311.pyc"
        parts = fullname[len(self._FT_PKG):].split(".")
        pkg_dir = self._FT_ROOT
        for part in parts[:-1]:
            pkg_dir = pkg_dir / part
        mod_name = parts[-1]
        pyc = pkg_dir / "__pycache__" / f"{mod_name}.{_PY_VERSION}.pyc"
        if pyc.exists():
            return importlib.util.spec_from_file_location(fullname, str(pyc))
        return None


# Insert at position 0 — check before default finders
_finder = _PycFinder()
if not any(isinstance(f, _PycFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _finder)

# ── Step 2: Now import the trainer (all sub-imports handled by finder) ───
from fine_tuning_experiments.train.scientific_trainer import run_scientific_training


# ── Step 3: Argument parsing and dispatch ────────────────────────────────
import argparse
import json


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Phase A/B training experiment.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--config-path",   required=True)
    parser.add_argument("--run-root",      required=True)
    args = parser.parse_args()

    exp_id   = args.experiment_id
    cfg_path = Path(args.config_path)
    run_root = Path(args.run_root)

    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(str(cfg_path))
    cfg["experiment_id"] = exp_id

    run_dir = run_root / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_experiment] experiment_id = {exp_id}")
    print(f"[run_experiment] schema        = {cfg.get('schema_id', '?')}")
    print(f"[run_experiment] encoder       = {cfg.get('scientific_trainer', {}).get('model_name', '?').split('/')[-1]}")
    print(f"[run_experiment] seed          = {cfg.get('seed', '?')}")
    print(f"[run_experiment] run_dir       = {run_dir}")

    run_scientific_training(cfg, exp_id, run_dir)

    print(f"[run_experiment] DONE: {exp_id}")


if __name__ == "__main__":
    main()
