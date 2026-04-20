"""Deprecated minimal_trainer shim.

The original `minimal_trainer` was a Phase A developer-iteration path, never
exercised in any of the 120 Phase A runs (see `trainer_inventory/minimal_trainer_api_contract.md`).
Phase B does not support it; if a config accidentally requests it we fail loud
rather than silently fall through to scientific_trainer.
"""
from __future__ import annotations


def run_minimal_training(*args, **kwargs):  # noqa: ANN001, D401
    raise NotImplementedError(
        "minimal_trainer was a Phase A legacy developer-iteration path and is "
        "not supported in Phase B. Set cfg['minimal_trainer']['enabled'] = False "
        "(the Phase A default) and use run_scientific_training instead."
    )
