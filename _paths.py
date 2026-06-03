"""Output path layout for the preparation pipeline (non-code artifacts)."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(
    os.environ.get("OUTPUT_ROOT", REPO_ROOT.parent / "projects" / "project_1")
).resolve()


def step_dirs(step_name: str) -> dict[str, Path]:
    """Return data/outputs/figures/reports/runs directories for a step."""
    out: dict[str, Path] = {}
    for kind in ("data", "outputs", "figures", "reports", "runs"):
        d = OUTPUT_ROOT / kind / step_name
        d.mkdir(parents=True, exist_ok=True)
        out[kind] = d
    return out
