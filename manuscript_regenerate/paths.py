"""Artifact paths for manuscript regeneration."""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", REPO.parent / "projects" / "project_1")).resolve()


def step_paths(step: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for kind in ("data", "outputs", "figures", "reports"):
        d = OUTPUT_ROOT / kind / step
        d.mkdir(parents=True, exist_ok=True)
        out[kind] = d
    return out


STEPS = {
    "00": "00_civic_feasibility",
    "01": "01_corpus_relevance",
    "02": "02_evaluation_protocol",
    "03": "03_candidate_pool",
    "04": "04_pilot_study",
    "05": "05_marker_quality_gate",
    "10": "10_recipe_sweep_and_training",
    "20": "20_round2_diagnostic",
}

VOCAB = {
    "benchmark": "in-distribution benchmark (self-measured BioRED presence F1)",
    "kb": "out-of-distribution knowledge-base ranking (CIViC)",
    "task": "relation presence ranking",
    "question": "evaluation-validity question",
    "gene_drug": "gene-drug",
    "gene_disease": "gene-disease",
}
