# -*- coding: utf-8 -*-
"""Filesystem roots for external evaluation outputs and training artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def code_root() -> Path:
    return Path(__file__).resolve().parents[2]


def external_eval_root() -> Path:
    env = os.environ.get("EXTERNAL_EVAL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / "projects" / "project_1" / "external_evaluation").resolve()


def project_data_root() -> Path:
    env = os.environ.get("PROJECT_1_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    p = (Path.home() / "projects" / "project_1").resolve()
    if p.is_dir():
        return p
    return (Path.home() / "project_1_data").resolve()


def training_processed() -> Path:
    return project_data_root() / "training_data_generation" / "data" / "processed"


def ft_runs_root() -> Path:
    return project_data_root() / "fine_tuning_experiments" / "runs"


def ensure_manifest_dirs(root: Path | None = None) -> dict[str, Path]:
    r = root or external_eval_root()
    manifests = r / "manifests"
    reports = r / "reports"
    tables = reports / "tables"
    audit = r / "audit"
    data_proc = r / "data" / "processed"
    for p in (manifests, reports, tables, audit, data_proc):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "root": r,
        "manifests": manifests,
        "reports": reports,
        "tables": tables,
        "audit": audit,
        "data_processed": data_proc,
    }


def mirror_reports_dir() -> Path:
    d = code_root() / "reports" / "external_evaluation"
    (d / "tables").mkdir(parents=True, exist_ok=True)
    return d
