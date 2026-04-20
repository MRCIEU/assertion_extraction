#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""
Write DrugProt benchmark-gap diagnosis artifacts (no torch / no GPU).

  PYTHONPATH=<project_1_root> python3.11 -m external_evaluation.write_drugprot_diagnosis
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from external_evaluation.utils.benchmark_diagnosis import write_drugprot_diagnosis_artifacts
from external_evaluation.utils.paths import ensure_manifest_dirs, training_processed


def main() -> int:
    proc = training_processed()
    if not proc.is_dir():
        print(f"Missing processed training dir: {proc}", file=sys.stderr)
        return 1
    layout = ensure_manifest_dirs()
    write_drugprot_diagnosis_artifacts(proc, layout["manifests"], layout["tables"])
    print(json.dumps({"ok": True, "root": str(layout["root"]), "wrote": "drugprot_gap_diagnosis"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
