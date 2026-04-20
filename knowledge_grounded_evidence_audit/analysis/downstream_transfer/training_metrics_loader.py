"""Load HR run_manifest + best-checkpoint metrics for a training run directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_run_metrics(run_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"run_dir": str(run_dir), "missing": False}
    mf = run_dir / "run_manifest.json"
    mb = run_dir / "metrics" / "metrics_best_checkpoint.json"
    if not mf.is_file():
        out["missing"] = True
        return out
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    st = manifest.get("scientific_trainer", {})
    out.update(
        {
            "experiment_id": manifest.get("experiment_id", ""),
            "schema_id": manifest.get("schema_id", ""),
            "seed": manifest.get("seed"),
            "encoder": st.get("model_name", ""),
            "architecture": "bert_sequence_classification",
            "max_length": st.get("max_length"),
            "batch_size": st.get("batch_size"),
            "learning_rate": st.get("learning_rate"),
            "max_updates": st.get("max_updates"),
            "schedule_resolved": manifest.get("resolved", {}).get("schedule", ""),
            "loss_mode": manifest.get("loss_mode", ""),
            "stages_executed": manifest.get("resolved", {}).get("stages_executed", []),
        }
    )
    if mb.is_file():
        best = json.loads(mb.read_text(encoding="utf-8"))
        stages = best.get("stages", [])
        t1 = next((s for s in stages if s.get("stage") == "T1"), None)
        t2 = next((s for s in stages if s.get("stage") == "T2"), None)
        out["hr_macro_f1_best_t1"] = t1.get("best_selection_score") if t1 else None
        out["hr_macro_f1_best_t2"] = t2.get("best_selection_score") if t2 else None
        out["hr_best_overall"] = max(
            [x for x in [out.get("hr_macro_f1_best_t1"), out.get("hr_macro_f1_best_t2")] if x is not None],
            default=None,
        )
    out["external_biored_macro_f1"] = "see_external_eval_bundle_if_present"
    out["external_bc5cdr_macro_f1"] = "see_external_eval_bundle_if_present"
    return out
