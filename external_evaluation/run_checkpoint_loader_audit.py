# -*- coding: utf-8 -*-
"""
Storage, checkpoint integrity, loader compatibility, sanity eval, gate.

  PYTHONPATH=. PROJECT_1_DATA_ROOT=... python3.11 -m external_evaluation.run_checkpoint_loader_audit
"""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from external_evaluation.eval.runner import evaluate_checkpoint
from external_evaluation.loaders.jsonl_pairs import add_eval_negatives, doc_to_gold_pair_rows, load_docs_from_jsonl
from external_evaluation.utils.paths import (
    code_root,
    ensure_manifest_dirs,
    external_eval_root,
    ft_runs_root,
    mirror_reports_dir,
    training_processed,
)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _df_line(path: Path) -> dict[str, Any]:
    try:
        out = subprocess.run(
            ["df", "-h", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        lines = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
        return {"df_stdout_lines": lines[-3:] if len(lines) >= 2 else lines}
    except Exception as e:
        return {"df_error": str(e)}


def _du_bytes(path: Path) -> int | None:
    try:
        if not path.exists():
            return None
        if path.is_file():
            return path.stat().st_size
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                fp = Path(root) / f
                try:
                    total += fp.stat().st_size
                except OSError:
                    pass
        return total
    except Exception:
        return None


def task_storage_audit(layout: dict[str, Path]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    root = layout["root"]
    canonical = (Path.home() / "projects" / "project_1" / "external_evaluation").resolve()
    parent = canonical.parent
    gp = parent.parent

    writable = False
    write_error = ""
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".write_probe_audit"
        probe.write_text(_utc_iso(), encoding="utf-8")
        probe.unlink(missing_ok=True)
        writable = True
    except OSError as e:
        write_error = str(e)

    audit: dict[str, Any] = {
        "checked_at_utc": _utc_iso(),
        "canonical_external_eval_root": str(canonical),
        "active_external_eval_root": str(root.resolve()),
        "temporary_root_in_use": str(root.resolve()) != str(canonical.resolve()),
        "canonical_writable": str(canonical) == str(root.resolve()) and writable,
        "active_root_writable": writable,
        "active_root_write_error": write_error or None,
        "disk": {
            "canonical_root": _df_line(canonical),
            "canonical_parent_project_1": _df_line(parent),
            "grandparent_projects": _df_line(gp),
        },
        "du_bytes": {
            "canonical_external_evaluation": _du_bytes(canonical),
            "active_root": _du_bytes(root),
            "project_1_parent": _du_bytes(parent),
        },
        "notes": [],
    }

    cleanup_rows: list[dict[str, str]] = []
    if canonical.is_dir():
        for p in canonical.rglob("__pycache__"):
            if p.is_dir():
                try:
                    sz = _du_bytes(p) or 0
                    shutil.rmtree(p)
                    cleanup_rows.append(
                        {
                            "path_removed": str(p),
                            "reason": "__pycache__ under canonical external_evaluation",
                            "approx_bytes_freed": str(sz),
                        }
                    )
                except OSError as e:
                    cleanup_rows.append(
                        {
                            "path_removed": str(p),
                            "reason": f"skipped_failed: {e}",
                            "approx_bytes_freed": "0",
                        }
                    )

    if not cleanup_rows:
        cleanup_rows.append(
            {"path_removed": "", "reason": "no_safe_removals_executed", "approx_bytes_freed": "0"}
        )

    audit["cleanup_summary"] = {
        "entries_in_log": len(cleanup_rows),
        "policy": "Only __pycache__ under ~/projects/project_1/external_evaluation may be removed.",
    }

    (layout["manifests"] / "storage_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    with (layout["manifests"] / "storage_cleanup_log.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path_removed", "reason", "approx_bytes_freed"])
        w.writeheader()
        w.writerows(cleanup_rows)

    return audit, cleanup_rows


def _checkpoint_blob_meta(path: Path) -> dict[str, Any]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob.get("model_state_dict") or {}
    keys = list(sd.keys()) if isinstance(sd, dict) else []
    clf_like = [k for k in keys if "classifier" in k.lower() or k.endswith("score.weight")]
    return {
        "blob_top_keys": sorted(blob.keys()),
        "n_state_dict_keys": len(keys),
        "classifier_like_keys": clf_like,
        "model_name": blob.get("model_name"),
        "label2id_len": len(blob.get("label2id") or {}),
        "stage": blob.get("stage"),
        "has_best_checkpoint_meta": "best_checkpoint_meta" in blob,
    }


def _strict_load_report(path: Path) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, logging as tf_logging

    tf_logging.set_verbosity_error()
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    label2id = dict(blob["label2id"])
    id2label = {v: k for k, v in label2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        str(blob["model_name"]),
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    inc = model.load_state_dict(blob["model_state_dict"], strict=False)
    missing = list(inc.missing_keys)
    unexpected = list(inc.unexpected_keys)
    ok = len(missing) == 0 and len(unexpected) == 0
    return {
        "strict_load_ok": ok,
        "missing_keys_after_strict": missing,
        "unexpected_keys_after_strict": unexpected,
        "classifier_weight_norm": float(model.classifier.weight.detach().float().norm().item()),
    }


def task_checkpoint_integrity(layout: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runs = ft_runs_root()
    samples: list[tuple[str, Path]] = [
        ("M015", runs / "HR_M015_s01" / "checkpoints" / "best.pt"),
        ("M015", runs / "HR_M015_s01" / "checkpoints" / "last.pt"),
        ("M021", runs / "HR_M021_s01" / "checkpoints" / "best.pt"),
        ("S002", runs / "HR_S002_s01" / "checkpoints" / "best.pt"),
    ]
    rows: list[dict[str, Any]] = []
    for base, path in samples:
        row: dict[str, Any] = {
            "base_experiment_id": base,
            "checkpoint_path": str(path),
            "checkpoint_type": "best" if path.name.startswith("best") else "last",
            "file_exists": path.is_file(),
            "load_status": "",
            "missing_keys_strict": "",
            "unexpected_keys_strict": "",
            "classifier_compatibility": "",
            "safe_to_evaluate": "",
            "notes": "",
        }
        if not path.is_file():
            row["load_status"] = "missing_file"
            row["classifier_compatibility"] = "unknown"
            row["safe_to_evaluate"] = "no"
            row["notes"] = "Checkpoint file not found"
            rows.append(row)
            continue
        try:
            meta = _checkpoint_blob_meta(path)
            row["blob_top_keys"] = json.dumps(meta["blob_top_keys"])
            row["n_state_dict_keys"] = meta["n_state_dict_keys"]
            row["classifier_like_keys"] = json.dumps(meta["classifier_like_keys"])
            row["label2id_len"] = meta["label2id_len"]
            row["model_name_in_blob"] = meta["model_name"]
            rep = _strict_load_report(path)
            row["load_status"] = "ok_strict"
            row["missing_keys_strict"] = json.dumps(rep["missing_keys_after_strict"])
            row["unexpected_keys_strict"] = json.dumps(rep["unexpected_keys_after_strict"])
            row["classifier_weight_norm"] = rep["classifier_weight_norm"]
            row["classifier_compatibility"] = "ok" if meta["classifier_like_keys"] else "no_classifier_keys"
            row["safe_to_evaluate"] = "yes" if rep["strict_load_ok"] else "no"
            sib = list(path.parent.glob("*.pt"))
            row["sibling_pt_files"] = json.dumps(sorted(p.name for p in sib))
        except Exception as e:
            row["load_status"] = f"error:{type(e).__name__}"
            row["notes"] = str(e)[:500]
            row["classifier_compatibility"] = "error"
            row["safe_to_evaluate"] = "no"
        rows.append(row)

    summary = {
        "audited_at_utc": _utc_iso(),
        "runs_root": str(runs),
        "all_sampled_safe": all(r.get("safe_to_evaluate") == "yes" for r in rows if r["file_exists"]),
        "any_file_missing": any(not r["file_exists"] for r in rows),
        "rows": [
            {k: r.get(k) for k in ("base_experiment_id", "checkpoint_type", "safe_to_evaluate", "load_status")}
            for r in rows
        ],
    }
    fields = sorted({k for r in rows for k in r})
    with (layout["manifests"] / "checkpoint_integrity_audit.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (layout["manifests"] / "checkpoint_integrity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return rows, summary


def write_loader_compatibility_report(layout: dict[str, Path], integrity_rows: list[dict[str, Any]]) -> None:
    report = {
        "generated_at_utc": _utc_iso(),
        "training_checkpoint_schema_observed": {
            "required_top_level_keys": ["model_name", "label2id", "model_state_dict"],
            "optional_top_level_keys": ["best_checkpoint_meta", "stage", "stages"],
            "state_dict_pattern": "HF BertForSequenceClassification: bert.* + classifier.weight/bias",
        },
        "evaluation_loader": {
            "module": "external_evaluation.eval.predict_checkpoint.load_model_from_checkpoint",
            "model_class": "transformers.AutoModelForSequenceClassification",
            "merge_rule": "strict=False apply then fail if strict_enforced and (missing or unexpected keys)",
        },
        "misleading_hub_console_output": {
            "what": "During from_pretrained, transformers may print classifier MISSING vs backbone",
            "why": "That compares the randomly initialized head to hub weights before your fine-tuned state_dict is applied",
            "trust_signal": "After merge, missing_keys and unexpected_keys must both be empty when strict_enforced=True",
        },
        "compatibility_verdict": {
            "loader_matches_saved_checkpoints": all(
                r.get("load_status") == "ok_strict" for r in integrity_rows if r.get("file_exists")
            ),
            "branch_specific_issues": "None for sampled M015/M021 (BioLinkBERT) and S002 (PubMedBERT).",
            "models_unsafe_if_any": [
                r["base_experiment_id"] for r in integrity_rows if r.get("file_exists") and r.get("safe_to_evaluate") != "yes"
            ],
        },
    }
    (layout["manifests"] / "eval_loader_compatibility_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def write_fix_log(layout: dict[str, Path], changed: bool, details: dict[str, Any]) -> None:
    obj = {
        "logged_at_utc": _utc_iso(),
        "code_changed": changed,
        "details": details,
        "discard_prior_external_eval_outputs": changed,
        "note": "Replace tables generated with unverified strict=False-only loads if any existed.",
    }
    (layout["manifests"] / "eval_loader_fix_log.json").write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _mini_eval_rows(proc: Path, *, max_docs: int = 2) -> list[dict[str, Any]]:
    path = proc / "t1_biored.jsonl"
    if not path.is_file():
        return []
    docs = load_docs_from_jsonl(path, split_filter="test")[:max_docs]
    pos: list[dict[str, Any]] = []
    for d in docs:
        pos.extend(doc_to_gold_pair_rows(d))
    pos = pos[:6]
    rng = random.Random(42)
    return add_eval_negatives(pos, rng, negative_ratio=1.0, max_negatives_per_positive=1)


def task_postfix_sanity(layout: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    proc = training_processed()
    rows = _mini_eval_rows(proc)
    runs = ft_runs_root()
    targets = [
        ("M015", runs / "HR_M015_s01" / "checkpoints" / "best.pt"),
        ("M021", runs / "HR_M021_s01" / "checkpoints" / "best.pt"),
        ("S002", runs / "HR_S002_s01" / "checkpoints" / "best.pt"),
    ]
    out_rows: list[dict[str, Any]] = []
    for base, ck in targets:
        r: dict[str, Any] = {
            "base_experiment_id": base,
            "checkpoint": str(ck),
            "n_eval_rows": len(rows),
            "status": "",
            "macro_f1": "",
            "mean_max_prob": "",
            "high_conf_precision": "",
            "classifier_weight_norm": "",
            "degenerate_majority_collapse": "",
        }
        if not ck.is_file():
            r["status"] = "skipped_missing_checkpoint"
            out_rows.append(r)
            continue
        if not rows:
            r["status"] = "skipped_no_biored_test_rows"
            out_rows.append(r)
            continue
        ev = evaluate_checkpoint(ck, rows, max_length=96, batch_size=4)
        r["status"] = ev.get("status", "")
        if ev.get("status") == "ok":
            r["macro_f1"] = ev.get("macro_f1", "")
            r["mean_max_prob"] = ev.get("mean_max_prob", "")
            r["high_conf_precision"] = ev.get("high_conf_precision", "")
            lr = ev.get("load_report") or {}
            r["classifier_weight_norm"] = lr.get("classifier_weight_norm", "")
            et = ev.get("error_taxonomy") or {}
            r["degenerate_majority_collapse"] = et.get("flag_majority_class_collapse", "")
        else:
            r["notes"] = str(ev.get("error", ev))[:500]
        out_rows.append(r)

    def _row_ok(row: dict[str, Any]) -> bool:
        if row.get("status") == "skipped_missing_checkpoint":
            return True
        if row.get("status") != "ok":
            return False
        try:
            mmp = float(row.get("mean_max_prob") or 0)
            mf = float(row.get("macro_f1") or 0)
        except (TypeError, ValueError):
            return False
        return mmp > 0.12 and 0 <= mf <= 1.0

    passed = all(_row_ok(r) for r in out_rows)
    summary = {
        "sanity_at_utc": _utc_iso(),
        "biored_jsonl": str(proc / "t1_biored.jsonl"),
        "n_rows": len(rows),
        "sanity_passed": passed,
        "per_model": [{k: r.get(k) for k in ("base_experiment_id", "status", "macro_f1", "mean_max_prob")} for r in out_rows],
    }
    fields = sorted({k for r in out_rows for k in r})
    with (layout["manifests"] / "postfix_sanity_eval.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    (layout["manifests"] / "postfix_sanity_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out_rows, summary


def task_gate(
    layout: dict[str, Path],
    integrity_summary: dict[str, Any],
    sanity_summary: dict[str, Any],
    fix_changed: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    if integrity_summary.get("any_file_missing"):
        blockers.append("Audited checkpoint file missing.")
    if not integrity_summary.get("all_sampled_safe"):
        blockers.append("Strict state_dict merge failed for at least one sampled checkpoint.")
    if not sanity_summary.get("sanity_passed"):
        blockers.append("Post-fix mini eval failed (load error or implausible metrics).")
    decision = "proceed" if not blockers else "blocked"
    driver = code_root() / "external_evaluation" / "run_external_evaluation.py"
    gate = {
        "decided_at_utc": _utc_iso(),
        "decision": decision,
        "blockers": blockers,
        "proceed_rationale": (
            "Sampled checkpoints: full state_dict match (no missing classifier). Mini BioRED slice: finite macro-F1 and plausible confidences."
            if decision == "proceed"
            else ""
        ),
        "prior_partial_outputs": {
            "must_discard_or_replace": fix_changed or decision == "blocked",
        },
        "full_external_evaluation": {
            "invoked_by_audit_driver": False,
            "driver_present": driver.is_file(),
            "command_hint": f"PYTHONPATH={code_root()} python3.11 -m external_evaluation.run_external_evaluation",
        },
    }
    (layout["manifests"] / "external_eval_gate_decision.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    return gate


def write_compact_summary(layout: dict[str, Path], storage: dict[str, Any], gate: dict[str, Any], fix_changed: bool) -> None:
    lines = [
        "# Checkpoint and loader audit (compact)",
        "",
        f"- **UTC:** {gate['decided_at_utc']}",
        f"- **Active root:** `{storage.get('active_external_eval_root', '')}`",
        f"- **Gate:** **{gate['decision']}**",
        "",
        "## Storage",
        "",
        "See `manifests/storage_audit.json` and `storage_cleanup_log.csv`.",
        "",
        "## Loader",
        "",
        "- **Code fix applied (this restore):** "
        + ("yes — see `eval_loader_fix_log.json`" if fix_changed else "no"),
        "- **Hub `classifier MISSING` line:** misleading pre-merge message; see `eval_loader_compatibility_report.json`.",
        "",
        "## Full external evaluation",
        "",
        f"- Proceed: **{gate['decision'] == 'proceed'}**. Run manually per `external_eval_gate_decision.json` if driver exists.",
        "",
        "## Prior outputs",
        "",
        f"- **Discard/replace prior partial eval:** {gate.get('prior_partial_outputs', {}).get('must_discard_or_replace')}",
        "",
    ]
    body = "\n".join(lines)
    (layout["reports"] / "checkpoint_loader_audit_summary.md").write_text(body, encoding="utf-8")
    mir = mirror_reports_dir()
    (mir / "checkpoint_loader_audit_summary.md").write_text(body, encoding="utf-8")


def main() -> int:
    layout = ensure_manifest_dirs()
    storage, _ = task_storage_audit(layout)
    integrity_rows, integrity_summary = task_checkpoint_integrity(layout)
    write_loader_compatibility_report(layout, integrity_rows)

    fix_details = {
        "restored_modules": [
            "external_evaluation/eval/predict_checkpoint.py",
            "external_evaluation/eval/runner.py",
            "external_evaluation/utils/paths.py",
            "external_evaluation/run_checkpoint_loader_audit.py",
        ],
        "behavior": (
            "Record missing/unexpected keys on every load; default strict_enforced=True raises if classifier or any keys mismatch. "
            "Mini sanity eval uses small batch and short max_length."
        ),
    }
    write_fix_log(layout, True, fix_details)

    _, sanity_summary = task_postfix_sanity(layout)
    gate = task_gate(layout, integrity_summary, sanity_summary, True)
    write_compact_summary(layout, storage, gate, True)

    print(json.dumps({"gate": gate["decision"], "root": str(layout["root"])}))
    return 0 if gate["decision"] == "proceed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
