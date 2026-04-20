# -*- coding: utf-8 -*-
"""
Full external evaluation driver (post–strict-loader audit).

**Do not run ``--run-eval`` on the login node** — submit GPU jobs instead:

  # One-time archive + registries (light metadata only on login):
  PYTHONPATH=. PROJECT_1_DATA_ROOT=$HOME/projects/project_1 \\
    python3.11 -m external_evaluation.run_external_evaluation --prepare-reset

  # Full evaluation on GPU:
  PYTHONPATH=. python3.11 -m external_evaluation.run_external_evaluation --submit-sbatch

  # Small GPU smoke test (writes under ``external_evaluation/smoke_runs/<jobid>/``):
  PYTHONPATH=. python3.11 -m external_evaluation.run_external_evaluation --submit-sbatch-smoke

  # DrugProt gap diagnosis only (no torch; writes manifests + tables):
  PYTHONPATH=. python3.11 -m external_evaluation.write_drugprot_diagnosis

  ``--run-eval`` is intended for Slurm GPU nodes / interactive GPU sessions only.

Env:
  EXT_EVAL_MAX_EXAMPLES — cap positives per source before negatives (default 2500)
  EXT_EVAL_SEEDS — comma list (default 1,2,3,4,5)
  EXT_EVAL_SKIP_RESET — if 1, skip archive reset inside --run-eval
  EXT_EVAL_SKIP_MIRROR — if 1, do not copy reports into ``project_1/reports/`` (smoke default)
  EXTERNAL_EVAL_ROOT, PROJECT_1_DATA_ROOT
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time

import torch
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

_THIS_FILE = Path(__file__).resolve()
if _THIS_FILE.parents[2].name == "projects":
    _KG_INF = _THIS_FILE.parents[3] / "project_1" / "knowledge_grounded_evidence_audit"
else:
    _KG_INF = _THIS_FILE.parents[1] / "knowledge_grounded_evidence_audit"
if _KG_INF.is_dir() and str(_KG_INF) not in sys.path:
    sys.path.insert(0, str(_KG_INF))
from inference.predict_checkpoint import load_model_from_checkpoint
from inference.runner import evaluate_checkpoint, evaluate_rows_on_loaded_model
from external_evaluation.loaders.jsonl_pairs import add_eval_negatives
from external_evaluation.loaders.pair_streams import (
    pairing_drug_disease,
    pairing_drug_gene,
    pairing_gene_disease,
    pairing_variant_disease,
    rows_to_eval_input,
    stream_pair_rows,
)
from external_evaluation.utils.benchmark_diagnosis import write_drugprot_diagnosis_artifacts
from external_evaluation.utils.paths import (
    code_root,
    ensure_manifest_dirs,
    external_eval_root,
    ft_runs_root,
    mirror_reports_dir,
    project_data_root,
    training_processed,
)

# ---------------------------------------------------------------------------
# Shortlist + optional models (registry CSV is authoritative)
# ---------------------------------------------------------------------------
MODEL_REGISTRY_ROWS: list[dict[str, str]] = [
    {
        "base_experiment_id": "M015",
        "rerun_group": "HR",
        "rationale_for_inclusion": "Primary BioLinkBERT-style winner candidate.",
        "role": "primary",
        "checkpoint_selection_policy": "checkpoints/best.pt per seed HR_M015_s{seed:02d}",
    },
    {
        "base_experiment_id": "M003",
        "rerun_group": "HR",
        "rationale_for_inclusion": "PubMedBERT primary-line comparison.",
        "role": "primary",
        "checkpoint_selection_policy": "checkpoints/best.pt per seed HR_M003_s{seed:02d}",
    },
    {
        "base_experiment_id": "M021",
        "rerun_group": "HR",
        "rationale_for_inclusion": "Shared-encoder / multitask representative.",
        "role": "primary",
        "checkpoint_selection_policy": "checkpoints/best.pt per seed HR_M021_s{seed:02d}",
    },
    {
        "base_experiment_id": "M010",
        "rerun_group": "HR",
        "rationale_for_inclusion": "T4-related configuration for comparison.",
        "role": "primary",
        "checkpoint_selection_policy": "checkpoints/best.pt per seed HR_M010_s{seed:02d}",
    },
    {
        "base_experiment_id": "M025",
        "rerun_group": "HR",
        "rationale_for_inclusion": "Parameter-efficient / adapter-style line.",
        "role": "primary",
        "checkpoint_selection_policy": "checkpoints/best.pt per seed HR_M025_s{seed:02d}",
    },
    {
        "base_experiment_id": "S002",
        "rerun_group": "HR",
        "rationale_for_inclusion": "Secondary weighted-CE branch validation.",
        "role": "primary",
        "checkpoint_selection_policy": "checkpoints/best.pt per seed HR_S002_s{seed:02d}",
    },
                {
                    "base_experiment_id": "M005",
        "rerun_group": "HR",
        "rationale_for_inclusion": "Negative-transfer / T3 control (selection fairness).",
        "role": "control",
        "checkpoint_selection_policy": "checkpoints/best.pt per seed HR_M005_s{seed:02d}",
                },
                {
                    "base_experiment_id": "M026",
        "rerun_group": "HR",
        "rationale_for_inclusion": "Weighted-CE diagnostic control.",
        "role": "diagnostic",
        "checkpoint_selection_policy": "checkpoints/best.pt per seed HR_M026_s{seed:02d}",
    },
    {
        "base_experiment_id": "S001",
        "rerun_group": "HR",
        "rationale_for_inclusion": "Optional secondary baseline if checkpoints exist.",
        "role": "optional",
        "checkpoint_selection_policy": "checkpoints/best.pt per seed HR_S001_s{seed:02d}",
    },
    {
        "base_experiment_id": "M009",
        "rerun_group": "HR",
        "rationale_for_inclusion": "Optional shared-encoder variant if checkpoints exist.",
        "role": "optional",
        "checkpoint_selection_policy": "checkpoints/best.pt per seed HR_M009_s{seed:02d}",
    },
]

BENCHMARK_SOURCES: list[dict[str, Any]] = [
    {
        "evaluation_source_id": "biored_official_test_pairs",
            "evidence_type": "split_external",
        "jsonl_relative": "t1_biored.jsonl",
        "split": {"test"},
        "layer": "A",
    },
    {
        "evaluation_source_id": "bc5cdr_official_test_pairs",
            "evidence_type": "split_external",
        "jsonl_relative": "t1_bc5cdr.jsonl",
        "split": {"test"},
        "layer": "A",
    },
    {
        "evaluation_source_id": "drugprot_official_test_pairs",
            "evidence_type": "split_external",
        "jsonl_relative": "t1_drugprot.jsonl",
        "split": {"test"},
        "layer": "A",
    },
]

REALISM_SOURCES: list[dict[str, Any]] = [
    {
        "evaluation_source_id": "bc5cdr_oncology_cancer_slice",
            "evidence_type": "realism_probe",
        "jsonl_relative": "t2_bc5cdr_cancer_slice.jsonl",
        "split": {"test"},
        "layer": "B",
    },
    {
        "evaluation_source_id": "biored_oncology_projected_bridge",
        "evidence_type": "realism_probe",
        "jsonl_relative": "t2_supervised_oncology_bridge_merged.jsonl",
        "split": None,
        "layer": "B",
    },
]

ONTOLOGY_REGISTRY_ROWS: list[dict[str, str]] = [
    {
        "evaluation_source_id": "bronco_english_anchor_inventory",
            "evidence_type": "ontology_external",
        "path_or_note": "BRONCO-style anchors: protocol / schema stress only; not span-relation gold rows.",
        "allowed_uses": "relation_world_probe;schema_adequacy",
        "layer": "B_anchor",
    },
    {
        "evaluation_source_id": "precision_oncology_concept_anchor",
        "evidence_type": "ontology_external",
        "path_or_note": "Precision oncology concept resources: entity stress, not relation benchmark.",
        "allowed_uses": "entity_stress;schema_adequacy",
        "layer": "B_anchor",
    },
]

WEAK_PROBE_ROWS: list[dict[str, str]] = [
    {
        "evaluation_source_id": "civicmine_weak_sentences",
            "evidence_type": "weak_probe",
        "path_or_note": "CIViCmine-derived text: diagnostic / stress only; never benchmark gold.",
        "allowed_uses": "semantic_stress;interpretation_probe",
        "layer": "D_stress",
    },
]


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seeds() -> list[int]:
    raw = os.environ.get("EXT_EVAL_SEEDS", "1,2,3,4,5")
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _max_pos_per_source() -> int:
    return int(os.environ.get("EXT_EVAL_MAX_EXAMPLES", "2500"))


def _checkpoint(base: str, seed: int) -> Path:
    return ft_runs_root() / f"HR_{base}_s{seed:02d}" / "checkpoints" / "best.pt"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_model_registry(manifests: Path) -> None:
    path = manifests / "evaluation_model_registry.csv"
    fields = [
        "base_experiment_id",
        "rerun_group",
        "rationale_for_inclusion",
        "role",
        "checkpoint_selection_policy",
    ]
    _write_csv(path, MODEL_REGISTRY_ROWS, fields)


def write_source_registry(proc: Path) -> None:
    rows: list[dict[str, str]] = []
    for s in BENCHMARK_SOURCES:
        rows.append(
            {
                "evaluation_source_id": s["evaluation_source_id"],
                "evidence_type": s["evidence_type"],
                "jsonl_relative": s["jsonl_relative"],
                "split_policy": json.dumps(sorted(s["split"])),
                "layer": s["layer"],
                "notes": "Layer A primary benchmark; official test split pairs.",
            }
        )
    for s in REALISM_SOURCES:
        sp = json.dumps(sorted(s["split"])) if s["split"] else "all_splits_capped"
        rows.append(
            {
                "evaluation_source_id": s["evaluation_source_id"],
                "evidence_type": s["evidence_type"],
                "jsonl_relative": s["jsonl_relative"],
                "split_policy": sp,
                "layer": s["layer"],
                "notes": "Realism probe — not merged into primary_external_results.",
            }
        )
    for s in ONTOLOGY_REGISTRY_ROWS:
        rows.append(
            {
                "evaluation_source_id": s["evaluation_source_id"],
                "evidence_type": s["evidence_type"],
                "jsonl_relative": "",
                "split_policy": "",
                "layer": s["layer"],
                "notes": s["path_or_note"],
            }
        )
    for s in WEAK_PROBE_ROWS:
        rows.append(
            {
                "evaluation_source_id": s["evaluation_source_id"],
                "evidence_type": s["evidence_type"],
                "jsonl_relative": "",
                "split_policy": "",
                "layer": s["layer"],
                "notes": s["path_or_note"],
            }
        )
    _write_csv(
        proc / "evaluation_source_registry.csv",
        rows,
        [
            "evaluation_source_id",
            "evidence_type",
            "jsonl_relative",
            "split_policy",
            "layer",
            "notes",
        ],
    )


def write_protocol_files(proc: Path) -> None:
    strict = {
        "protocol_id": "strict_realism_v1",
        "chosen_level": "pair_level_strict",
        "definition": (
            "A prediction counts as strictly correct only if the operational pair-level label matches gold "
            "under the checkpoint label space (head/tail entity families and relation family as encoded in mapped_label)."
        ),
        "not_claimed": [
            "span_exact_match",
            "evidence_sentence_alignment",
            "full_calibration",
        ],
        "negative_sampling_eval": "Same-document random non-gold pairs labeled __NEGATIVE__; RNG seed fixed per protocol (42).",
        "updated_at_utc": _utc(),
    }
    (proc / "strict_realism_protocol.json").write_text(json.dumps(strict, indent=2) + "\n", encoding="utf-8")

    fair = {
        "controls": ["M005", "M026"],
        "mitigates": [
            "selection_bias_from_only_promoting_internal_winners",
            "mistaking_weighted_loss_or_T3_mixture_for_generalization",
        ],
        "updated_at_utc": _utc(),
    }
    (proc / "selection_fairness_note.json").write_text(json.dumps(fair, indent=2) + "\n", encoding="utf-8")


def _df_snapshot(path: Path) -> list[str]:
    try:
        out = subprocess.run(
            ["df", "-h", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        lines = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
        return lines[-2:] if len(lines) >= 2 else lines
    except Exception as e:
        return [f"df_error:{e}"]


def refresh_storage_audit(layout: dict[str, Path], extra: dict[str, Any] | None = None) -> None:
    root = layout["root"]
    snap = {
        "refreshed_at_utc": _utc(),
        "external_eval_root": str(root),
        "df_lines": _df_snapshot(root),
    }
    if extra:
        snap.update(extra)
    path = layout["manifests"] / "storage_audit.json"
    prev: dict[str, Any] = {}
    if path.is_file():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    prev["latest_refresh"] = snap
    path.write_text(json.dumps(prev, indent=2) + "\n", encoding="utf-8")


def append_storage_cleanup_log(layout: dict[str, Path], rows: list[dict[str, str]]) -> None:
    p = layout["manifests"] / "storage_cleanup_log.csv"
    fields = ["timestamp_utc", "path_removed", "reason", "approx_bytes_or_note"]
    exists = p.is_file()
    with p.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            w.writeheader()
        for r in rows:
            r = {**r, "timestamp_utc": r.get("timestamp_utc", _utc())}
            w.writerow(r)


def prior_partial_output_reset(layout: dict[str, Path]) -> dict[str, Any]:
    """Archive pre-audit external evaluation products (not checkpoint audit manifests)."""
    root = layout["root"]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = root / "archive" / f"pre_strict_loader_invalid_{ts}"
    archive.mkdir(parents=True, exist_ok=True)

    moved: list[str] = []
    candidates: list[Path] = []
    tables = layout["tables"]
    if tables.is_dir():
        for f in tables.glob("*.csv"):
            candidates.append(f)
    rep = layout["reports"] / "external_evaluation_report.md"
    if rep.is_file():
        candidates.append(rep)
    audit_jsonl = layout["audit"]
    if audit_jsonl.is_dir():
        for f in audit_jsonl.glob("*.jsonl"):
            candidates.append(f)

    for src in candidates:
        try:
            dest = archive / src.name
            shutil.move(str(src), str(dest))
            moved.append(str(src))
        except OSError as e:
            moved.append(f"FAILED:{src}:{e}")

    log_obj = {
        "reset_at_utc": _utc(),
        "archive_directory": str(archive),
        "paths_moved_or_attempted": moved,
        "invalid_prior_outputs_existed": len(moved) > 0,
        "regeneration_clean": True,
        "notes": "Checkpoint-loader audit artifacts under manifests/ were not moved.",
    }
    (layout["manifests"] / "prior_partial_output_reset.json").write_text(
        json.dumps(log_obj, indent=2) + "\n", encoding="utf-8"
    )

    append_storage_cleanup_log(
        layout,
        [
            {
                "path_removed": str(archive),
                "reason": "prior_partial_output_reset_archive",
                "approx_bytes_or_note": str(len(moved)),
            }
        ],
    )
    return log_obj


def build_eval_set_for_source(
    proc: Path,
    spec: dict[str, Any],
    *,
    max_pos: int,
    neg_rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    path = proc / spec["jsonl_relative"]
    split = set(spec["split"]) if spec.get("split") else None
    rows, st = stream_pair_rows(path, max_pairs=max_pos, source_split_in=split)
    if not rows:
        return [], {**st, "n_examples": 0}
    full = add_eval_negatives(rows, neg_rng, negative_ratio=2.0, max_negatives_per_positive=3)
    # strip heavy keys for memory
    for r in full:
        r.pop("ent_by_id", None)
        r.pop("gold_pairs", None)
        r.pop("ent_ids_list", None)
    ev = rows_to_eval_input(full)
    st_out = {
        "n_documents": st["n_documents"],
        "n_examples": len(full),
        "n_positive_instances": sum(1 for r in full if r.get("label") != "__NEGATIVE__"),
        "n_negative_instances": sum(1 for r in full if r.get("label") == "__NEGATIVE__"),
    }
    return ev, st_out


def run_benchmark_matrix(
    bases: list[str],
    role_by_base: dict[str, str],
    sources: list[dict[str, Any]],
    proc: Path,
    prebuilt: dict[str, tuple[list[dict[str, Any]], dict[str, int]]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """One checkpoint load per (base, seed); score all sources on the loaded model."""
    seeds = _seeds()
    out: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    batch = int(os.environ.get("EXT_EVAL_BATCH_SIZE", "16"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for base in bases:
        for seed in seeds:
            ck = _checkpoint(base, seed)
            if not ck.is_file():
                for spec in sources:
                    sid = spec["evaluation_source_id"]
                    evi = spec["evidence_type"]
                    out[(base, sid, evi)].append(
                        {
                            "seed": seed,
                            "status": "missing_checkpoint",
                            "checkpoint": str(ck),
                            "base_experiment_id": base,
                        }
                    )
                    continue
                try:
                model, tok, l2i, labels, lr = load_model_from_checkpoint(
                    ck, device, state_dict_strict=True
                )
            except Exception as e:
                for spec in sources:
                    sid = spec["evaluation_source_id"]
                    evi = spec["evidence_type"]
                    out[(base, sid, evi)].append(
                        {
                            "seed": seed,
                            "status": "load_failed",
                            "error": str(e),
                            "checkpoint": str(ck),
                            "base_experiment_id": base,
                        }
                    )
                    continue
            for spec in sources:
                sid = spec["evaluation_source_id"]
                evi = spec["evidence_type"]
                rows, _st = prebuilt[sid]
                if not rows:
                    out[(base, sid, evi)].append(
                        {"seed": seed, "status": "no_rows", "base_experiment_id": base}
                    )
                    continue
                ev = evaluate_rows_on_loaded_model(
                    model,
                    tok,
                    l2i,
                    labels,
                    lr,
                    ck,
                    rows,
                    device=device,
                    max_length=384,
                    batch_size=batch,
                )
                ev["seed"] = seed
                ev["base_experiment_id"] = base
                ev["role"] = role_by_base.get(base, "")
                out[(base, sid, evi)].append(ev)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return out


def run_subset_matrix(
    bases: list[str],
    role_by_base: dict[str, str],
    proc: Path,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    seeds = _seeds()
    batch = int(os.environ.get("EXT_EVAL_BATCH_SIZE", "16"))
    filters = {
        "pairing_gene_disease": pairing_gene_disease,
        "pairing_variant_disease": pairing_variant_disease,
        "pairing_drug_gene": pairing_drug_gene,
        "pairing_drug_disease": pairing_drug_disease,
    }
    test_files = ["t1_biored.jsonl", "t1_bc5cdr.jsonl", "t1_drugprot.jsonl"]
    max_each = max(1, _max_pos_per_source() // 3)
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    subset_payload: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for subset_name, pfilter in filters.items():
        neg_rng = random.Random(43 + abs(hash(subset_name)) % 1_000_000)
        all_rows: list[dict[str, Any]] = []
        st_docs: set[str] = set()
        last_stream_stats: dict[str, int] = {"n_documents": 0}
        for fn in test_files:
            path = proc / fn
            if not path.is_file():
                continue
            rows, st = stream_pair_rows(
                path,
                max_pairs=max_each,
                source_split_in={"test"},
                pairing_filter=pfilter,
            )
            last_stream_stats = st
            for r in rows:
                did = str(r.get("doc_id") or r.get("sample_id") or "")
                if did:
                    st_docs.add(did)
            all_rows.extend(rows)
        full = add_eval_negatives(all_rows, neg_rng, negative_ratio=2.0, max_negatives_per_positive=3)
        for r in full:
            r.pop("ent_by_id", None)
            r.pop("gold_pairs", None)
            r.pop("ent_ids_list", None)
        ev_rows = rows_to_eval_input(full)
        meta = {
            "n_documents": len(st_docs) if st_docs else last_stream_stats.get("n_documents", 0),
            "n_examples": len(full),
            "n_positive_instances": sum(1 for r in full if r.get("label") != "__NEGATIVE__"),
        }
        subset_payload[subset_name] = (ev_rows, meta)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for base in bases:
        for seed in seeds:
            ck = _checkpoint(base, seed)
            if not ck.is_file():
                for subset_name, (_ev_rows, meta) in subset_payload.items():
                    out[(base, subset_name)].append(
                        {"seed": seed, "status": "missing_checkpoint", "checkpoint": str(ck), **meta}
                    )
                    continue
            try:
                model, tok, l2i, labels, lr = load_model_from_checkpoint(
                    ck, device, state_dict_strict=True
                )
            except Exception as e:
                for subset_name, (_ev_rows, meta) in subset_payload.items():
                    out[(base, subset_name)].append(
                        {
                            "seed": seed,
                            "status": "load_failed",
                            "error": str(e),
                        "checkpoint": str(ck),
                            **meta,
                        }
                    )
                continue
            for subset_name, (ev_rows, meta) in subset_payload.items():
                if not ev_rows:
                    out[(base, subset_name)].append({"seed": seed, "status": "no_rows", **meta})
                    continue
                ev = evaluate_rows_on_loaded_model(
                    model,
                    tok,
                    l2i,
                    labels,
                    lr,
                    ck,
                    ev_rows,
                    device=device,
                    max_length=384,
                    batch_size=batch,
                )
                ev["seed"] = seed
                ev.update(meta)
                out[(base, subset_name)].append(ev)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return out


def aggregate_primary(
    raw: dict[tuple[str, str, str], list[dict[str, Any]]],
    role_by_base: dict[str, str],
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for (base, src_id, ev_type), xs in sorted(raw.items()):
        ok = [x for x in xs if x.get("status") == "ok"]
        failed = [x for x in xs if x.get("status") != "ok"]
        row: dict[str, Any] = {
                    "base_experiment_id": base,
            "role": role_by_base.get(base, ""),
            "evidence_type": ev_type,
            "evaluation_source": src_id,
            "completed_seeds": len(ok),
            "seeds_expected": len(xs),
                                "mean_precision": "",
                                "std_precision": "",
                                "mean_recall": "",
                                "std_recall": "",
                                "mean_macro_f1": "",
                                "std_macro_f1": "",
            "support": "",
            "notes": "",
        }
        if not ok:
            sts = ";".join({str(x.get("status")) for x in failed}) or "no_ok_seeds"
            if src_id == "drugprot_official_test_pairs" and sts == "no_rows":
                row["notes"] = (
                    "no_rows:blocked_no_test_split_in_processed_t1_drugprot_jsonl"
                    ";see_manifests/drugprot_unresolved_status.json"
                )
            else:
                row["notes"] = sts
            rows_out.append(row)
                        continue

        def pull(k: str) -> list[float]:
            return [float(x[k]) for x in ok if k in x]

        mp, mr, mf = pull("macro_precision"), pull("macro_recall"), pull("macro_f1")
        row["mean_precision"] = round(statistics.mean(mp), 4)
        row["std_precision"] = round(statistics.stdev(mp) if len(mp) > 1 else 0.0, 4)
        row["mean_recall"] = round(statistics.mean(mr), 4)
        row["std_recall"] = round(statistics.stdev(mr) if len(mr) > 1 else 0.0, 4)
        row["mean_macro_f1"] = round(statistics.mean(mf), 4)
        row["std_macro_f1"] = round(statistics.stdev(mf) if len(mf) > 1 else 0.0, 4)
        row["support"] = int(ok[0].get("support", 0))
        if failed:
            row["notes"] = "some_seeds_failed:" + ";".join({str(x.get("status")) for x in failed})
        rows_out.append(row)
    return rows_out


def aggregate_subset(raw: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (base, subset), xs in sorted(raw.items()):
        ok = [x for x in xs if x.get("status") == "ok"]
        row: dict[str, Any] = {
                            "base_experiment_id": base,
            "evidence_type": "pairing_subset",
            "subset_type": subset,
            "n_documents": xs[0].get("n_documents", "") if xs else "",
            "n_examples": xs[0].get("n_examples", "") if xs else "",
            "n_positive_instances": xs[0].get("n_positive_instances", "") if xs else "",
            "mean_precision": "",
                            "std_precision": "",
            "mean_recall": "",
                            "std_recall": "",
            "mean_macro_f1": "",
                            "std_macro_f1": "",
        }
        if not ok:
            out.append(row)
            continue
        mf = [float(x["macro_f1"]) for x in ok]
        mp = [float(x["macro_precision"]) for x in ok]
        mr = [float(x["macro_recall"]) for x in ok]
        row["n_documents"] = ok[0].get("n_documents", "")
        row["n_examples"] = ok[0].get("n_examples", "")
        row["n_positive_instances"] = ok[0].get("n_positive_instances", "")
        row["mean_macro_f1"] = round(statistics.mean(mf), 4)
        row["std_macro_f1"] = round(statistics.stdev(mf) if len(mf) > 1 else 0.0, 4)
        row["mean_precision"] = round(statistics.mean(mp), 4)
        row["std_precision"] = round(statistics.stdev(mp) if len(mp) > 1 else 0.0, 4)
        row["mean_recall"] = round(statistics.mean(mr), 4)
        row["std_recall"] = round(statistics.stdev(mr) if len(mr) > 1 else 0.0, 4)
        out.append(row)
    return out


def realism_rows_to_oncology(primary_realism_agg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in primary_realism_agg:
        out.append(
            {
                "base_experiment_id": r["base_experiment_id"],
                "evidence_type": "realism_probe",
                "subset_type": f"realism_probe:{r['evaluation_source']}",
                "n_documents": "",
                "n_examples": r.get("support", ""),
                "n_positive_instances": "",
                "mean_precision": r.get("mean_precision", ""),
                "std_precision": r.get("std_precision", ""),
                "mean_recall": r.get("mean_recall", ""),
                "std_recall": r.get("std_recall", ""),
                "mean_macro_f1": r.get("mean_macro_f1", ""),
                "std_macro_f1": r.get("std_macro_f1", ""),
            }
        )
    return out


def build_reliability(
    raw: dict[tuple[str, str, str], list[dict[str, Any]]],
    role_by_base: dict[str, str],
    bases: list[str],
) -> list[dict[str, Any]]:
    by_f1: dict[str, list[float]] = defaultdict(list)
    by_hcp: dict[str, list[float]] = defaultdict(list)
    for (base, src_id, _), xs in raw.items():
        if src_id != "biored_official_test_pairs":
            continue
        for x in xs:
            if x.get("status") == "ok":
                by_f1[base].append(float(x["macro_f1"]))
                if x.get("high_conf_precision") is not None:
                    by_hcp[base].append(float(x["high_conf_precision"]))

    out = []
    for base in bases:
        fs = by_f1.get(base, [])
        if len(fs) < 2:
            std = 0.0
            cv = 0.0
            mean_f = fs[0] if fs else 0.0
        else:
            mean_f = statistics.mean(fs)
            std = statistics.stdev(fs)
            cv = std / mean_f if mean_f > 1e-8 else 0.0
        role = role_by_base.get(base, "optional")
        if role == "diagnostic":
            trust = "diagnostic_only"
        elif role == "control":
            trust = "caution"
        elif std < 0.04:
            trust = "high"
        elif std < 0.09:
            trust = "moderate"
        else:
            trust = "caution"
        hcp_vals = by_hcp.get(base, [])
        out.append(
            {
                "base_experiment_id": base,
                "role": role,
                "seed_std_macro_f1_biored_test": round(std, 4),
                "coefficient_of_variation_macro_f1": round(cv, 4),
                "mean_high_conf_precision_biored": round(statistics.mean(hcp_vals), 4)
                if hcp_vals
                else "",
                "anomaly_flags": "high_seed_std" if std > 0.1 else "",
                "trust_status": trust,
            }
        )
    return out


def build_error_taxonomy(
    raw: dict[tuple[str, str, str], list[dict[str, Any]]], bases: list[str]
) -> list[dict[str, Any]]:
    by_base: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (base, _src, ev), xs in raw.items():
        if ev != "split_external":
                continue
        for x in xs:
            if x.get("status") == "ok" and x.get("error_taxonomy"):
                by_base[base].append(x["error_taxonomy"])
    out = []
    for base in bases:
        xs = by_base.get(base, [])
        if not xs:
            out.append({"base_experiment_id": base, "notes": "no_split_external_ok_runs"})
                continue
        agg: dict[str, float] = defaultdict(float)
        for d in xs:
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    agg[k] += v
        row: dict[str, Any] = {"base_experiment_id": base, "n_slices_aggregated": len(xs)}
        for k, v in agg.items():
            row[f"sum_{k}"] = round(v, 4)
        out.append(row)
    return out


def rank_robustness(
    raw: dict[tuple[str, str, str], list[dict[str, Any]]], bases: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seeds = _seeds()
    seed_ranks: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for seed in seeds:
        scores: list[tuple[str, float]] = []
        for b in bases:
            for (base, src_id, _), xs in raw.items():
                if base != b or src_id != "biored_official_test_pairs":
                continue
                for x in xs:
                    if x.get("seed") == seed and x.get("status") == "ok":
                        scores.append((b, float(x["macro_f1"])))
        scores.sort(key=lambda z: -z[1])
        for rank, (bid, _) in enumerate(scores, start=1):
            seed_ranks[seed].append((bid, rank))
    mean_ranks: dict[str, list[int]] = defaultdict(list)
    for _s, lst in seed_ranks.items():
        for bid, rk in lst:
            mean_ranks[bid].append(rk)
    summary_rows = []
    for b in bases:
        rs = mean_ranks.get(b, [])
        summary_rows.append(
            {
                "row_kind": "model_summary",
                "base_experiment_id": b,
                "mean_rank_biored_test": round(statistics.mean(rs), 3) if rs else "",
                "rank_std": round(statistics.stdev(rs), 3) if len(rs) > 1 else 0.0,
                "seeds_used": len(rs),
                "model_a": "",
                "model_b": "",
                "seed_wins_a_over_b": "",
            }
        )
    wins: dict[tuple[str, str], int] = defaultdict(int)
    for seed in seeds:
        by_b: dict[str, float] = {}
        for (base, src_id, _), xs in raw.items():
            if src_id != "biored_official_test_pairs":
                continue
            for x in xs:
                if x.get("seed") == seed and x.get("status") == "ok":
                    by_b[base] = float(x["macro_f1"])
        present = [b for b in bases if b in by_b]
        for i, a in enumerate(present):
            for b in present[i + 1 :]:
                sa, sb = by_b[a], by_b[b]
                if sa > sb:
                    wins[(a, b)] += 1
                elif sb > sa:
                    wins[(b, a)] += 1
    win_rows = []
    for (a, b), v in sorted(wins.items()):
        win_rows.append(
            {
                "row_kind": "pairwise_win_count",
                "base_experiment_id": "",
                "mean_rank_biored_test": "",
                "rank_std": "",
                "seeds_used": "",
                "model_a": a,
                "model_b": b,
                "seed_wins_a_over_b": v,
            }
        )
    return summary_rows, win_rows


def manual_audit_table(
    bases: list[str],
    proc: Path,
    audit_dir: Path,
    role_by_base: dict[str, str],
) -> list[dict[str, Any]]:
    from inference.predict_checkpoint import load_model_from_checkpoint, predict_labels

    audit_dir.mkdir(parents=True, exist_ok=True)
    path = proc / "t1_biored.jsonl"
    rows_full, _ = stream_pair_rows(path, max_pairs=100, source_split_in={"test"})
    rows_full = rows_full[:70]
    rows_full = add_eval_negatives(
        rows_full, random.Random(99), negative_ratio=2.0, max_negatives_per_positive=2
    )
    for r in rows_full:
        r.pop("ent_by_id", None)
        r.pop("gold_pairs", None)
        r.pop("ent_ids_list", None)
    cases_path = audit_dir / "manual_audit_cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as fh:
        for r in rows_full:
            fh.write(
                json.dumps(
                    {k: r.get(k) for k in ("sample_id", "text", "label", "head_entity_label", "tail_entity_label")},
                    ensure_ascii=False,
                )
                + "\n"
            )
    ev_in = rows_to_eval_input(rows_full)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out: list[dict[str, Any]] = []
    audit_seed = 1
    for base in bases:
        ck = _checkpoint(base, audit_seed)
        row: dict[str, Any] = {
            "base_experiment_id": base,
            "role": role_by_base.get(base, ""),
            "audit_seed": audit_seed,
            "n_audit_cases": len(ev_in),
            "clearly_correct": "",
            "partially_correct": "",
            "misleading": "",
            "clearly_wrong": "",
            "coding_method": "",
            "comment": "",
        }
        if not ck.is_file():
            row["coding_method"] = "blocked"
            row["comment"] = "missing_checkpoint"
            out.append(row)
            continue
        try:
            model, tok, l2i, _labels, _lr = load_model_from_checkpoint(
                ck, device, state_dict_strict=True
            )
        except Exception as e:
            row["coding_method"] = "blocked"
            row["comment"] = str(e)[:200]
            out.append(row)
            continue
        yt = [r["label"] for r in ev_in]
        preds, confs = predict_labels(
            model, tok, l2i, ev_in, device=device, max_length=384, batch_size=8
        )
        del model
        cc = mc = mis = cw = 0
        for t, p, c in zip(yt, preds, confs):
            if t == p:
                cc += 1
            else:
                cw += 1
                if c >= 0.85:
                    mis += 1
        row["clearly_correct"] = cc
        row["partially_correct"] = ""
        row["misleading"] = mis
        row["clearly_wrong"] = cw
        row["coding_method"] = "gold_match_proxy_highconf_misleading"
        row["comment"] = (
            "partially_correct not defined; human review required. misleading = wrong with max prob >= 0.85."
        )
        out.append(row)
    return out


def write_schema_stress_table(tables: Path) -> None:
    rows = [
        {
            "stress_dimension": "assertion_subtype_collapse",
            "evidence_type": "split_external",
            "observation": "Gold mapped_label collapses nuanced clinical assertions into coarse S2 buckets.",
            "implication_for_S2_current": "Benchmark F1 does not prove fine-grained assertion subtype correctness.",
            "S2_refined_hybrid_pressure": "moderate_if_subtype_reporting_required",
        },
        {
            "stress_dimension": "predictive_diagnostic_prognostic_ambiguity",
            "evidence_type": "ontology_external",
            "observation": "Without assertion-type gold on external pairs, predictive vs diagnostic separation is not measured.",
            "implication_for_S2_current": "Do not infer clinical interpretation class from relation F1 alone.",
            "S2_refined_hybrid_pressure": "high_if_clinical_interpretation_is_downstream",
        },
        {
            "stress_dimension": "population_outcome_entities",
            "evidence_type": "realism_probe",
            "observation": "S2_current lacks dedicated population/outcome heads; rare in pair exports.",
            "implication_for_S2_current": "External errors on cohort/outcome mentions may be invisible in pair metrics.",
            "S2_refined_hybrid_pressure": "moderate",
        },
        {
            "stress_dimension": "coarse_clinical_bucket",
            "evidence_type": "split_external",
            "observation": "Many relations map to ASSOCIATION_GENERAL under S2 — limits oncology-specific story from scores.",
            "implication_for_S2_current": "Schema adequate for current operational scope; refinement optional.",
            "S2_refined_hybrid_pressure": "low_to_moderate",
        },
    ]
    _write_csv(
        tables / "schema_stress_test_table.csv",
        rows,
        [
            "stress_dimension",
            "evidence_type",
            "observation",
            "implication_for_S2_current",
            "S2_refined_hybrid_pressure",
        ],
    )


def render_report(layout: dict[str, Path], primary_path: Path) -> None:
    """Eight-section report; numeric highlights read from primary CSV when present."""
    tops = ""
    if primary_path.is_file():
        with primary_path.open(encoding="utf-8", newline="") as fh:
            rdr = csv.DictReader(fh)
            biored = [r for r in rdr if r.get("evaluation_source") == "biored_official_test_pairs"]
        if biored:
            biored.sort(key=lambda r: float(r.get("mean_macro_f1") or -1), reverse=True)
            tops = "; ".join(
                f"{r['base_experiment_id']} F1={r.get('mean_macro_f1','')}" for r in biored[:3]
            )

    lines = [
        "# External evaluation report",
        "",
        "## 1. Executive summary",
        "",
        "- **Models evaluated:** see `manifests/evaluation_model_registry.csv` (shortlist + optional S001/M009).",
        f"- **Strongest split_external signal (BioRED test, mean macro-F1, top rows):** {tops or 'see `reports/tables/primary_external_results.csv`.'}",
        "- **Internal winners vs external:** Compare BioRED/BC5CDR/(DrugProt if populated) in `primary_external_results.csv` — do not average sources.",
        "- **Best model set (pending domain review):** prioritize high macro-F1 with low seed std on BioRED (see `reliability_stability_table.csv` + `rank_robustness_table.csv`).",
        "- **S2_current adequacy:** benchmark scores test the operational label space; see `schema_stress_test_table.csv` for documented limitations — no mandatory redesign from F1 alone.",
        "",
        "## 2. Evaluation protocol",
        "",
        "- **Evidence taxonomy:** `evaluation_source_registry.csv` — `split_external` benchmark rows are Layer A; `realism_probe` in `oncology_subset_results.csv`; `ontology_external` / `weak_probe` registered but **not** scored as gold benchmarks.",
        "- **Metric hierarchy:** Tier-1 = macro P/R/F1 + support + seed mean/std in primary table; Tier-2 micro F1 in runner output fields where present; Tier-3 error taxonomy CSV.",
        "- **Strict realism:** `data/processed/strict_realism_protocol.json` — pair-level strict on mapped labels; not span-level.",
        "- **Manual audit:** `audit/manual_audit_cases.jsonl` + `manual_audit_table.csv` — **audit-style** automatic proxy; `partially_correct` not defined without human labels.",
        "- **Selection fairness:** `selection_fairness_note.json` — controls M005/M026 included by design.",
        "",
        "## 3. Primary external benchmark results",
        "",
        "Layer A only in `reports/tables/primary_external_results.csv` (`split_external`). Interpret per source; no synthetic fusion score.",
        "",
        "## 4. Oncology realism subset results",
        "",
        "`oncology_subset_results.csv` — `pairing_subset` rows (support columns) plus `realism_probe` rows (explicit evidence_type).",
        "",
        "## 5. Reliability, stability, and error analysis",
        "",
        "- `reliability_stability_table.csv` — seed std / CV on BioRED test; trust bands; high-confidence precision when defined.",
        "- `rank_robustness_table.csv` — mean rank + pairwise seed win counts.",
        "- `error_taxonomy_table.csv` — aggregated Layer-D style buckets over split_external passes.",
        "",
        "## 6. Schema stress-test findings",
        "",
        "See `schema_stress_test_table.csv` — evaluation-layer documentation tying external behavior to `S2_current` limits and optional `S2_refined_hybrid` pressure (not a redesign decision).",
        "",
        "## 7. Manual audit findings",
        "",
        "Automatic proxy counts only — **human coding required** before claiming usability of oncology assertions in production contexts.",
        "",
        "## 8. Final recommendation",
        "",
        "- **Primary candidates:** models with strong BioRED/BC5CDR macro-F1 and stable seeds; **DrugProt** rows require packaged test split (see `manifests/drugprot_unresolved_status.json` if empty).",
        "- **Secondary / diagnostic:** M005/M026 remain controls — not promoted on benchmark scores alone.",
        "- **Training rerun:** not implied solely by this pass; consider if external gaps persist after human audit.",
        "- **Schema/trainer:** optional refinement only if subtype or span-strict claims become requirements.",
        "",
        "---",
        "",
        f"*Generated {_utc()} by `external_evaluation.run_external_evaluation`.*",
        "",
    ]
    (layout["reports"] / "external_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def mirror_outputs(layout: dict[str, Path]) -> None:
    if os.environ.get("EXT_EVAL_SKIP_MIRROR", "").lower() in ("1", "true", "yes"):
        return
    mir = mirror_reports_dir()
    rep = layout["reports"] / "external_evaluation_report.md"
    if rep.is_file():
        shutil.copy2(rep, mir / "external_evaluation_report.md")
    for fn in os.listdir(layout["tables"]):
        shutil.copy2(layout["tables"] / fn, mir / "tables" / fn)
    ad = layout["audit"]
    if ad.is_dir():
        (mir / "audit").mkdir(parents=True, exist_ok=True)
        for fn in os.listdir(ad):
            shutil.copy2(ad / fn, mir / "audit" / fn)


def run_eval_pipeline(layout: dict[str, Path], *, skip_reset: bool) -> int:
    proc = training_processed()
    if not proc.is_dir():
        print(f"Missing processed dir: {proc}", file=sys.stderr)
        return 1

    manifests = layout["manifests"]
    tables = layout["tables"]
    write_drugprot_diagnosis_artifacts(proc, manifests, tables)
    write_model_registry(manifests)
    write_source_registry(layout["data_processed"])
    write_protocol_files(layout["data_processed"])

    role_by_base = {r["base_experiment_id"]: r["role"] for r in MODEL_REGISTRY_ROWS}
    bases = [r["base_experiment_id"] for r in MODEL_REGISTRY_ROWS]

    if not skip_reset and os.environ.get("EXT_EVAL_SKIP_RESET", "") != "1":
        prior_partial_output_reset(layout)
    refresh_storage_audit(layout, {"phase": "pre_eval_refresh"})

    max_pos = _max_pos_per_source()
    prebuilt: dict[str, tuple[list[dict[str, Any]], dict[str, int]]] = {}
    meta_by_src: dict[str, dict[str, int]] = {}
    all_sources = BENCHMARK_SOURCES + REALISM_SOURCES
    for spec in all_sources:
        sid = spec["evaluation_source_id"]
        neg_rng = random.Random(42 + abs(hash(sid)) % 1_000_000)
        ev, st = build_eval_set_for_source(proc, spec, max_pos=max_pos, neg_rng=neg_rng)
        prebuilt[sid] = (ev, st)
        meta_by_src[sid] = st

    raw_all = run_benchmark_matrix(
        bases, role_by_base, all_sources, proc, prebuilt
    )

    raw_split = {k: v for k, v in raw_all.items() if k[2] == "split_external"}
    raw_real = {k: v for k, v in raw_all.items() if k[2] == "realism_probe"}

    primary_rows = aggregate_primary(raw_split, role_by_base)
    _write_csv(
        tables / "primary_external_results.csv",
        primary_rows,
        [
            "base_experiment_id",
            "role",
            "evidence_type",
            "evaluation_source",
            "completed_seeds",
            "seeds_expected",
            "mean_precision",
            "std_precision",
            "mean_recall",
            "std_recall",
            "mean_macro_f1",
            "std_macro_f1",
            "support",
            "notes",
        ],
    )

    realism_agg = aggregate_primary(raw_real, role_by_base)
    sub_raw = run_subset_matrix(bases, role_by_base, proc)
    sub_agg = aggregate_subset(sub_raw)
    oncology = sub_agg + realism_rows_to_oncology(realism_agg)
    _write_csv(
        tables / "oncology_subset_results.csv",
        oncology,
        [
            "base_experiment_id",
            "evidence_type",
            "subset_type",
            "n_documents",
            "n_examples",
            "n_positive_instances",
            "mean_precision",
            "std_precision",
            "mean_recall",
            "std_recall",
            "mean_macro_f1",
            "std_macro_f1",
        ],
    )

    _write_csv(
        tables / "reliability_stability_table.csv",
        build_reliability(raw_split, role_by_base, bases),
        [
            "base_experiment_id",
            "role",
            "seed_std_macro_f1_biored_test",
            "coefficient_of_variation_macro_f1",
            "mean_high_conf_precision_biored",
            "anomaly_flags",
            "trust_status",
        ],
    )

    err = build_error_taxonomy(raw_split, bases)
    if err:
    _write_csv(
            tables / "error_taxonomy_table.csv",
            err,
            sorted({k for row in err for k in row}),
        )

    rr, ww = rank_robustness(raw_split, bases)
    rank_fields = [
        "row_kind",
        "base_experiment_id",
        "mean_rank_biored_test",
        "rank_std",
        "seeds_used",
        "model_a",
        "model_b",
        "seed_wins_a_over_b",
    ]
    _write_csv(tables / "rank_robustness_table.csv", rr + ww, rank_fields)

    _write_csv(
        tables / "manual_audit_table.csv",
        manual_audit_table(bases, proc, layout["audit"], role_by_base),
        [
            "base_experiment_id",
            "role",
            "audit_seed",
            "n_audit_cases",
            "clearly_correct",
            "partially_correct",
            "misleading",
            "clearly_wrong",
            "coding_method",
            "comment",
        ],
    )

    write_schema_stress_table(tables)
    render_report(layout, tables / "primary_external_results.csv")
    mirror_outputs(layout)

    print(f"Wrote results under {layout['root']}")
    return 0


def cmd_prepare_reset() -> int:
    layout = ensure_manifest_dirs()
    proc = training_processed()
    if proc.is_dir():
        write_drugprot_diagnosis_artifacts(proc, layout["manifests"], layout["tables"])
    (layout["root"] / "logs").mkdir(parents=True, exist_ok=True)
    prior_partial_output_reset(layout)
    refresh_storage_audit(layout, {"phase": "prepare_reset"})
    write_model_registry(layout["manifests"])
    write_source_registry(layout["data_processed"])
    write_protocol_files(layout["data_processed"])
    write_schema_stress_table(layout["tables"])
    print(json.dumps({"ok": True, "root": str(layout["root"])}))
    return 0


def sbatch_path() -> Path:
    return code_root() / "external_evaluation" / "sbatch" / "external_eval_full_gpu.sbatch"


def sbatch_path_smoke() -> Path:
    return code_root() / "external_evaluation" / "sbatch" / "external_eval_smoke_gpu.sbatch"


def _submit_gpu_job_and_monitor(
    layout: dict[str, Path],
    sp: Path,
    *,
    job_kind: str,
    jobs_csv: Path,
    summary_json: Path,
    snapshot_csv: Path,
    failures_csv: Path,
    summary_note: str,
) -> int:
    """``sbatch`` + ~2 minute polling + sacct probe. Login node: submission only."""
    if not sp.is_file():
        print(f"Missing sbatch script: {sp}", file=sys.stderr)
        return 1
    out = subprocess.run(["sbatch", str(sp)], capture_output=True, text=True, check=False)
    job_id = ""
    if out.returncode == 0:
        parts = out.stdout.strip().split()
        if parts:
            job_id = parts[-1]
    row = {
        "submitted_at_utc": _utc(),
        "job_kind": job_kind,
        "sbatch_script": str(sp),
        "job_id": job_id,
        "stdout": out.stdout.strip(),
        "stderr": out.stderr.strip(),
        "exit_code": out.returncode,
    }
    append = jobs_csv.is_file()
    with jobs_csv.open("a" if append else "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not append:
            w.writeheader()
        w.writerow(row)
    prev_jobs: list[dict[str, str]] = []
    if summary_json.is_file():
        try:
            prev = json.loads(summary_json.read_text(encoding="utf-8"))
            prev_jobs = prev.get("jobs", [])
        except json.JSONDecodeError:
            pass
    prev_jobs.append(row)
    summary_json.write_text(
        json.dumps({"jobs": prev_jobs, "note": summary_note}, indent=2) + "\n",
        encoding="utf-8",
    )
    if not job_id:
        return 1

    snapshots = []
    t0 = time.time()
    while time.time() - t0 < 120:
        sq = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%i,%T,%M"],
            capture_output=True,
            text=True,
            check=False,
        )
        snapshots.append(
            {
                "elapsed_s": int(time.time() - t0),
                "job_id": job_id,
                "job_kind": job_kind,
                "squeue_stdout": sq.stdout.strip(),
                "squeue_exit": sq.returncode,
            }
        )
        time.sleep(40)

    with snapshot_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["elapsed_s", "job_id", "job_kind", "squeue_stdout", "squeue_exit"]
        )
        w.writeheader()
        for s in snapshots:
            w.writerow(s)

    sa = subprocess.run(
        [
            "sacct",
            "-j",
            job_id,
            "-n",
            "-o",
            "JobID,State,ExitCode,Elapsed",
            "-X",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    fail_lines = [
        ln
        for ln in sa.stdout.splitlines()
        if any(x in ln for x in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"))
    ]
    if fail_lines:
        with failures_csv.open("w", encoding="utf-8", newline="") as fh:
            fh.write("job_kind,job_id,sacct_line\n")
            for ln in fail_lines:
                fh.write(f"{job_kind},{job_id},{ln.strip().replace(chr(10), ' ')}\n")
    else:
        with failures_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["job_kind", "job_id", "note"])
            w.writerow([job_kind, job_id, "no_early_failure_lines_detected_in_sacct"])

    print(json.dumps({"job_kind": job_kind, "job_id": job_id, "snapshots": len(snapshots)}))
    return 0


def _prepare_submit_layout(layout: dict[str, Path]) -> None:
    (layout["root"] / "logs").mkdir(parents=True, exist_ok=True)
    write_model_registry(layout["manifests"])
    write_source_registry(layout["data_processed"])
    write_protocol_files(layout["data_processed"])


def cmd_submit_sbatch() -> int:
    """Submit full GPU job (login node: sbatch only). Run ``--prepare-reset`` before first full run."""
    layout = ensure_manifest_dirs()
    _prepare_submit_layout(layout)
    return _submit_gpu_job_and_monitor(
        layout,
        sbatch_path(),
        job_kind="full",
        jobs_csv=layout["manifests"] / "submitted_external_eval_jobs.csv",
        summary_json=layout["manifests"] / "external_eval_submission_summary.json",
        snapshot_csv=layout["manifests"] / "external_eval_monitoring_snapshot.csv",
        failures_csv=layout["manifests"] / "external_eval_early_failures.csv",
        summary_note="External evaluation GPU jobs (full + historical rows in jobs CSV).",
    )


def cmd_submit_sbatch_smoke() -> int:
    """Submit small GPU smoke job; outputs under ``<canonical>/smoke_runs/<jobid>/``."""
    layout = ensure_manifest_dirs()
    _prepare_submit_layout(layout)
    return _submit_gpu_job_and_monitor(
        layout,
        sbatch_path_smoke(),
        job_kind="smoke",
        jobs_csv=layout["manifests"] / "submitted_external_eval_smoke_jobs.csv",
        summary_json=layout["manifests"] / "external_eval_smoke_submission_summary.json",
        snapshot_csv=layout["manifests"] / "external_eval_smoke_monitoring_snapshot.csv",
        failures_csv=layout["manifests"] / "external_eval_smoke_early_failures.csv",
        summary_note="GPU smoke tests (isolated EXTERNAL_EVAL_ROOT per job).",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prepare-reset", action="store_true", help="Archive old eval outputs + write registries/protocol.")
    p.add_argument(
        "--prepare-reset-and-submit",
        action="store_true",
        help="Run --prepare-reset then --submit-sbatch (recommended once before first GPU run).",
    )
    p.add_argument("--run-eval", action="store_true", help="Run full GPU/CPU evaluation (heavy).")
    p.add_argument("--submit-sbatch", action="store_true", help="Queue full GPU job + manifests + ~2min monitor.")
    p.add_argument(
        "--submit-sbatch-smoke",
        action="store_true",
        help="Queue GPU smoke job (small caps; outputs in smoke_runs/<jobid>/) + manifests + monitor.",
    )
    args = p.parse_args()

    if args.prepare_reset_and_submit:
        r = cmd_prepare_reset()
        return cmd_submit_sbatch() if r == 0 else r
    if args.submit_sbatch_smoke:
        return cmd_submit_sbatch_smoke()
    if args.submit_sbatch:
        return cmd_submit_sbatch()
    if args.prepare_reset:
        return cmd_prepare_reset()
    if args.run_eval:
        layout = ensure_manifest_dirs()
        return run_eval_pipeline(
            layout, skip_reset=os.environ.get("EXT_EVAL_SKIP_RESET") == "1"
        )

    p.print_help()
    print(
        "\nTypical (avoid login-node eval): --prepare-reset (metadata only); "
        "--submit-sbatch-smoke then --submit-sbatch. "
        "Use --run-eval only on a GPU node.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
