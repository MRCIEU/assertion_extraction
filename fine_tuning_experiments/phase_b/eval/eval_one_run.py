#!/usr/bin/env python3.11
"""Phase-B eval entry point — evaluate a single PB_* run.

Thin wrapper around the Phase A eval engine (schema_exp.eval.eval_one_run):
reuses the benchmark / KB-surface passes, the model-loading helper, and
the eval inputs (Phase B is S_pair-only, so the same Spair eval shards
are used).  Only differences from Phase A:

    - run-name regex is PB_{ENC}_{UPD}_{SCHED}_s{NN}
    - output is <run_dir>/eval/phase_b_eval.json
    - eval JSON carries update_key + schedule_key (instead of schema_key)
      and schema_key is pinned to "Spair"
    - EVAL_VERSION is read from phase_b/eval/EVAL_VERSION.txt (which is a
      symlink to schema_exp/eval/EVAL_VERSION.txt — Phase B reuses the
      Phase A eval contract byte-for-byte, per §4.6 and §7.6).

This file was lost in the 2026-04-24 source-tree deletion incident and
re-created byte-equivalently from the conversation record on the same
day.  It is functionally identical to the destroyed original and depends
only on stable Phase A eval primitives.

Usage:
    python3.11 -m fine_tuning_experiments.phase_b.eval.eval_one_run \\
        --run-dir /path/to/PB_PB_FT_T2_s01
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch

from fine_tuning_experiments.schema_exp.eval.eval_one_run import (
    EVAL_INPUTS,
    eval_benchmark_pass,
    eval_kb_surface,
    load_model_from_checkpoint,
)

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_VERSION_FILE = SCRIPT_DIR / "EVAL_VERSION.txt"


def _load_eval_version() -> str:
    return EVAL_VERSION_FILE.read_text().strip()


def parse_run_id(run_dir: Path) -> dict[str, Any]:
    name = run_dir.name
    m = re.match(r"PB_([A-Z]+)_([A-Z]+)_([A-Za-z0-9]+)_s(\d+)", name)
    if not m:
        raise ValueError(f"Run name does not match PB pattern: {name}")
    return {
        "encoder_key": m.group(1),
        "update_key": m.group(2),
        "schedule_key": m.group(3),
        "seed": int(m.group(4)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="Path to PB_* run directory")
    ap.add_argument("--checkpoint", default="best.pt",
                    help="Checkpoint name under <run_dir>/checkpoints/ (default: best.pt)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional explicit output path (default: <run_dir>/eval/phase_b_eval.json)")
    args = ap.parse_args()

    run_dir: Path = args.run_dir.resolve()
    assert run_dir.is_dir(), f"Not a directory: {run_dir}"
    ids = parse_run_id(run_dir)

    if args.out is not None:
        out_json = args.out.resolve()
        out_dir = out_json.parent
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = run_dir / "eval"
        out_dir.mkdir(exist_ok=True)
        out_json = out_dir / "phase_b_eval.json"
    if out_json.exists() and not args.overwrite:
        print(f"SKIP (exists): {out_json}")
        return

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    schema_id = manifest["schema_id"]
    assert schema_id == "S_pair", (
        f"Phase B runs must use S_pair schema; got {schema_id!r} for {run_dir.name}"
    )
    max_length = manifest["scientific_trainer"]["max_length"]
    update_regime = manifest.get("update_regime", "full_finetune")
    schedule = manifest.get("schedule", "T1_to_T2")

    ckpt_path = run_dir / "checkpoints" / args.checkpoint
    assert ckpt_path.exists(), f"Missing checkpoint {ckpt_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{run_dir.name}] device={device} max_length={max_length} "
          f"schema={schema_id} update={update_regime} schedule={schedule}")

    t0 = time.time()
    model, tokenizer, label2id, labels_ordered, _ = load_model_from_checkpoint(
        ckpt_path, device, state_dict_strict=True,
    )
    t_load = time.time() - t0
    print(f"  loaded checkpoint in {t_load:.1f}s n_labels={len(label2id)}")

    result: dict[str, Any] = {
        "eval_version": _load_eval_version(),
        "run_id": run_dir.name,
        "phase": "phase_b",
        "encoder_key": ids["encoder_key"],
        "update_key": ids["update_key"],
        "schedule_key": ids["schedule_key"],
        "schema_key": "Spair",
        "schema_id": schema_id,
        "update_regime": update_regime,
        "schedule": schedule,
        "seed": ids["seed"],
        "checkpoint": str(ckpt_path),
        "label2id": label2id,
        "labels_ordered": labels_ordered,
        "max_length": max_length,
        "device": str(device),
        "duration_seconds": {"load": round(t_load, 2)},
    }

    # BioRED test (S_pair inputs)
    t0 = time.time()
    biored_file = EVAL_INPUTS / "biored_test_pairs_Spair.jsonl"
    result["biored_test"] = eval_benchmark_pass(
        model, tokenizer, label2id, labels_ordered, biored_file,
        device=device, max_length=max_length, batch_size=args.batch_size,
    )
    result["duration_seconds"]["biored"] = round(time.time() - t0, 2)
    print(f"  biored macro_f1={result['biored_test']['macro_f1']:.4f} "
          f"(ex-NEG {result['biored_test']['macro_f1_excluding_negative']:.4f})"
          f" in {result['duration_seconds']['biored']:.1f}s")

    # BC5CDR test
    t0 = time.time()
    bc_file = EVAL_INPUTS / "bc5cdr_test_pairs.jsonl"
    bc_res = eval_benchmark_pass(
        model, tokenizer, label2id, labels_ordered, bc_file,
        device=device, max_length=max_length, batch_size=args.batch_size,
    )
    dd = bc_res["per_label"].get("DRUG_DISEASE", {})
    bc_res["drug_disease_f1"] = float(dd.get("f1", 0.0))
    bc_res["drug_disease_support"] = int(dd.get("support", 0))
    result["bc5cdr_test"] = bc_res
    result["duration_seconds"]["bc5cdr"] = round(time.time() - t0, 2)
    print(f"  bc5cdr DD_F1={bc_res['drug_disease_f1']:.4f} "
          f"macro_f1={bc_res['macro_f1']:.4f} "
          f"in {result['duration_seconds']['bc5cdr']:.1f}s")

    # KB surface
    t0 = time.time()
    kb_file = EVAL_INPUTS / "kb_surface_pairs.jsonl"
    result["kb_surface"] = eval_kb_surface(
        model, tokenizer, label2id, kb_file,
        device=device, max_length=max_length, batch_size=args.batch_size,
    )
    result["duration_seconds"]["kb"] = round(time.time() - t0, 2)
    kb = result["kb_surface"]
    print(f"  kb surface_mean={kb['kb_surface_mean']:.4f} "
          f"hit_A_sv={kb['kb_hit_A_setvalued']:.4f} "
          f"pmass_B_sv={kb['kb_pmass_B_setvalued']:.4f} "
          f"auc_C_sv={kb['kb_auc_C_setvalued']:.4f} "
          f"in {result['duration_seconds']['kb']:.1f}s")

    result["duration_seconds"]["total"] = round(
        sum(v for k, v in result["duration_seconds"].items() if k != "total"),
        2,
    )

    # Split heavy arrays (softmax + logits) into a separate file, same
    # contract as Phase A (schema_exp/eval/eval_one_run.py).
    heavy_records = result["kb_surface"].pop("targets")
    (out_dir / "kb_surface_targets.jsonl").write_text(
        "\n".join(json.dumps(t) for t in heavy_records) + "\n"
    )
    out_json.write_text(json.dumps(result, indent=2))
    print(f"  wrote {out_json}")
    print(f"  wrote {out_dir / 'kb_surface_targets.jsonl'} "
          f"({len(heavy_records)} per-target records incl. full logits)")


if __name__ == "__main__":
    main()
