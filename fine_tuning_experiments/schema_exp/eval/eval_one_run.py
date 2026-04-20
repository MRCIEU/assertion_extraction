#!/usr/bin/env python3.11
"""
Phase A-eval — evaluate a single PA_* run.

Loads the run's `best.pt` checkpoint (end-of-T2 best by dev macro-F1) and runs
three inference passes:

  1. BioRED test  — per-head F1 within the run's schema
  2. BC5CDR test  — DRUG_DISEASE F1 (control metric)
  3. KB surface   — softmax over 165 CIViC targets → mean(1 - P(__NEGATIVE__))
                    and KB_surface_matched (P(expected_label) > P(__NEGATIVE__))

Output: <run_dir>/eval/phase_a_eval.json

Invocation:
  python3.11 -m fine_tuning_experiments.schema_exp.eval.eval_one_run \
      --run-dir <path_to_PA_XX_XX_sNN>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

SCRIPT_DIR = Path(__file__).resolve().parent
# schema_exp/eval -> schema_exp -> fine_tuning_experiments -> project_1
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use the KB audit inference helpers (checkpoint loader).
_KG_INF = PROJECT_ROOT / "knowledge_grounded_evidence_audit"
if str(_KG_INF) not in sys.path:
    sys.path.insert(0, str(_KG_INF))
from inference.predict_checkpoint import load_model_from_checkpoint  # noqa: E402

EVAL_INPUTS = SCRIPT_DIR / "inputs"


# ───────────────────────────────────────────────────────────────────
# Inference helpers
# ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def softmax_probs(
    model: Any, tokenizer: Any, texts: list[str], *,
    device: torch.device, max_length: int, batch_size: int,
) -> list[list[float]]:
    """Return full softmax probability vector per text (row-major)."""
    out: list[list[float]] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        out.extend(probs)
    return out


def probs_to_pred(probs: list[float], id2label: dict[int, str]) -> tuple[str, float]:
    max_i = max(range(len(probs)), key=lambda i: probs[i])
    return id2label[max_i], float(probs[max_i])


# ───────────────────────────────────────────────────────────────────
# Metric helpers
# ───────────────────────────────────────────────────────────────────

def per_head_metrics(
    gold: list[str], pred: list[str], labels_ordered: list[str],
) -> dict[str, Any]:
    p, r, f, s = precision_recall_fscore_support(
        gold, pred, labels=labels_ordered, average=None, zero_division=0,
    )
    by_label = {
        lab: {"precision": float(p[i]), "recall": float(r[i]),
              "f1": float(f[i]), "support": int(s[i])}
        for i, lab in enumerate(labels_ordered)
    }
    macro_f1 = float(f1_score(gold, pred, labels=labels_ordered,
                              average="macro", zero_division=0))
    # Micro F1 excluding NEGATIVE for class-balanced view
    non_neg = [lab for lab in labels_ordered if lab != "__NEGATIVE__"]
    macro_f1_no_neg = float(f1_score(
        gold, pred, labels=non_neg, average="macro", zero_division=0,
    )) if non_neg else 0.0
    return {"per_label": by_label, "macro_f1": macro_f1,
            "macro_f1_excluding_negative": macro_f1_no_neg, "n": len(gold)}


# ───────────────────────────────────────────────────────────────────
# Evaluation passes
# ───────────────────────────────────────────────────────────────────

def eval_benchmark_pass(
    model: Any, tokenizer: Any, label2id: dict[str, int], labels_ordered: list[str],
    pair_file: Path, *, device: torch.device, max_length: int, batch_size: int,
) -> dict[str, Any]:
    rows = [json.loads(l) for l in pair_file.read_text().splitlines() if l.strip()]
    if not rows:
        return {"status": "empty", "file": str(pair_file)}
    texts = [r["text"] for r in rows]
    gold = [r["label"] for r in rows]
    id2label = {v: k for k, v in label2id.items()}
    probs = softmax_probs(model, tokenizer, texts,
                          device=device, max_length=max_length, batch_size=batch_size)
    preds: list[str] = []
    confs: list[float] = []
    # Remap gold labels that do not exist in this schema to __NEGATIVE__
    # (so that e.g. DGR_INHIBIT gold on an Sflat model is not counted as a new label)
    schema_labels = set(label2id)
    gold_mapped = [g if g in schema_labels else "__NEGATIVE__" for g in gold]
    for pv in probs:
        pr, cf = probs_to_pred(pv, id2label)
        preds.append(pr); confs.append(cf)
    result = per_head_metrics(gold_mapped, preds, labels_ordered)
    # Per-source-dataset breakdown
    by_source: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"gold": [], "pred": []})
    for r, gm, pr in zip(rows, gold_mapped, preds):
        by_source[r.get("source_dataset", "unknown")]["gold"].append(gm)
        by_source[r.get("source_dataset", "unknown")]["pred"].append(pr)
    source_metrics = {}
    for src, gp in by_source.items():
        source_metrics[src] = per_head_metrics(gp["gold"], gp["pred"], labels_ordered)
    result["by_source"] = source_metrics
    result["gold_remapping_count"] = sum(1 for g, gm in zip(gold, gold_mapped) if g != gm)
    result["file"] = str(pair_file)
    return result


def eval_kb_surface(
    model: Any, tokenizer: Any, label2id: dict[str, int],
    kb_file: Path, *, device: torch.device, max_length: int, batch_size: int,
) -> dict[str, Any]:
    rows = [json.loads(l) for l in kb_file.read_text().splitlines() if l.strip()]
    if not rows:
        return {"status": "empty"}
    texts = [r["text"] for r in rows]
    id2label = {v: k for k, v in label2id.items()}
    probs = softmax_probs(model, tokenizer, texts,
                          device=device, max_length=max_length, batch_size=batch_size)
    neg_idx = label2id.get("__NEGATIVE__")
    assert neg_idx is not None, "Schema must include __NEGATIVE__"

    target_records: list[dict[str, Any]] = []
    for r, pv in zip(rows, probs):
        pr, cf = probs_to_pred(pv, id2label)
        p_neg = float(pv[neg_idx])
        surface = 1.0 - p_neg
        expected = r.get("expected_label") or ""
        expected_in_schema = expected in label2id
        if expected_in_schema:
            p_expected = float(pv[label2id[expected]])
            matched = p_expected > p_neg
        else:
            p_expected = None
            matched = False
        target_records.append({
            "target_id": r["target_id"],
            "pairing_family": r.get("pairing_family"),
            "expected_label": expected,
            "expected_in_schema": expected_in_schema,
            "pred_label": pr,
            "p_negative": p_neg,
            "p_expected": p_expected,
            "surface_score": surface,
            "matched": matched,
            "non_negative": pr != "__NEGATIVE__",
        })

    n = len(target_records)
    surf_mean = sum(t["surface_score"] for t in target_records) / n
    surf_50 = sum(1 for t in target_records if t["surface_score"] > 0.5) / n
    matched_rate = sum(1 for t in target_records if t["matched"]) / n
    non_neg_rate = sum(1 for t in target_records if t["non_negative"]) / n

    # Per-family breakdown
    fam_stats: dict[str, dict[str, float]] = defaultdict(lambda: {
        "n": 0, "surface_mean": 0.0, "matched_rate": 0.0, "non_neg_rate": 0.0,
    })
    for t in target_records:
        f = t["pairing_family"] or "unknown"
        fam_stats[f]["n"] += 1
        fam_stats[f]["surface_mean"] += t["surface_score"]
        fam_stats[f]["matched_rate"] += 1 if t["matched"] else 0
        fam_stats[f]["non_neg_rate"] += 1 if t["non_negative"] else 0
    for f, d in fam_stats.items():
        if d["n"]:
            d["surface_mean"] /= d["n"]
            d["matched_rate"] /= d["n"]
            d["non_neg_rate"] /= d["n"]

    return {
        "n_targets": n,
        "kb_surface_mean": surf_mean,
        "kb_surface_matched": matched_rate,
        "kb_surface_50": surf_50,
        "kb_nonneg_rate": non_neg_rate,
        "per_family": {k: dict(v) for k, v in fam_stats.items()},
        "targets": target_records,
    }


# ───────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────

def parse_run_id(run_dir: Path) -> dict[str, str]:
    name = run_dir.name
    import re
    m = re.match(r"PA_([A-Z]+)_([A-Za-z]+)_s(\d+)", name)
    if not m:
        raise ValueError(f"Run name does not match PA pattern: {name}")
    return {"encoder_key": m.group(1), "schema_key": m.group(2), "seed": int(m.group(3))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="Path to PA_*_* run directory")
    ap.add_argument("--checkpoint", default="best.pt",
                    help="Checkpoint name under <run_dir>/checkpoints/ (default: best.pt)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-run even if phase_a_eval.json already exists")
    args = ap.parse_args()

    run_dir: Path = args.run_dir.resolve()
    assert run_dir.is_dir(), f"Not a directory: {run_dir}"
    ids = parse_run_id(run_dir)

    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "phase_a_eval.json"
    if out_json.exists() and not args.overwrite:
        print(f"SKIP (exists): {out_json}")
        return

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    schema_id = manifest["schema_id"]
    max_length = manifest["scientific_trainer"]["max_length"]

    ckpt_path = run_dir / "checkpoints" / args.checkpoint
    assert ckpt_path.exists(), f"Missing checkpoint {ckpt_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{run_dir.name}] device={device} max_length={max_length} schema={schema_id}")

    t0 = time.time()
    model, tokenizer, label2id, labels_ordered, _ = load_model_from_checkpoint(
        ckpt_path, device, state_dict_strict=True,
    )
    t_load = time.time() - t0
    print(f"  loaded checkpoint in {t_load:.1f}s n_labels={len(label2id)}")

    result: dict[str, Any] = {
        "run_id": run_dir.name,
        "encoder_key": ids["encoder_key"],
        "schema_key": ids["schema_key"],
        "schema_id": schema_id,
        "seed": ids["seed"],
        "checkpoint": str(ckpt_path),
        "label2id": label2id,
        "labels_ordered": labels_ordered,
        "max_length": max_length,
        "device": str(device),
        "duration_seconds": {"load": round(t_load, 2)},
    }

    # ── BioRED test ───────────────────────────────────────────
    t0 = time.time()
    biored_file = EVAL_INPUTS / f"biored_test_pairs_{ids['schema_key']}.jsonl"
    result["biored_test"] = eval_benchmark_pass(
        model, tokenizer, label2id, labels_ordered, biored_file,
        device=device, max_length=max_length, batch_size=args.batch_size,
    )
    result["duration_seconds"]["biored"] = round(time.time() - t0, 2)
    print(f"  biored macro_f1={result['biored_test']['macro_f1']:.4f} "
          f"(ex-NEG {result['biored_test']['macro_f1_excluding_negative']:.4f})"
          f" in {result['duration_seconds']['biored']:.1f}s")

    # ── BC5CDR test ───────────────────────────────────────────
    t0 = time.time()
    bc_file = EVAL_INPUTS / "bc5cdr_test_pairs.jsonl"
    bc_res = eval_benchmark_pass(
        model, tokenizer, label2id, labels_ordered, bc_file,
        device=device, max_length=max_length, batch_size=args.batch_size,
    )
    # Extract DRUG_DISEASE-specific F1
    dd = bc_res["per_label"].get("DRUG_DISEASE", {})
    bc_res["drug_disease_f1"] = float(dd.get("f1", 0.0))
    bc_res["drug_disease_support"] = int(dd.get("support", 0))
    result["bc5cdr_test"] = bc_res
    result["duration_seconds"]["bc5cdr"] = round(time.time() - t0, 2)
    print(f"  bc5cdr DD_F1={bc_res['drug_disease_f1']:.4f} "
          f"macro_f1={bc_res['macro_f1']:.4f} "
          f"in {result['duration_seconds']['bc5cdr']:.1f}s")

    # ── KB surface ────────────────────────────────────────────
    t0 = time.time()
    kb_file = EVAL_INPUTS / "kb_surface_pairs.jsonl"
    result["kb_surface"] = eval_kb_surface(
        model, tokenizer, label2id, kb_file,
        device=device, max_length=max_length, batch_size=args.batch_size,
    )
    result["duration_seconds"]["kb"] = round(time.time() - t0, 2)
    print(f"  kb surface_mean={result['kb_surface']['kb_surface_mean']:.4f} "
          f"matched={result['kb_surface']['kb_surface_matched']:.4f} "
          f"non_neg={result['kb_surface']['kb_nonneg_rate']:.4f} "
          f"in {result['duration_seconds']['kb']:.1f}s")

    result["duration_seconds"]["total"] = round(sum(result["duration_seconds"].values()), 2)
    # Strip heavy target-level records to a side file
    targets = result["kb_surface"].pop("targets")
    (out_dir / "kb_surface_targets.jsonl").write_text(
        "\n".join(json.dumps(t) for t in targets) + "\n"
    )
    out_json.write_text(json.dumps(result, indent=2))
    print(f"  wrote {out_json}")


if __name__ == "__main__":
    main()
