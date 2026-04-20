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

# Schema-aligned expected label mapping (§11.7.1 of paper design).
from fine_tuning_experiments.schema_exp.eval.schema_expected_label import (  # noqa: E402
    resolve_family,
    schema_expected_label_set,
)

EVAL_INPUTS = SCRIPT_DIR / "inputs"

# Abstention sweep grid for Method C (pre-specified; §11.7.1 of paper design).
_METHOD_C_TAU_GRID = [round(0.05 * i, 2) for i in range(21)]  # 0.00, 0.05, ..., 1.00
_PROJECTION_MODES = ("set_valued", "single_label")


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


@torch.no_grad()
def logits_and_softmax(
    model: Any, tokenizer: Any, texts: list[str], *,
    device: torch.device, max_length: int, batch_size: int,
) -> tuple[list[list[float]], list[list[float]]]:
    """Return (logits, softmax) per text — pre-softmax logits are saved for
    future-proof calibration analyses (§11.7.4)."""
    all_logits: list[list[float]] = []
    all_probs: list[list[float]] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        all_logits.extend(logits.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    return all_logits, all_probs


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


def _method_c_auc(
    rows: list[tuple[float, int]], tau_grid: list[float],
) -> float:
    """Trapezoidal AUC of precision_kept vs reject_rate over a pre-specified tau grid.

    `rows` = list of (p_negative, hit_0_or_1) per evaluable target.
    Returns NaN if the curve is degenerate.
    """
    n = len(rows)
    if n == 0:
        return float("nan")
    points: list[tuple[float, float]] = []  # (reject_rate, precision_kept)
    for tau in tau_grid:
        kept = [h for (p, h) in rows if p <= tau]
        rejected = n - len(kept)
        rej_rate = rejected / n
        if not kept:
            continue
        prec_kept = sum(kept) / len(kept)
        points.append((rej_rate, prec_kept))
    if len(points) < 2:
        return float("nan")
    points.sort(key=lambda p: p[0])
    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        if dx < 0:
            continue
        auc += 0.5 * dx * (points[i][1] + points[i - 1][1])
    return auc


def eval_kb_surface(
    model: Any, tokenizer: Any, label2id: dict[str, int],
    kb_file: Path, *, device: torch.device, max_length: int, batch_size: int,
) -> dict[str, Any]:
    """KB surface evaluation with Method A/B/C under both projection modes.

    Saves full pre-softmax logits per target (§11.7.4 logit save policy).
    """
    rows = [json.loads(l) for l in kb_file.read_text().splitlines() if l.strip()]
    if not rows:
        return {"status": "empty"}

    texts = [r["text"] for r in rows]
    id2label = {v: k for k, v in label2id.items()}
    labels_ordered = [id2label[i] for i in range(len(label2id))]
    neg_idx = label2id.get("__NEGATIVE__")
    assert neg_idx is not None, "Schema must include __NEGATIVE__"

    # Schema name resolution: label2id shape identifies which schema this is.
    schema_name_by_size = {4: "S_flat", 8: "S_pair", 13: "S_mech"}
    schema = schema_name_by_size.get(len(label2id))
    assert schema is not None, f"Unknown schema for label2id size {len(label2id)}"

    all_logits, probs = logits_and_softmax(
        model, tokenizer, texts, device=device,
        max_length=max_length, batch_size=batch_size,
    )

    # ── Per-target records ────────────────────────────────────────────────
    target_records: list[dict[str, Any]] = []
    # For Method C AUC computation we collect (p_neg, hit) per projection mode.
    hit_rows_sv: list[tuple[float, int]] = []
    hit_rows_sl: list[tuple[float, int]] = []

    for r, pv, lv in zip(rows, probs, all_logits):
        pr, cf = probs_to_pred(pv, id2label)
        p_neg = float(pv[neg_idx])

        # Resolve expected label set under both projection modes.
        civic_like = {
            "expected_pairing_family": r.get("pairing_family"),
            "heuristic_gold_s2_label": r.get("expected_label"),
        }
        _, fam_conf = resolve_family(civic_like, strategy="primary")
        evaluable = fam_conf != "unmapped"

        exp_set_sv, _ = schema_expected_label_set(civic_like, schema, "primary", "set_valued")
        exp_set_sl, _ = schema_expected_label_set(civic_like, schema, "primary", "single_label")

        hit_sv = int(pr in exp_set_sv) if evaluable and exp_set_sv else 0
        hit_sl = int(pr in exp_set_sl) if evaluable and exp_set_sl else 0

        # Method B pmass: sum P over expected set (0 if unmapped or set empty)
        pmass_sv = sum(float(pv[label2id[l]]) for l in exp_set_sv if l in label2id)
        pmass_sl = sum(float(pv[label2id[l]]) for l in exp_set_sl if l in label2id)

        # Legacy p_expected (probability on the S2-vocab expected label, if present).
        s2_expected = r.get("expected_label") or ""
        s2_expected_in_schema = s2_expected in label2id
        p_expected_legacy = float(pv[label2id[s2_expected]]) if s2_expected_in_schema else None

        target_records.append({
            "target_id": r["target_id"],
            "pairing_family": r.get("pairing_family"),
            "expected_label_s2": s2_expected,
            "expected_in_schema_s2": s2_expected_in_schema,
            "evaluable": evaluable,
            "family_confidence": fam_conf,
            "expected_set_sv": sorted(exp_set_sv),
            "expected_set_sl": sorted(exp_set_sl),
            "pred_label": pr,
            "pred_confidence": cf,
            "p_negative": p_neg,
            "p_expected_legacy": p_expected_legacy,
            "surface_score": 1.0 - p_neg,
            "non_negative": pr != "__NEGATIVE__",
            # Method A per-target
            "hit_A_sv": hit_sv,
            "hit_A_sl": hit_sl,
            # Method B per-target
            "pmass_B_sv": pmass_sv,
            "pmass_B_sl": pmass_sl,
            # Full softmax + logits — full-precision preservation (§11.7.4)
            "softmax": pv,
            "logits": lv,
            "label_order": labels_ordered,
        })
        if evaluable:
            hit_rows_sv.append((p_neg, hit_sv))
            hit_rows_sl.append((p_neg, hit_sl))

    n_total = len(target_records)
    n_eval = sum(1 for t in target_records if t["evaluable"])

    # ── Aggregates over all 165 targets (legacy KB_surface_mean) ──────────
    surf_mean = sum(t["surface_score"] for t in target_records) / n_total
    surf_50 = sum(1 for t in target_records if t["surface_score"] > 0.5) / n_total
    non_neg_rate = sum(1 for t in target_records if t["non_negative"]) / n_total

    # ── Method A / B / C aggregates over evaluable targets (162) ──────────
    def mean_or_nan(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")
    eval_targets = [t for t in target_records if t["evaluable"]]
    kb_hit_A_sv = mean_or_nan([t["hit_A_sv"] for t in eval_targets])
    kb_hit_A_sl = mean_or_nan([t["hit_A_sl"] for t in eval_targets])
    kb_pmass_B_sv = mean_or_nan([t["pmass_B_sv"] for t in eval_targets])
    kb_pmass_B_sl = mean_or_nan([t["pmass_B_sl"] for t in eval_targets])
    kb_auc_C_sv = _method_c_auc(hit_rows_sv, _METHOD_C_TAU_GRID)
    kb_auc_C_sl = _method_c_auc(hit_rows_sl, _METHOD_C_TAU_GRID)

    # ── Per-family breakdown on evaluable targets ────────────────────────
    fam_stats: dict[str, dict[str, float]] = defaultdict(lambda: {
        "n": 0,
        "hit_A_sv": 0.0, "hit_A_sl": 0.0,
        "pmass_B_sv": 0.0, "pmass_B_sl": 0.0,
        "surface_mean": 0.0, "non_neg_rate": 0.0,
    })
    for t in eval_targets:
        f = t["pairing_family"] or "unknown"
        fam_stats[f]["n"] += 1
        fam_stats[f]["hit_A_sv"] += t["hit_A_sv"]
        fam_stats[f]["hit_A_sl"] += t["hit_A_sl"]
        fam_stats[f]["pmass_B_sv"] += t["pmass_B_sv"]
        fam_stats[f]["pmass_B_sl"] += t["pmass_B_sl"]
        fam_stats[f]["surface_mean"] += t["surface_score"]
        fam_stats[f]["non_neg_rate"] += 1 if t["non_negative"] else 0
    for f, d in fam_stats.items():
        if d["n"]:
            for k in ("hit_A_sv", "hit_A_sl", "pmass_B_sv", "pmass_B_sl",
                     "surface_mean", "non_neg_rate"):
                d[k] /= d["n"]

    return {
        "schema": schema,
        "n_targets_total": n_total,
        "n_targets_evaluable": n_eval,
        "n_tau_grid_points": len(_METHOD_C_TAU_GRID),
        # Legacy (backwards-compat)
        "kb_surface_mean": surf_mean,
        "kb_surface_50": surf_50,
        "kb_nonneg_rate": non_neg_rate,
        # Method A (primary for §11.7.1.1)
        "kb_hit_A_setvalued": kb_hit_A_sv,
        "kb_hit_A_singlelabel": kb_hit_A_sl,
        # Method B (sensitivity)
        "kb_pmass_B_setvalued": kb_pmass_B_sv,
        "kb_pmass_B_singlelabel": kb_pmass_B_sl,
        # Method C (sensitivity)
        "kb_auc_C_setvalued": kb_auc_C_sv,
        "kb_auc_C_singlelabel": kb_auc_C_sl,
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
    kb = result["kb_surface"]
    print(f"  kb surface_mean={kb['kb_surface_mean']:.4f} "
          f"hit_A_sv={kb['kb_hit_A_setvalued']:.4f} "
          f"hit_A_sl={kb['kb_hit_A_singlelabel']:.4f} "
          f"pmass_B_sv={kb['kb_pmass_B_setvalued']:.4f} "
          f"auc_C_sv={kb['kb_auc_C_setvalued']:.4f} "
          f"in {result['duration_seconds']['kb']:.1f}s")

    result["duration_seconds"]["total"] = round(sum(result["duration_seconds"].values()), 2)

    # Split heavy arrays (softmax + logits) into a separate file. The
    # summary eval JSON stays compact; full per-target logits live in
    # kb_surface_targets.jsonl for post-hoc analysis (§11.7.4).
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
