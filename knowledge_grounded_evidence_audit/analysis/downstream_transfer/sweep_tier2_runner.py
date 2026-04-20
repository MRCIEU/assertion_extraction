"""
Tier-2 GPU sweep: selected families × all seeds × high-value downstream settings.

Settings (from Tier-1 design + value):
  S1_current_realistic — R1_current × C1_abstract, linkage L1_strict
  S2_improved_realistic — R2_expanded_lexical × C4_window, linkage L2_relaxed
  S3_oracle_like — O3_oracle_pair_sentence aggregate ALL, linkage L1_strict on oracle preds
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

_KG = Path(__file__).resolve().parent.parent.parent
if str(_KG) not in sys.path:
    sys.path.insert(0, str(_KG))

from run_pipeline import RawAssertion
from inference.predict_checkpoint import load_model_from_checkpoint, predict_labels

from .linkage_adapter import classify_linkage_outcome
from .paths import FT_RUNS, MANIFESTS, PROC, REPORTS, TABLES, ensure_dirs
from .sweep_tier1_runner import (
    R2_HINT,
    _ckpt,
    _macro_f1,
    _read_csv,
    _trainer,
)

CTX = {
    "C1_abstract": lambda e: (e.get("context_C1_abstract") or "")[:2400],
    "C2_sentence": lambda e: e.get("context_C2_sentence_only") or "",
    "C3_pm1": lambda e: e.get("context_C3_pm1") or "",
    "C4_window": lambda e: e.get("context_C4_window") or "",
    "C5_pmc_skip": lambda e: e.get("context_C2_sentence_only") or "",
}


def _eval_rc_block(
    model: Any,
    tok: Any,
    l2i: Dict[str, int],
    labels_ordered: Sequence[str],
    base: str,
    seed: str,
    R: str,
    cname: str,
    ev_list: List[Dict[str, str]],
    targets: Dict[str, Dict[str, str]],
    harm: List[Dict[str, str]],
    device: torch.device,
    batch_size: int,
    linkage_mode: str,
) -> Dict[str, Any]:
    fn = CTX[cname]
    texts, golds, metas = [], [], []
    for ev in ev_list:
        t = targets.get(ev["goldlite_target_id"], {})
        head = ev.get("oracle_head_entity") or t.get("gene", "")
        tail = ev.get("oracle_tail_entity") or t.get("drug_primary", "") or "lung_cancer_context"
        ctx = fn(ev)
        if R == "R2_expanded_lexical" and cname in ("C1_abstract", "C4_window"):
            ctx = (ctx or "") + R2_HINT
        if len(ctx) < 6:
            ctx = (ev.get("context_C1_abstract") or "")[:2400]
        texts.append(_trainer(head, tail, ctx))
        golds.append(t.get("heuristic_gold_s2_label", "ASSOCIATION_GENERAL"))
        metas.append((ev, t, head, tail, cname))
    preds, confs = predict_labels(
        model,
        tok,
        l2i,
        [{"text": x} for x in texts],
        device=device,
        batch_size=batch_size,
    )
    labs = sorted(set(golds) | set(preds) | set(labels_ordered))
    mf1 = _macro_f1(golds, preds, labs)
    nn = sum(1 for p in preds if p != "__NEGATIVE__")
    kb_s = amb = gap = uns = 0
    for (ev, t, head, tail, _cn), pred, conf in zip(metas, preds, confs):
        fam = "gene_disease"
        if pred == "DRUG_GENE_REGULATION":
            fam = "drug_gene"
        elif pred == "VARIANT_GENE":
            fam = "variant_disease"
        elif pred == "DRUG_DISEASE":
            fam = "drug_disease"
        a = RawAssertion(
            assertion_id=f"t2|{ev['goldlite_target_id']}|{base}|{R}|{cname}",
            model_id=f"{base}_s{seed}",
            doc_pmid=ev.get("pmid", ""),
            sentence=(fn(ev) or "")[:800],
            relation_family=fam if pred != "__NEGATIVE__" else "negative",
            entity_a={"type": "gene", "text": head, "normalized": head},
            entity_b={"type": "entity", "text": tail, "normalized": tail},
            confidence=float(conf),
            provenance=["tier2_transfer", R, cname, linkage_mode],
        )
        oc, _lvl = classify_linkage_outcome(a, harm, linkage_mode)
        if oc == "kb_supported_aligned":
            kb_s += 1
        elif oc == "conflict_or_ambiguity":
            amb += 1
        elif oc == "literature_supported_kb_absent_candidate":
            gap += 1
        else:
            uns += 1
    n = len(golds)
    return {
        "macro_f1_heuristic": str(mf1),
        "pred_nonnegative_count": str(nn),
        "pred_nonnegative_rate": str(round(nn / max(1, n), 6)),
        "n_items": str(n),
        "kb_supported_aligned": str(kb_s),
        "conflict_or_ambiguity": str(amb),
        "literature_kb_absent": str(gap),
        "unsupported_or_low_trust": str(uns),
        "support_ready_rate": str(round(kb_s / max(1, n), 6)),
        "ambiguity_rate": str(round(amb / max(1, n), 6)),
    }


def _eval_o3_all(
    model: Any,
    tok: Any,
    l2i: Dict[str, int],
    labels_ordered: Sequence[str],
    base: str,
    seed: str,
    ev_list: List[Dict[str, str]],
    targets: Dict[str, Dict[str, str]],
    harm: List[Dict[str, str]],
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    """O3_oracle_pair_sentence on C2_sentence; report ALL macro-F1 + nn + linkage (L1)."""
    fn = CTX["C2_sentence"]
    texts, golds, metas = [], [], []
    for ev in ev_list:
        t = targets.get(ev["goldlite_target_id"], {})
        head = ev.get("oracle_head_entity") or t.get("gene", "")
        tail = ev.get("oracle_tail_entity") or t.get("drug_primary", "") or "lung_cancer_context"
        texts.append(_trainer(head, tail, fn(ev)))
        golds.append(t.get("heuristic_gold_s2_label", "ASSOCIATION_GENERAL"))
        metas.append((ev, t, head, tail))
    preds, confs = predict_labels(
        model,
        tok,
        l2i,
        [{"text": x} for x in texts],
        device=device,
        batch_size=batch_size,
    )
    labs = sorted(set(golds) | set(preds) | set(labels_ordered))
    mf1 = _macro_f1(golds, preds, labs)
    nn = sum(1 for p in preds if p != "__NEGATIVE__")
    n = len(golds)
    kb_s = amb = gap = uns = 0
    for (ev, t, head, tail), pred, conf in zip(metas, preds, confs):
        fam = "gene_disease"
        if pred == "DRUG_GENE_REGULATION":
            fam = "drug_gene"
        elif pred == "VARIANT_GENE":
            fam = "variant_disease"
        elif pred == "DRUG_DISEASE":
            fam = "drug_disease"
        a = RawAssertion(
            assertion_id=f"t2o3|{ev['goldlite_target_id']}|{base}",
            model_id=f"{base}_s{seed}",
            doc_pmid=ev.get("pmid", ""),
            sentence=(fn(ev) or "")[:800],
            relation_family=fam if pred != "__NEGATIVE__" else "negative",
            entity_a={"type": "gene", "text": head, "normalized": head},
            entity_b={"type": "entity", "text": tail, "normalized": tail},
            confidence=float(conf),
            provenance=["tier2_transfer", "O3_oracle_pair_sentence", "L1_strict"],
        )
        oc, _ = classify_linkage_outcome(a, harm, "L1_strict")
        if oc == "kb_supported_aligned":
            kb_s += 1
        elif oc == "conflict_or_ambiguity":
            amb += 1
        elif oc == "literature_supported_kb_absent_candidate":
            gap += 1
        else:
            uns += 1
    return {
        "macro_f1_heuristic": str(mf1),
        "oracle_o3_macro_f1_all": str(mf1),
        "pred_nonnegative_count": str(nn),
        "pred_nonnegative_rate": str(round(nn / max(1, n), 6)),
        "n_items": str(n),
        "kb_supported_aligned": str(kb_s),
        "conflict_or_ambiguity": str(amb),
        "literature_kb_absent": str(gap),
        "unsupported_or_low_trust": str(uns),
        "support_ready_rate": str(round(kb_s / max(1, n), 6)),
        "ambiguity_rate": str(round(amb / max(1, n), 6)),
    }


def _load_decision() -> Dict[str, Any]:
    p = PROC / "tier2_family_selection_decision.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p} — create Tier-2 family selection first.")
    return json.loads(p.read_text(encoding="utf-8"))


def run_tier2_sweep() -> Dict[str, Any]:
    ensure_dirs()
    dec = _load_decision()
    families: List[str] = dec.get("selected_families") or []
    seeds: List[str] = [str(s).zfill(2) for s in dec.get("seeds", ["01", "02", "03", "04", "05"])]
    if not families:
        return {"error": "no families in tier2_family_selection_decision.json"}

    ev_list = _read_csv(PROC / "goldlite_evidence_candidates.csv")
    targets = {r["goldlite_target_id"]: r for r in _read_csv(PROC / "goldlite_audit_targets.csv")}
    harm = _read_csv(PROC / "kb_target_ledger_harmonized.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(__import__("os").environ.get("KG_AUDIT_BATCH_SIZE", "8"))

    rows: List[Dict[str, str]] = []
    skipped: List[str] = []

    settings: List[Tuple[str, str, str, str, str]] = [
        ("S1_current_realistic", "R1_current", "C1_abstract", "L1_strict", "rc"),
        ("S2_improved_realistic", "R2_expanded_lexical", "C4_window", "L2_relaxed", "rc"),
        ("S3_oracle_like", "O3_oracle_pair_sentence", "C2_sentence", "L1_strict", "oracle_o3"),
    ]

    for base in families:
        for seed in seeds:
            ck = _ckpt(base, seed)
            if not ck.is_file():
                skipped.append(f"{base}_s{seed}")
                continue
            model, tok, l2i, labels_ordered, _ = load_model_from_checkpoint(ck, device, state_dict_strict=True)
            try:
                for sid, R, cname, lk, kind in settings:
                    if kind == "rc":
                        out = _eval_rc_block(
                            model,
                            tok,
                            l2i,
                            labels_ordered,
                            base,
                            seed,
                            R,
                            cname,
                            ev_list,
                            targets,
                            harm,
                            device,
                            batch_size,
                            lk,
                        )
                    else:
                        out = _eval_o3_all(
                            model,
                            tok,
                            l2i,
                            labels_ordered,
                            base,
                            seed,
                            ev_list,
                            targets,
                            harm,
                            device,
                            batch_size,
                        )
                    row = {
                        "model_base_id": base,
                        "model_seed_id": seed,
                        "downstream_setting_id": sid,
                        "retrieval_variant": R,
                        "context_variant": cname,
                        "linkage_mode": lk,
                        **out,
                    }
                    if "oracle_o3_macro_f1_all" not in row:
                        row["oracle_o3_macro_f1_all"] = ""
                    rows.append(row)
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if rows:
        keys: set = set()
        for r in rows:
            keys |= set(r.keys())
        fieldnames = sorted(keys)
        with open(TABLES / "tier2_multiseed_raw.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

    (MANIFESTS / "tier2_sweep_status.json").write_text(
        json.dumps(
            {
                "status": "completed" if rows else "empty",
                "rows": len(rows),
                "skipped_checkpoints": skipped,
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"rows": len(rows), "skipped": len(skipped)}

