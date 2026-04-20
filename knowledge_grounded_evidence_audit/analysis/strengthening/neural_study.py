"""
GPU checkpoint runs for context, oracle, proposal metrics on gold-lite.

Imports pipeline types and inference stack; does not use placeholder extraction.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

_KG = Path(__file__).resolve().parent.parent.parent
if str(_KG) not in sys.path:
    sys.path.insert(0, str(_KG))

from run_pipeline import (
    RawAssertion,
    SENT_SPLIT,
    collect_lexicons,
    find_drugs,
    find_genes,
    parse_pubmed_article,
)
from inference.predict_checkpoint import load_model_from_checkpoint, predict_labels

from .linkage_modes import audit_outcome_variant, link_to_kb_variant
from .paths import CACHE, PROC, PROJECT_ROOT, TABLES, REPORTS, MANIFESTS, ensure_dirs

MODELS = ["M015", "M021", "M003", "S002"]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _checkpoint_path(model_id: str) -> Path:
    return (
        PROJECT_ROOT
        / "fine_tuning_experiments"
        / "runs"
        / f"HR_{model_id}_s01"
        / "checkpoints"
        / "best.pt"
    )


def _trainer(head: str, tail: str, ctx: str) -> str:
    return f"{head.strip()} [ENT] {tail.strip()} [SEP] {ctx.strip()[:2500]}"


def _macro_f1(y_true: List[str], y_pred: List[str], labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    fs = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        fs.append(f1)
    return round(sum(fs) / len(fs), 4)


def _load_harm_rows() -> List[Dict[str, str]]:
    return _read_csv(PROC / "kb_target_ledger_harmonized.csv")


def _proposal_metrics_for_target(
    pmid: str,
    gene: str,
    drug: str,
    variant: str,
    genes: set,
    drugs: set,
    ev_sentence: str,
) -> Dict[str, float]:
    xp = CACHE / f"{pmid}.xml"
    if not xp.is_file():
        return {f"P{i}": 0.0 for i in range(1, 6)}
    title, abstract = parse_pubmed_article(xp.read_text(encoding="utf-8", errors="replace"))
    doc = f"{title}. {abstract}"
    sents = [s.strip() for s in SENT_SPLIT.split(doc) if len(s.strip()) > 20]

    def sent_ok_gd(s: str) -> bool:
        if not gene or not drug:
            return False
        return bool(re.search(rf"\b{re.escape(gene)}\b", s, re.I)) and bool(
            re.search(rf"\b{re.escape(drug)}\b", s, re.I)
        )

    p1 = any(sent_ok_gd(s) for s in sents)
    p2_gd = p1
    p2_gdis = any(
        find_genes(s, genes) and LUNG_RE.search(s)
        for s in sents
    )
    p2_dd = any(find_drugs(s, drugs) and LUNG_RE.search(s) for s in sents)
    var_hit = any(
        len(tok) >= 4 and tok.lower() in s.lower()
        for s in sents
        for tok in re.findall(r"[A-Za-z0-9]+", variant or "")
    )
    p2_vd = var_hit and any(LUNG_RE.search(s) for s in sents)
    p2 = float(p2_gd or p2_gdis or p2_dd or p2_vd)

    p3 = p1  # KB lexicon already drives genes/drugs in this project

    p4 = sent_ok_gd(ev_sentence) if ev_sentence else False

    p5 = True

    return {
        "P1_gene_drug": float(p1),
        "P2_expanded_binary": p2,
        "P3_kb_constrained": float(p3),
        "P4_sentence_conditioned": float(p4),
        "P5_oracle_pair": float(p5),
    }


LUNG_RE = re.compile(
    r"lung|nsclc|adenocarcinoma|carcinoma|neoplasm",
    re.I,
)


def run_neural_experiments() -> Dict[str, Any]:
    ensure_dirs()
    harm = _load_harm_rows()
    gene_set = {h["gene"] for h in harm}
    targets = {r["goldlite_target_id"]: r for r in _read_csv(PROC / "goldlite_audit_targets.csv")}
    ev_list = _read_csv(PROC / "goldlite_evidence_candidates.csv")
    if not ev_list:
        return {"error": "Missing goldlite_evidence_candidates.csv"}

    genes, drugs, _ = collect_lexicons(harm)

    # All S2 labels union
    label_union: List[str] = []

    context_results: List[Dict[str, Any]] = []
    oracle_results: List[Dict[str, Any]] = []
    proposal_rows: List[Dict[str, str]] = []
    proposal_density: List[Dict[str, str]] = []
    proposal_family: List[Dict[str, str]] = []

    # Proposal stats aggregated
    prop_sum = defaultdict(float)
    prop_n = 0

    for ev in ev_list:
        tid = ev["goldlite_target_id"]
        t = targets.get(tid, {})
        pm = ev.get("pmid", "").strip()
        gene = t.get("gene", "")
        drug = t.get("drug_primary", "")
        variant = t.get("variant_text", "")
        gold_lab = t.get("heuristic_gold_s2_label", "ASSOCIATION_GENERAL")

        pmets = _proposal_metrics_for_target(pm, gene, drug, variant, genes, drugs, ev.get("evidence_sentence", ""))
        prop_n += 1
        for k, v in pmets.items():
            prop_sum[k] += v
        proposal_rows.append(
            {
                "goldlite_target_id": tid,
                "pmid": pm,
                **{k: str(int(v)) for k, v in pmets.items()},
            }
        )

    for pk, pv in prop_sum.items():
        proposal_density.append(
            {
                "proposal_variant": pk,
                "mean_recall_on_goldlite": round(pv / max(1, prop_n), 4),
                "targets": str(prop_n),
            }
        )

    fam_ct = Counter(t.get("expected_pairing_family", "") for t in targets.values())
    for fam, c in fam_ct.items():
        proposal_family.append({"pairing_family": fam, "goldlite_target_count": str(c)})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(__import__("os").environ.get("KG_AUDIT_BATCH_SIZE", "8"))
    per_row_c1: List[Dict[str, str]] = []

    CTX_MAP = {
        "C1_abstract": lambda e: (e.get("context_C1_abstract") or "")[:2400],
        "C2_sentence": lambda e: e.get("context_C2_sentence_only") or "",
        "C3_pm1": lambda e: e.get("context_C3_pm1") or "",
        "C4_window": lambda e: e.get("context_C4_window") or "",
        "C5_pmc_skip": lambda e: e.get("context_C2_sentence_only") or "",
    }

    oracle_defs = [
        ("O1_oracle_pair", "C1_abstract"),
        ("O2_oracle_sentence", "C2_sentence"),
        ("O3_oracle_pair_sentence", "C2_sentence"),
        ("O4_oracle_pair_rich_excerpt", "C4_window"),
    ]

    for model_id in MODELS:
        ck = _checkpoint_path(model_id)
        if not ck.is_file():
            print(f"[strengthening] skip model {model_id}: missing checkpoint {ck}", flush=True)
            continue
        model, tok, l2i, labels_ordered, _ = load_model_from_checkpoint(ck, device, state_dict_strict=True)

        try:
            # --- Context variants (oracle head/tail) ---
            for ctx_name, ctx_fn in CTX_MAP.items():
                texts = []
                golds = []
                metas = []
                for ev in ev_list:
                    t = targets.get(ev["goldlite_target_id"], {})
                    head = ev.get("oracle_head_entity") or t.get("gene", "")
                    tail = ev.get("oracle_tail_entity") or t.get("drug_primary", "") or "lung_cancer_context"
                    ctx = ctx_fn(ev)
                    if len(ctx) < 8:
                        ctx = ev.get("context_C1_abstract", "")[:2400]
                    texts.append(_trainer(head, tail, ctx))
                    golds.append(t.get("heuristic_gold_s2_label", "ASSOCIATION_GENERAL"))
                    metas.append((ev, t, head, tail))

                preds, confs = predict_labels(
                    model, tok, l2i, [{"text": x} for x in texts],
                    device=device, batch_size=batch_size,
                )

                labels_for_f1 = sorted(set(golds) | set(preds) | set(labels_ordered))
                # macro f1 expects string labels as in data
                mf1 = _macro_f1(golds, preds, labels_for_f1)
                nonneg_gold = sum(1 for g in golds if g != "__NEGATIVE__")
                nonneg_pred = sum(1 for p in preds if p != "__NEGATIVE__")
                prec_nn = (
                    sum(1 for g, p in zip(golds, preds) if p != "__NEGATIVE__" and g == p)
                    / max(1, sum(1 for p in preds if p != "__NEGATIVE__"))
                )

                out_counts = Counter()
                amb = 0
                for (ev, t, head, tail), pred, conf in zip(metas, preds, confs):
                    fam = "gene_disease"
                    if pred == "DRUG_GENE_REGULATION":
                        fam = "drug_gene"
                    elif pred == "DRUG_DISEASE":
                        fam = "drug_disease"
                    elif pred == "VARIANT_GENE":
                        fam = "variant_disease"
                    a = RawAssertion(
                        assertion_id=f"ctx|{ev['goldlite_target_id']}|{model_id}",
                        model_id=model_id,
                        doc_pmid=ev.get("pmid", ""),
                        sentence=ctx_fn(ev)[:800],
                        relation_family=fam if pred != "__NEGATIVE__" else "negative",
                        entity_a={"type": "gene", "text": head, "normalized": head},
                        entity_b={"type": "entity", "text": tail, "normalized": tail},
                        confidence=float(conf),
                        provenance=["strengthening_context_variant", ctx_name],
                    )
                    lvl, _, _ = link_to_kb_variant(a, harm, "L1_strict")
                    in_g = head in gene_set
                    oc = audit_outcome_variant(a, lvl, in_g)
                    out_counts[oc] += 1
                    if oc == "conflict_or_ambiguity":
                        amb += 1
                    if ctx_name == "C1_abstract":
                        per_row_c1.append(
                            {
                                "goldlite_target_id": ev["goldlite_target_id"],
                                "model_id": model_id,
                                "pred_label": pred,
                                "gold_label": t.get("heuristic_gold_s2_label", ""),
                                "confidence": str(conf),
                                "head": head,
                                "tail": tail,
                                "sentence_excerpt": (ctx_fn(ev) or "")[:400],
                            }
                        )

                context_results.append(
                    {
                        "model_id": model_id,
                        "context_variant": ctx_name,
                        "macro_f1_vs_heuristic_gold": mf1,
                        "nonnegative_precision_proxy": round(prec_nn, 4),
                        "pred_nonnegative_count": nonneg_pred,
                        "gold_nonnegative_count": nonneg_gold,
                        "kb_supported_aligned": out_counts.get("kb_supported_aligned", 0),
                        "conflict_or_ambiguity": amb,
                        "literature_kb_absent": out_counts.get("literature_supported_kb_absent_candidate", 0),
                        "unsupported_or_low_trust": out_counts.get("unsupported_or_low_trust", 0),
                    }
                )

            # --- Oracle block (explicit conditions) ---
            for oc_name, ctx_key in oracle_defs:
                ctx_fn = CTX_MAP[ctx_key] if ctx_key in CTX_MAP else CTX_MAP["C1_abstract"]
                texts = []
                golds = []
                metas = []
                for ev in ev_list:
                    t = targets.get(ev["goldlite_target_id"], {})
                    head = ev.get("oracle_head_entity") or t.get("gene", "")
                    tail = ev.get("oracle_tail_entity") or t.get("drug_primary", "") or "lung_cancer_context"
                    ctx = ctx_fn(ev)
                    texts.append(_trainer(head, tail, ctx))
                    golds.append(t.get("heuristic_gold_s2_label", "ASSOCIATION_GENERAL"))
                    metas.append((ev, t))

                preds, confs = predict_labels(
                    model, tok, l2i, [{"text": x} for x in texts],
                    device=device, batch_size=batch_size,
                )
                labs = sorted(set(golds) | set(preds) | set(labels_ordered))
                mf1 = _macro_f1(golds, preds, labs)

                per_fam: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
                for (ev, t), g, p in zip(metas, golds, preds):
                    per_fam[t.get("expected_pairing_family", "unknown")].append((g, p))

                for fam, pairs in per_fam.items():
                    gf = [x[0] for x in pairs]
                    pf = [x[1] for x in pairs]
                    oracle_results.append(
                        {
                            "model_id": model_id,
                            "oracle_condition": oc_name,
                            "pairing_family": fam,
                            "macro_f1": _macro_f1(gf, pf, sorted(set(gf) | set(pf))),
                            "n": len(pairs),
                        }
                    )

                oracle_results.append(
                    {
                        "model_id": model_id,
                        "oracle_condition": oc_name,
                        "pairing_family": "ALL",
                        "macro_f1": mf1,
                        "n": len(golds),
                    }
                )

        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if per_row_c1:
        (PROC / "strengthening_per_row_C1.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in per_row_c1),
            encoding="utf-8",
        )

    # Write tables
    if context_results:
        fields = list(context_results[0].keys())
        with open(TABLES / "context_variant_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in context_results:
                w.writerow({k: str(r[k]) for k in fields})

        ctx_sorted = sorted(context_results, key=lambda x: (x["model_id"], x["context_variant"]))
        with open(TABLES / "context_variant_by_model.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in ctx_sorted:
                w.writerow({k: str(r[k]) for k in fields})

    if oracle_results:
        with open(TABLES / "oracle_upper_bound_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(oracle_results[0].keys()))
            w.writeheader()
            for r in oracle_results:
                w.writerow({k: str(r[k]) for k in r})

        fam_only = [r for r in oracle_results if r["pairing_family"] != "ALL"]
        with open(TABLES / "oracle_family_results.csv", "w", newline="", encoding="utf-8") as f:
            if fam_only:
                w = csv.DictWriter(f, fieldnames=list(fam_only[0].keys()))
                w.writeheader()
                for r in fam_only:
                    w.writerow({k: str(r[k]) for k in r})

    with open(TABLES / "proposal_variant_results.csv", "w", newline="", encoding="utf-8") as f:
        if proposal_rows:
            w = csv.DictWriter(f, fieldnames=list(proposal_rows[0].keys()))
            w.writeheader()
            w.writerows(proposal_rows)

    with open(TABLES / "proposal_density_table.csv", "w", newline="", encoding="utf-8") as f:
        if proposal_density:
            w = csv.DictWriter(f, fieldnames=list(proposal_density[0].keys()))
            w.writeheader()
            w.writerows(proposal_density)

    with open(TABLES / "proposal_family_coverage.csv", "w", newline="", encoding="utf-8") as f:
        if proposal_family:
            w = csv.DictWriter(f, fieldnames=list(proposal_family[0].keys()))
            w.writeheader()
            w.writerows(proposal_family)

    prop_status = {
        "targets": prop_n,
        "mean_recalls": {k: round(prop_sum[k] / max(1, prop_n), 4) for k in prop_sum},
    }
    with open(MANIFESTS / "proposal_variant_status.json", "w", encoding="utf-8") as f:
        json.dump(prop_status, f, indent=2)

    # Markdown stubs filled if empty models
    (REPORTS / "context_variant_analysis.md").write_text(
        _md_context(context_results),
        encoding="utf-8",
    )
    (REPORTS / "oracle_upper_bound_analysis.md").write_text(
        _md_oracle(oracle_results),
        encoding="utf-8",
    )
    (REPORTS / "proposal_variant_analysis.md").write_text(
        _md_proposal(proposal_density),
        encoding="utf-8",
    )

    return {"context_rows": len(context_results), "oracle_rows": len(oracle_results), "device": str(device)}


def _md_context(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "# Context variant analysis\n\nNo results (checkpoints missing or gold-lite empty).\n"
    lines = ["# Context variant analysis", "", "| model | context | macro_F1 | kb_supported | ambiguity |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['model_id']} | {r['context_variant']} | {r['macro_f1_vs_heuristic_gold']} | "
            f"{r['kb_supported_aligned']} | {r['conflict_or_ambiguity']} |"
        )
    lines.append("\nHeuristic gold S2 labels — interpret F1 as **ceiling proxy**, not clinical accuracy.\n")
    return "\n".join(lines)


def _md_oracle(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "# Oracle upper-bound analysis\n\nNo results.\n"
    sub = [r for r in rows if r["pairing_family"] == "ALL"]
    lines = ["# Oracle upper-bound analysis", "", "| model | condition | macro_F1 | n |", "|---|---|---|---|"]
    for r in sub:
        lines.append(f"| {r['model_id']} | {r['oracle_condition']} | {r['macro_f1']} | {r['n']} |")
    lines.append(
        "\nIf macro-F1 remains **low** even under O3/O4, **classification / label schema mismatch** is a plausible bottleneck; "
        "if F1 **jumps** vs C1_abstract rows, **context localization** was suppressing signal.\n"
    )
    return "\n".join(lines)


def _md_proposal(density: List[Dict[str, str]]) -> str:
    lines = ["# Proposal-space analysis", "", "| variant | mean recall on gold-lite |", "|---|---|"]
    for r in density:
        lines.append(f"| {r['proposal_variant']} | {r['mean_recall_on_goldlite']} |")
    lines.append("\nP5 oracle pair is **1.0** by construction (pair given).\n")
    return "\n".join(lines)
