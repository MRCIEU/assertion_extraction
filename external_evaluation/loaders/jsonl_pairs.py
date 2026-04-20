# -*- coding: utf-8 -*-
"""Load relation pair rows from packaged JSONL (same layout as scientific_data)."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


def doc_to_gold_pair_rows(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per gold-supervised relation (positives only)."""
    ent_by_id = {e["entity_id"]: e for e in doc.get("entities") or []}
    id_list = list(ent_by_id.keys())
    gold_pairs: set[tuple[str, str]] = set()
    for rel in doc.get("relations") or []:
        hid = rel.get("head_entity_id")
        tid = rel.get("tail_entity_id")
        if hid and tid:
            gold_pairs.add((hid, tid))
            gold_pairs.add((tid, hid))

    text = doc.get("text") or ""
    rows: list[dict[str, Any]] = []
    for rel in doc.get("relations") or []:
        if not rel.get("is_gold_supervision", True):
            continue
        hid = rel.get("head_entity_id")
        tid = rel.get("tail_entity_id")
        if not hid or not tid or hid not in ent_by_id or tid not in ent_by_id:
            continue
        h, t = ent_by_id[hid], ent_by_id[tid]
        seq = f"{h['text']} [ENT] {t['text']} [SEP] {text}"
        hl, tl = h.get("mapped_label", ""), t.get("mapped_label", "")
        rows.append(
            {
                "text": seq[:8000],
                "label": rel["mapped_label"],
                "sample_id": doc.get("sample_id", ""),
                "doc_id": doc.get("doc_id", ""),
                "source_dataset": doc.get("source_dataset"),
                "source_split": doc.get("source_split"),
                "relation_family": rel.get("relation_family"),
                "supervision": "gold",
                "ent_by_id": ent_by_id,
                "ent_ids_list": id_list,
                "gold_pairs": gold_pairs,
                "entity_ids": (hid, tid),
                "doc_text": text,
                "head_entity_label": hl,
                "tail_entity_label": tl,
            }
        )
    return rows


def load_docs_from_jsonl(path: Path, *, split_filter: str | None = None) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if split_filter is not None and d.get("source_split") != split_filter:
                continue
            docs.append(d)
    return docs


def pairing_key(row: dict[str, Any]) -> str:
    h = row.get("head_entity_label") or ""
    t = row.get("tail_entity_label") or ""
    return f"{h}__{t}"


def add_eval_negatives(
    positive_rows: list[dict[str, Any]],
    rng: random.Random,
    *,
    negative_ratio: float = 2.0,
    max_negatives_per_positive: int = 4,
) -> list[dict[str, Any]]:
    """
    Add __NEGATIVE__ rows using same-document random non-gold pairs (protocol-fixed RNG).
    Documented in strict_realism_protocol.json (negative_sampling_eval).
    """
    neg_label = "__NEGATIVE__"
    out: list[dict[str, Any]] = []
    n_neg_target = min(max_negatives_per_positive, max(1, int(negative_ratio)))
    pool_text = [r["text"] for r in positive_rows if r.get("supervision") == "gold"]
    for ex in positive_rows:
        out.append(ex)
        ent_by_id = ex.get("ent_by_id")
        id_list = ex.get("ent_ids_list") or []
        doc_text = ex.get("doc_text", "")
        if not ent_by_id or len(id_list) < 2:
            for _ in range(n_neg_target):
                donor = rng.choice(positive_rows)
                if donor is ex and len(positive_rows) > 1:
                    donor = rng.choice([r for r in positive_rows if r is not ex])
                neg = {
                    **{k: v for k, v in ex.items() if k not in ("ent_by_id", "entity_ids", "gold_pairs")},
                    "text": donor["text"][:8000],
                    "label": neg_label,
                    "supervision": "negative_sample",
                    "relation_family": None,
                    "head_entity_label": "",
                    "tail_entity_label": "",
                }
                out.append(neg)
            continue
        gp = ex.get("gold_pairs") or set()
        added = 0
        for _ in range(256 * n_neg_target):
            if added >= n_neg_target:
                break
            h2 = rng.choice(id_list)
            t2 = rng.choice(id_list)
            if h2 == t2:
                continue
            if (h2, t2) in gp or (t2, h2) in gp:
                continue
            h_e, t_e = ent_by_id[h2], ent_by_id[t2]
            seq = f"{h_e['text']} [ENT] {t_e['text']} [SEP] {doc_text}"[:8000]
            out.append(
                {
                    "text": seq,
                    "label": neg_label,
                    "sample_id": ex["sample_id"],
                    "doc_id": ex["doc_id"],
                    "source_dataset": ex["source_dataset"],
                    "source_split": ex["source_split"],
                    "relation_family": None,
                    "supervision": "negative_sample",
                    "head_entity_label": h_e.get("mapped_label", ""),
                    "tail_entity_label": t_e.get("mapped_label", ""),
                }
            )
            added += 1
        while added < n_neg_target and pool_text:
            neg = {
                **{k: v for k, v in ex.items() if k not in ("ent_by_id", "entity_ids", "gold_pairs")},
                "text": rng.choice(pool_text)[:8000],
                "label": neg_label,
                "supervision": "negative_sample",
                "head_entity_label": "",
                "tail_entity_label": "",
            }
            out.append(neg)
            added += 1
    return out


def strip_heavy(row: dict[str, Any]) -> None:
    row.pop("ent_by_id", None)
    row.pop("gold_pairs", None)


def subset_by_pairing(rows: list[dict[str, Any]], allowed_head_tail: set[tuple[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        key = (r.get("head_entity_label") or "", r.get("tail_entity_label") or "")
        if key in allowed_head_tail:
            out.append(r)
    return out


def subset_oncology_context(rows: list[dict[str, Any]], doc_meta: dict[str, bool]) -> list[dict[str, Any]]:
    """doc_meta: sample_id -> True if oncology-context doc."""
    return [r for r in rows if doc_meta.get(r.get("sample_id", ""), False)]


def build_doc_oncology_flag(docs: list[dict[str, Any]]) -> dict[str, bool]:
    """Heuristic: any entity flagged oncology-projected relevant or cancer-like disease text."""
    flags: dict[str, bool] = {}
    cancer_tokens = ("cancer", "tumor", "tumour", "carcinoma", "sarcoma", "lymphoma", "leukemia", "oncolog")
    for d in docs:
        sid = d.get("sample_id", "")
        ok = False
        for e in d.get("entities") or []:
            if e.get("is_oncology_projected_relevant"):
                ok = True
                break
            if e.get("mapped_label") == "DISEASE":
                et = (e.get("text") or "").lower()
                if any(t in et for t in cancer_tokens):
                    ok = True
                    break
        flags[sid] = ok
    return flags


def aggregate_support(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_doc = len({r.get("sample_id") for r in rows})
    labs = Counter(r["label"] for r in rows)
    n_pos = sum(c for lab, c in labs.items() if lab != "__NEGATIVE__")
    n_neg = labs.get("__NEGATIVE__", 0)
    return {
        "n_documents": n_doc,
        "n_examples": len(rows),
        "n_positive_instances": n_pos,
        "n_negative_instances": n_neg,
        "label_counts_json": json.dumps(dict(labs)),
    }
