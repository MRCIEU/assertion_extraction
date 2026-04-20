# -*- coding: utf-8 -*-
"""Stream relation-pair rows from processed JSONL for external evaluation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

from external_evaluation.loaders.jsonl_pairs import doc_to_gold_pair_rows


def pairing_gene_disease(hf: str, tf: str, _: str) -> bool:
    return (hf == "GENE" and tf == "DISEASE") or (hf == "DISEASE" and tf == "GENE")


def pairing_variant_disease(hf: str, tf: str, _: str) -> bool:
    return (hf == "VARIANT" and tf == "DISEASE") or (hf == "DISEASE" and tf == "VARIANT")


def pairing_drug_gene(hf: str, tf: str, _: str) -> bool:
    return (hf == "DRUG" and tf == "GENE") or (hf == "GENE" and tf == "DRUG")


def pairing_drug_disease(hf: str, tf: str, _: str) -> bool:
    return (hf == "DRUG" and tf == "DISEASE") or (hf == "DISEASE" and tf == "DRUG")


def iter_docs_jsonl(path: Path, *, source_split_in: set[str] | None) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if source_split_in is not None:
                sp = d.get("source_split") or ""
                if sp not in source_split_in:
                    continue
            yield d


def stream_pair_rows(
    path: Path,
    *,
    max_pairs: int,
    source_split_in: set[str] | None,
    pairing_filter: Callable[[str, str, str], bool] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collect up to ``max_pairs`` supervised pair rows; optional head/tail pairing filter."""
    rows: list[dict[str, Any]] = []
    doc_ids: set[str] = set()
    for doc in iter_docs_jsonl(path, source_split_in=source_split_in):
        if len(rows) >= max_pairs:
            break
        for r in doc_to_gold_pair_rows(doc):
            if len(rows) >= max_pairs:
                break
            hf = r.get("head_entity_label") or ""
            tf = r.get("tail_entity_label") or ""
            lab = r.get("label") or ""
            if pairing_filter is not None and not pairing_filter(hf, tf, lab):
                continue
            did = str(r.get("doc_id") or r.get("sample_id") or "")
            if did:
                doc_ids.add(did)
            rows.append(r)
    stats = {
        "n_documents": len(doc_ids),
        "n_examples": len(rows),
        "n_positive_instances": len(rows),
    }
    return rows, stats


def rows_to_eval_input(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Minimal dicts for ``evaluate_checkpoint``."""
    return [{"text": r["text"], "label": r["label"]} for r in rows]


def label_support_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(r["label"] for r in rows))
