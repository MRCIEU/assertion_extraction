"""Build per-abstract candidate pools and positive coverage analysis."""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from typing import Any

import pandas as pd

from .config import (
    ALL_PAIR_TYPES,
    DESCRIPTIVE_PAIR_TYPES,
    FROZEN_PROTOCOL_JSON,
    PAIR_TYPE_ENTITY_TYPES,
    PRIMARY_PAIR_TYPES,
)
from .matching import entities_match, normalize_text
from .parse import entities_by_type, parse_entities


def load_frozen_positives() -> pd.DataFrame:
    protocol = json.loads(FROZEN_PROTOCOL_JSON.read_text(encoding="utf-8"))
    return pd.DataFrame(protocol["targets"])


def load_all_grounded_pmids() -> list[str]:
    """All PMIDs from the step-00 abstract-grounded inventory (2074 pairs, ~1079 PMIDs)."""
    from .config import STEP00_OUTPUTS

    alignment = pd.read_csv(STEP00_OUTPUTS / "alignment_details.csv")
    inventory = pd.read_csv(STEP00_OUTPUTS / "evaluable_inventory.csv")
    grounded_ids = set(
        alignment.loc[alignment["both_mentioned"] == True, "evidence_id"].astype(int)  # noqa: E712
    )
    sub = inventory[inventory["evidence_id"].astype(int).isin(grounded_ids)]
    return sorted(sub["pmid"].astype(str).unique().tolist())


def load_variant_positives_for_diagnostics() -> pd.DataFrame:
    """Abstract-grounded variant pairs from step-00 inventory (not ranking targets)."""
    from .config import DESCRIPTIVE_PAIR_TYPES, STEP00_OUTPUTS

    alignment = pd.read_csv(STEP00_OUTPUTS / "alignment_details.csv")
    inventory = pd.read_csv(STEP00_OUTPUTS / "evaluable_inventory.csv")
    grounded_ids = set(
        alignment.loc[alignment["both_mentioned"] == True, "evidence_id"].astype(int)  # noqa: E712
    )
    sub = inventory[inventory["evidence_id"].astype(int).isin(grounded_ids)].copy()
    rows: list[dict[str, Any]] = []
    for _, r in sub.iterrows():
        pt = str(r.get("entity_pair_type") or "").replace("–", "-")
        if pt not in DESCRIPTIVE_PAIR_TYPES:
            continue
        rows.append(
            {
                "target_id": f"rank_{int(r['evidence_id'])}",
                "evidence_id": int(r["evidence_id"]),
                "pmid": str(r["pmid"]),
                "pair_type": pt,
                "scope": "descriptive_only",
                "head_entity": r["head_entity"],
                "head_type": r["head_type"],
                "tail_entity": r["tail_entity"],
                "tail_type": r["tail_type"],
            }
        )
    return pd.DataFrame(rows)


def _pair_key(head: str, tail: str, pair_type: str) -> tuple[str, str, str]:
    return (normalize_text(head), normalize_text(tail), pair_type)


def _asserted_pairs_by_pmid(positives: pd.DataFrame) -> dict[str, set[tuple[str, str, str]]]:
    out: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for _, r in positives.iterrows():
        out[str(r["pmid"])].add(
            _pair_key(str(r["head_entity"]), str(r["tail_entity"]), str(r["pair_type"]))
        )
    return dict(out)


def _find_matching_entity(
    civic_entity: str,
    entities: list[dict[str, Any]],
    civic_type: str = "",
) -> dict[str, Any] | None:
    for ent in entities:
        texts = ent.get("all_texts") or [ent["text"]]
        for text in texts:
            if entities_match(civic_entity, text, civic_type):
                return ent
    return None


def analyze_positive_coverage(
    positives: pd.DataFrame,
    pubtator_docs: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each CIViC positive, check whether PubTator3 identifies head and tail.
    Returns per-positive coverage table and per-entity-type summary.
    """
    rows = []
    entity_slots: list[dict[str, Any]] = []

    for _, r in positives.iterrows():
        pmid = str(r["pmid"])
        doc = pubtator_docs.get(pmid)
        parsed = parse_entities(doc) if doc else []
        by_type = entities_by_type(parsed)

        head_type = str(r["head_type"])
        tail_type = str(r["tail_type"])
        head_match = _find_matching_entity(str(r["head_entity"]), by_type.get(head_type, []), head_type)
        tail_match = _find_matching_entity(str(r["tail_entity"]), by_type.get(tail_type, []), tail_type)

        head_found = head_match is not None
        tail_found = tail_match is not None
        both_found = head_found and tail_found

        rows.append(
            {
                "target_id": r["target_id"],
                "evidence_id": r["evidence_id"],
                "pmid": pmid,
                "pair_type": r["pair_type"],
                "scope": r["scope"],
                "head_entity": r["head_entity"],
                "head_type": head_type,
                "head_found": head_found,
                "tail_entity": r["tail_entity"],
                "tail_type": tail_type,
                "tail_found": tail_found,
                "both_found": both_found,
                "evaluable": both_found,
            }
        )

        for role, etype, found in [
            ("head", head_type, head_found),
            ("tail", tail_type, tail_found),
        ]:
            entity_slots.append(
                {
                    "target_id": r["target_id"],
                    "pmid": pmid,
                    "role": role,
                    "entity_type": etype,
                    "found": found,
                }
            )

    coverage_df = pd.DataFrame(rows)
    slot_df = pd.DataFrame(entity_slots)

    type_summary = (
        slot_df.groupby("entity_type")["found"]
        .agg(n_total="count", n_found="sum")
        .reset_index()
    )
    type_summary["coverage_rate"] = type_summary["n_found"] / type_summary["n_total"]

    return coverage_df, type_summary


def build_candidate_pools(
    positives: pd.DataFrame,
    pubtator_docs: dict[str, dict[str, Any]],
    asserted_by_pmid: dict[str, set[tuple[str, str, str]]],
    pool_pmids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Enumerate co-occurring same-type entity pairs per abstract.
    Returns flat candidate list and per-abstract summaries.
    """
    pmids = pool_pmids or sorted(positives["pmid"].astype(str).unique())
    candidates: list[dict[str, Any]] = []
    abstract_summaries: list[dict[str, Any]] = []

    for pmid in pmids:
        doc = pubtator_docs.get(pmid)
        if not doc:
            abstract_summaries.append(
                {
                    "pmid": pmid,
                    "has_pubtator": False,
                    "n_candidates_total": 0,
                    "n_candidates_primary": 0,
                    "n_positives_in_pool": 0,
                }
            )
            continue

        by_type = entities_by_type(parse_entities(doc))
        asserted = asserted_by_pmid.get(pmid, set())
        pmid_candidates: list[dict[str, Any]] = []

        for pair_type in ALL_PAIR_TYPES:
            head_type, tail_type = PAIR_TYPE_ENTITY_TYPES[pair_type]
            heads = by_type.get(head_type, [])
            tails = by_type.get(tail_type, [])
            if not heads or not tails:
                continue

            scope = "primary" if pair_type in PRIMARY_PAIR_TYPES else "descriptive_only"
            for head, tail in itertools.product(heads, tails):
                if head["entity_key"] == tail["entity_key"]:
                    continue
                is_positive = _pair_key(head["text"], tail["text"], pair_type) in asserted
                # Also match via normalized keys from CIViC asserted set
                if not is_positive:
                    for ah, at, pt in asserted:
                        if pt != pair_type:
                            continue
                        if entities_match(ah, head["text"], head_type) or any(
                            entities_match(ah, t, head_type) for t in head.get("all_texts", [])
                        ):
                            if entities_match(at, tail["text"], tail_type) or any(
                                entities_match(at, t, tail_type) for t in tail.get("all_texts", [])
                            ):
                                is_positive = True
                                break

                cand_id = f"{pmid}_{pair_type}_{head['entity_key']}_{tail['entity_key']}"
                pmid_candidates.append(
                    {
                        "candidate_id": cand_id,
                        "pmid": pmid,
                        "pair_type": pair_type,
                        "scope": scope,
                        "head_entity": head["text"],
                        "head_type": head_type,
                        "head_normalized_id": head.get("normalized_id"),
                        "head_offset": head.get("offset"),
                        "tail_entity": tail["text"],
                        "tail_type": tail_type,
                        "tail_normalized_id": tail.get("normalized_id"),
                        "tail_offset": tail.get("offset"),
                        "is_civic_positive": bool(is_positive),
                    }
                )

        candidates.extend(pmid_candidates)
        primary = [c for c in pmid_candidates if c["scope"] == "primary"]
        n_pos = sum(1 for c in pmid_candidates if c["is_civic_positive"])
        abstract_summaries.append(
            {
                "pmid": pmid,
                "has_pubtator": True,
                "n_candidates_total": len(pmid_candidates),
                "n_candidates_primary": len(primary),
                "n_positives_in_pool": n_pos,
                "positive_fraction": n_pos / len(pmid_candidates) if pmid_candidates else None,
                "n_gene": len(by_type.get("gene", [])),
                "n_drug": len(by_type.get("drug", [])),
                "n_disease": len(by_type.get("disease", [])),
                "n_variant": len(by_type.get("variant", [])),
            }
        )

    return candidates, abstract_summaries
