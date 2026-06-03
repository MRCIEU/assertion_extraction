"""Build frozen ranking evaluation targets from step-00 alignment."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import PRIMARY_PAIR_TYPES, STEP00_OUTPUTS


def build_ranking_targets() -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Return evaluable ranking targets (gene-drug + gene-disease only) and inventory counts.
    Variant pairs in the step-00 inventory are excluded — they cannot enter the candidate pool.
    """
    alignment_path = STEP00_OUTPUTS / "alignment_details.csv"
    inventory_path = STEP00_OUTPUTS / "evaluable_inventory.csv"
    alignment = pd.read_csv(alignment_path)
    inventory = pd.read_csv(inventory_path)

    grounded_ids = set(
        alignment.loc[alignment["both_mentioned"] == True, "evidence_id"].astype(int)  # noqa: E712
    )
    sub = inventory[inventory["evidence_id"].astype(int).isin(grounded_ids)].copy()

    inventory_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for _, r in sub.iterrows():
        pt = str(r.get("entity_pair_type") or "").replace("–", "-")
        if not pt or pt.count("-") != 1:
            continue
        inventory_counts[pt] = inventory_counts.get(pt, 0) + 1
        if pt not in PRIMARY_PAIR_TYPES:
            continue
        pub_year = r.get("publication_year")
        rows.append(
            {
                "target_id": f"rank_{int(r['evidence_id'])}",
                "evidence_id": int(r["evidence_id"]),
                "pmid": str(r["pmid"]),
                "pair_type": pt,
                "scope": "primary",
                "head_entity": r["head_entity"],
                "head_type": r["head_type"],
                "tail_entity": r["tail_entity"],
                "tail_type": r["tail_type"],
                "publication_year": int(pub_year) if pd.notna(pub_year) else None,
                "abstract_grounded": True,
                "evaluable_for_ranking": True,
            }
        )

    df = pd.DataFrame(rows)
    n_inventory_grounded = sum(inventory_counts.values())
    n_variant = sum(
        inventory_counts.get(pt, 0)
        for pt in ("variant-drug", "variant-disease")
    )
    print(f"  Abstract-grounded inventory (step 00): {n_inventory_grounded}")
    print(f"  Evaluable ranking targets (gene-drug + gene-disease): {len(df)}")
    print(f"  Variant pairs in inventory (not evaluable): {n_variant}")
    print(f"  Unique PMIDs (evaluable): {df['pmid'].nunique()}")
    for pt, n in df["pair_type"].value_counts().items():
        print(f"    {pt}: {n}")

    meta = {
        "abstract_grounded_inventory_total": n_inventory_grounded,
        "variant_pairs_excluded": n_variant,
        "inventory_by_pair_type": inventory_counts,
    }
    return df, meta
