"""Load bigbio corpora and build comprehensive inventories."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import datasets
import pandas as pd
from datasets import load_dataset

from .config import CORPORA, DATA_DIR, INVENTORY_FILE, TRAIN_STATS_FILE
from .entity_normalization import civic_pair_type, normalize_entity_type


def _count_split(split) -> dict[str, Any]:
    relation_types: Counter = Counter()
    civic_pairs: Counter = Counter()
    raw_pairs: Counter = Counter()
    entity_types: Counter = Counter()
    unmapped_entities: Counter = Counter()
    relation_by_pair: Counter = Counter()

    for doc in split:
        entity_map = {e["id"]: e["type"] for e in doc.get("entities", [])}
        for entity in doc.get("entities", []):
            entity_types[entity["type"]] += 1
            if normalize_entity_type(entity["type"]) is None:
                unmapped_entities[entity["type"]] += 1

        for relation in doc.get("relations", []):
            rel_type = relation["type"]
            relation_types[rel_type] += 1
            t1 = entity_map.get(relation["arg1_id"])
            t2 = entity_map.get(relation["arg2_id"])
            if not t1 or not t2:
                continue
            raw_pairs[tuple(sorted([t1, t2]))] += 1
            pair = civic_pair_type(t1, t2)
            if pair:
                civic_pairs[pair] += 1
                relation_by_pair[(pair, rel_type)] += 1

    return {
        "documents": len(split),
        "relation_types": dict(relation_types),
        "civic_pair_counts": dict(civic_pairs),
        "raw_pair_counts": {f"{a} | {b}": c for (a, b), c in raw_pairs.items()},
        "entity_types": dict(entity_types),
        "unmapped_entity_types": dict(unmapped_entities),
        "total_relations": sum(relation_types.values()),
        "relation_by_civic_pair": {
            f"{pair}::{rel}": c for (pair, rel), c in relation_by_pair.items()
        },
    }


def _merge_counters(dicts: list[dict]) -> dict:
    merged: Counter = Counter()
    for d in dicts:
        merged.update(d)
    return dict(merged)


def build_inventories(force: bool = False) -> dict[str, Any]:
    """Pull all corpora via bigbio and cache comprehensive stats."""
    if INVENTORY_FILE.exists() and not force:
        return json.loads(INVENTORY_FILE.read_text(encoding="utf-8"))

    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets_library_version": datasets.__version__,
        "corpora": {},
    }
    long_rows: list[dict] = []

    for key, spec in CORPORA.items():
        print(f"Loading {spec['display_name']} ({spec['hf_id']}, {spec['config']})...")
        dataset = load_dataset(spec["hf_id"], spec["config"], trust_remote_code=True)

        per_split = {split: _count_split(dataset[split]) for split in dataset}
        all_rel = _merge_counters([s["relation_types"] for s in per_split.values()])
        all_civic = _merge_counters([s["civic_pair_counts"] for s in per_split.values()])
        all_entities = _merge_counters([s["entity_types"] for s in per_split.values()])

        entry = {
            "corpus_key": key,
            "display_name": spec["display_name"],
            "hf_id": spec["hf_id"],
            "config": spec["config"],
            "language": spec["language"],
            "role": spec["role"],
            "description": spec["description"],
            "load_status": "ok",
            "split_sizes": {s: per_split[s]["documents"] for s in per_split},
            "per_split": per_split,
            "relation_type_counts": all_rel,
            "civic_pair_counts": all_civic,
            "entity_type_counts": all_entities,
            "total_relations": sum(all_rel.values()),
        }
        results["corpora"][key] = entry

        for split_name, stats in per_split.items():
            for table, counter in [
                ("relation_type", stats["relation_types"]),
                ("civic_pair", stats["civic_pair_counts"]),
                ("entity_type", stats["entity_types"]),
            ]:
                for label, count in counter.items():
                    long_rows.append(
                        {
                            "corpus": key,
                            "display_name": spec["display_name"],
                            "split": split_name,
                            "table": table,
                            "label": label,
                            "count": count,
                        }
                    )
            long_rows.append(
                {
                    "corpus": key,
                    "display_name": spec["display_name"],
                    "split": split_name,
                    "table": "documents",
                    "label": "documents",
                    "count": stats["documents"],
                }
            )

        print(
            f"  ok: {entry['total_relations']} relations, "
            f"splits={entry['split_sizes']}, civic_pairs={list(all_civic.keys())}"
        )

    INVENTORY_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(long_rows).to_csv(DATA_DIR / "corpus_inventory_long.csv", index=False)

    # Train-split stats (used by volume + DrugProt mapping).
    train_stats = {
        "generated_at": results["generated_at"],
        "source": "train splits from cached inventories",
        "corpora": {},
    }
    for key, spec in CORPORA.items():
        per_split = results["corpora"][key]["per_split"]
        train_parts = [per_split[s] for s in spec["train_splits"] if s in per_split]
        train_stats["corpora"][key] = {
            "display_name": spec["display_name"],
            "train_splits": spec["train_splits"],
            "relation_type_counts": _merge_counters([p["relation_types"] for p in train_parts]),
            "civic_pair_counts": _merge_counters([p["civic_pair_counts"] for p in train_parts]),
            "total_train_relations": sum(
                sum(p["relation_types"].values()) for p in train_parts
            ),
        }
    TRAIN_STATS_FILE.write_text(json.dumps(train_stats, indent=2), encoding="utf-8")

    return results
