"""Alternative partitions of the native predicate labels.

These partitions are recorded comparisons. They do not replace P1–P4, and
they are not chosen by extraction performance.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from schema_study.evaluation import information_content, smoothed_fine_probabilities
from schema_study.granularity import nats_to_bits, shannon_entropy_nats
from schema_study.schemas import Schema, random_merge_versions


def load_alternatives(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return list(raw["alternatives"])


def partition_errors(schema: Schema, groups: list[dict]) -> list[str]:
    """A usable alternative uses every native label once."""
    seen = []
    errors = []
    for group in groups:
        members = list(group["members"])
        if len(members) != len(set(members)):
            errors.append(f"{group['display_name']} repeats a member")
        seen.extend(members)
    if len(seen) != len(set(seen)):
        errors.append("A native label belongs to more than one alternative group")
    missing = [fine_id for fine_id in schema.fine_ids if fine_id not in set(seen)]
    unknown = [fine_id for fine_id in seen if fine_id not in schema.fine_ids]
    if missing:
        errors.append(f"Missing native labels: {missing}")
    if unknown:
        errors.append(f"Unknown labels: {unknown}")
    return errors


def partition_information(schema: Schema, counts: dict[str, int], groups: list[dict]) -> dict:
    """Entropy of one partition under the study's add-one native distribution."""
    errors = partition_errors(schema, groups)
    if errors:
        raise ValueError("; ".join(errors))
    probabilities = smoothed_fine_probabilities(counts, schema.fine_ids)
    native_entropy = shannon_entropy_nats(probabilities.values())
    grouped = []
    changed = 0
    unchanged = 0
    for group in groups:
        members = tuple(group["members"])
        probability = sum(probabilities[member] for member in members)
        count = sum(int(counts.get(member, 0)) for member in members)
        if len(members) == 1:
            unchanged += count
        else:
            changed += count
        grouped.append(
            {
                "display_name": group["display_name"],
                "members": [schema.fine_names[member] for member in members],
                "n_native_members": len(members),
                "reference_count": count,
                "probability": probability,
                "information_content_bits": nats_to_bits(information_content(probability)),
            }
        )
    entropy = shannon_entropy_nats(item["probability"] for item in grouped)
    observed = changed + unchanged
    return {
        "n_categories": len(grouped),
        "entropy_bits": nats_to_bits(entropy),
        "retained_fraction": None if native_entropy == 0.0 else entropy / native_entropy,
        "n_relations_whose_group_is_broader": changed,
        "n_relations_unchanged": unchanged,
        "proportion_broader": None if observed == 0 else changed / observed,
        "categories": grouped,
    }


def random_control_description(schema: Schema) -> dict:
    """The existing BioRED size-matched control. It is not a biological schema."""
    if schema.schema_id != "biored":
        return {
            "applies": False,
            "reason": "The existing random-merge control is defined for the eight BioRED labels and group sizes 1, 1, and 6. DrugProt P2 does not have that size pattern.",
        }
    versions = random_merge_versions(list(schema.fine_ids), seed=42, n_versions=3)
    return {
        "applies": True,
        "seed": 42,
        "n_versions": len(versions),
        "group_sizes": [1, 1, 6],
        "matches_biored_p2_sizes": True,
        "matches_label_frequencies": False,
        "biological_names": False,
        "versions": [[sorted(group) for group in version] for version in versions],
    }
