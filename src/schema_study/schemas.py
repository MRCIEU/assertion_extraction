"""Schema loading, partition checks, and label rollup.

Display names and stable identifiers stay distinct. No_Relation is never placed
inside a positive group. Every fine label belongs to exactly one group at each
level.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import yaml

LEVELS = ("P4", "P3", "P2", "P1")


@dataclass(frozen=True)
class Label:
    stable_id: str
    display_name: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class Schema:
    schema_id: str
    corpus: str
    no_relation_id: str
    no_relation_name: str
    fine_ids: tuple[str, ...]
    fine_names: dict[str, str]
    directions: dict[str, str]
    levels: dict[str, tuple[Label, ...]]
    raw: dict

    def level_labels(self, level: str) -> tuple[Label, ...]:
        if level not in self.levels:
            raise KeyError(f"Unknown predicate level {level} for {self.schema_id}")
        return self.levels[level]

    def by_display(self, level: str) -> dict[str, Label]:
        return {label.display_name: label for label in self.level_labels(level)}

    def by_id(self, level: str) -> dict[str, Label]:
        return {label.stable_id: label for label in self.level_labels(level)}

    def positive_display_names(self, level: str) -> tuple[str, ...]:
        return tuple(label.display_name for label in self.level_labels(level))

    def label_cardinality(self) -> str:
        return str(self.raw.get("label_cardinality", "single"))


def load_schema(path: Path) -> Schema:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    fine_ids = tuple(item["stable_id"] for item in raw["fine_labels"])
    fine_names = {item["stable_id"]: item["display_name"] for item in raw["fine_labels"]}
    directions = {item["stable_id"]: item["direction"] for item in raw["fine_labels"]}
    levels = {}
    for level, payload in raw["levels"].items():
        levels[level] = tuple(
            Label(
                stable_id=item["stable_id"],
                display_name=item["display_name"],
                members=tuple(item["members"]),
            )
            for item in payload["labels"]
        )
    return Schema(
        schema_id=raw["schema_id"],
        corpus=raw["corpus"],
        no_relation_id=raw["no_relation"]["stable_id"],
        no_relation_name=raw["no_relation"]["display_name"],
        fine_ids=fine_ids,
        fine_names=fine_names,
        directions=directions,
        levels=levels,
        raw=raw,
    )


def default_schema_path(code_root: Path, corpus: str) -> Path:
    name = {"biored": "biored.yaml", "drugprot": "drugprot.yaml"}[corpus]
    return code_root / "configs" / "schemas" / name


def partition_errors(schema: Schema) -> list[str]:
    """Return human-readable violations of the complete exclusive mapping."""
    errors = []
    fine = set(schema.fine_ids)
    if len(schema.fine_ids) != len(fine):
        errors.append("duplicate fine stable ids")
    if schema.no_relation_id in fine:
        errors.append("No_Relation is listed as a fine positive label")
    for level, labels in schema.levels.items():
        seen: list[str] = []
        displays = [label.display_name for label in labels]
        ids = [label.stable_id for label in labels]
        if len(displays) != len(set(displays)):
            errors.append(f"{level} has duplicate display names")
        if len(ids) != len(set(ids)):
            errors.append(f"{level} has duplicate stable ids")
        for label in labels:
            if schema.no_relation_id in label.members or schema.no_relation_name in label.members:
                errors.append(f"{level} group {label.stable_id} contains No_Relation")
            unknown = [member for member in label.members if member not in fine]
            if unknown:
                errors.append(f"{level} group {label.stable_id} has unknown members {unknown}")
            seen.extend(label.members)
        if len(seen) != len(set(seen)):
            errors.append(f"{level} assigns at least one fine label more than once")
        missing = sorted(fine - set(seen))
        if missing:
            errors.append(f"{level} does not cover {missing}")
    return errors


def assert_partition(schema: Schema) -> None:
    errors = partition_errors(schema)
    if errors:
        raise ValueError(f"{schema.schema_id} schema is not a complete partition: {errors}")


def map_fine_ids(schema: Schema, fine_ids: set[str] | list[str], level: str) -> set[str]:
    """Map fine stable ids onto the stable ids of `level`, dropping duplicates."""
    grouped = set()
    wanted = set(fine_ids)
    unknown = wanted - set(schema.fine_ids)
    if unknown:
        raise KeyError(f"Fine labels outside {schema.schema_id}: {sorted(unknown)}")
    for label in schema.level_labels(level):
        if wanted & set(label.members):
            grouped.add(label.stable_id)
    return grouped


def map_group_to_level(schema: Schema, source_level: str, group_id: str, target_level: str) -> set[str]:
    source = schema.by_id(source_level)[group_id]
    return map_fine_ids(schema, set(source.members), target_level)


def rollup_ids(schema: Schema, source_level: str, group_ids: set[str], target_level: str) -> set[str]:
    mapped: set[str] = set()
    for group_id in group_ids:
        mapped |= map_group_to_level(schema, source_level, group_id, target_level)
    return mapped


def composed_equals_direct(schema: Schema) -> bool:
    """P4 -> P3 -> P2 must equal P4 -> P2 for every fine label."""
    for fine_id in schema.fine_ids:
        direct = map_fine_ids(schema, {fine_id}, "P2")
        through_p3 = rollup_ids(schema, "P3", map_fine_ids(schema, {fine_id}, "P3"), "P2")
        if direct != through_p3:
            return False
    return True


def group_direction(schema: Schema, level: str, group_id: str) -> str:
    members = schema.by_id(level)[group_id].members
    polar = {schema.directions[member] for member in members if schema.directions[member] != "nonpolar"}
    if polar == {"positive"}:
        return "positive"
    if polar == {"negative"}:
        return "negative"
    return "nonpolar"


def resolve_output_label(schema: Schema, level: str, text: str) -> tuple[str | None, str | None]:
    """Return `(stable_id, error)`.

    `error` is `no_relation_string` or `illegal` when the text is not a positive
    display name or stable id at this level.
    """
    cleaned = text.strip()
    if cleaned == schema.no_relation_name or cleaned == schema.no_relation_id:
        return None, "no_relation_string"
    by_display = schema.by_display(level)
    by_id = schema.by_id(level)
    if cleaned in by_display:
        return by_display[cleaned].stable_id, None
    if cleaned in by_id:
        return cleaned, None
    return None, "illegal"


def random_merge_versions(fine_ids: list[str], *, seed: int = 42, n_versions: int = 3) -> list[list[set[str]]]:
    """Three unique partitions with group sizes 1, 1, and 6.

    Labels are sorted by stable id before the first shuffle. `random.Random`
    is the only generator. Identical member-set partitions are not repeated.
    """
    labels = sorted(fine_ids)
    if len(labels) != 8:
        raise ValueError("BioRED random-merge controls require the eight fine labels")
    rng = random.Random(seed)
    versions: list[frozenset[frozenset[str]]] = []
    while len(versions) < n_versions:
        order = labels[:]
        rng.shuffle(order)
        groups = frozenset({frozenset(order[:1]), frozenset(order[1:2]), frozenset(order[2:])})
        if groups not in versions:
            versions.append(groups)
    materialized = []
    for groups in versions:
        ordered = sorted((set(group) for group in groups), key=lambda group: (len(group), min(group)))
        materialized.append(ordered)
    return materialized


def control_schema_from_groups(control_id: str, groups: list[dict], fine_schema: Schema) -> Schema:
    """Build a one-level schema whose groups are the supplied member lists."""
    labels = tuple(
        Label(item["stable_id"], item["display_name"], tuple(item["members"]))
        for item in groups
    )
    return Schema(
        schema_id=control_id,
        corpus=fine_schema.corpus,
        no_relation_id=fine_schema.no_relation_id,
        no_relation_name=fine_schema.no_relation_name,
        fine_ids=fine_schema.fine_ids,
        fine_names=fine_schema.fine_names,
        directions=fine_schema.directions,
        levels={"P_control": labels},
        raw={"control_id": control_id, "status": "draft_pending_confirmation"},
    )
