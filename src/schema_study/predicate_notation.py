"""Research predicate levels and the historical experiment names they reuse.

P0 is the binary distinction between an annotated positive relation and
NO_RELATION. It matches historical P1 when the label membership is unchanged.
The negative outcome stays outside the positive hierarchy.
"""

from __future__ import annotations

from pathlib import Path

import yaml

NOTATION_VERSION = "research_predicate_notation_v1"


def load_notation(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw["version"] != NOTATION_VERSION:
        raise ValueError(f"Unexpected predicate notation version: {raw['version']}")
    return raw


def historical_level(notation: dict, research_level: str) -> str:
    for row in notation["correspondence"]:
        if row["research_level"] == research_level:
            return row["historical_level"]
    raise KeyError(research_level)


def research_level(notation: dict, historical: str) -> str:
    for row in notation["correspondence"]:
        if row["historical_level"] == historical:
            return row["research_level"]
    raise KeyError(historical)


def positive_group_count(research_level_name: str, historical_positive_groups: int) -> int:
    """P0 keeps one positive group. Finer research levels keep the historical counts."""
    if research_level_name == "P0":
        return 1
    return historical_positive_groups
