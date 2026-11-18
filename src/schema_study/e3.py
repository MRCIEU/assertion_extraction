"""Checks for a future entity-subtype feasibility table.

This module does not decide whether a mention belongs to a subtype.
"""

from __future__ import annotations

from schema_study.records import write_tsv
from schema_study.validation import ValidationReport

FORBIDDEN_RULES = (
    "pharmacologic_action_implies_drug",
    "gene_database_class_implies_molecular_form",
    "variant_id_format_implies_full_variant_type",
)
COLUMNS = (
    "source_record",
    "mention_id",
    "original_type",
    "proposed_subtype",
    "rule_id",
    "resource_name",
    "resource_version",
    "support_count",
    "conflict",
    "missing",
)


def check_mapping_rows(rows: list[dict]) -> ValidationReport:
    report = ValidationReport(name="e3_mapping")
    by_source: dict[str, set[str]] = {}
    for row in rows:
        rule = row.get("rule_id", "")
        if rule in FORBIDDEN_RULES:
            report.add("error", "forbidden_rule", f"Rule {rule} is not allowed.", source_record=row.get("source_record"))
        subtype = row.get("proposed_subtype", "")
        by_source.setdefault(row.get("source_record", ""), set()).add(subtype)
        if row.get("missing"):
            report.add("warning", "missing_mapping", "Source record has no subtype proposal.")
    for source, subtypes in by_source.items():
        if len({item for item in subtypes if item}) > 1:
            report.add("error", "mapping_conflict", f"{source} has more than one proposed subtype.")
    report.counts["n_rows"] = len(rows)
    report.counts["n_sources"] = len(by_source)
    return report


def write_empty_template(path) -> None:
    write_tsv(path, [], COLUMNS)
