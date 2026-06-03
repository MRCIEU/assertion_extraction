"""Analyses A–E for CIViC feasibility diagnostics."""

from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests

from .config import DATA_DIR, FIGURE_DIR, OUTPUT_DIR
from .matching import check_alignment, summarize_alignment


def _load_inventory() -> pd.DataFrame:
    path = OUTPUT_DIR / "evaluable_inventory.csv"
    if not path.exists():
        raise FileNotFoundError("Run inventory builder first.")
    return pd.read_csv(path)


def _load_evidence_json() -> list[dict]:
    return json.loads((DATA_DIR / "evidence_items.json").read_text(encoding="utf-8"))


def _load_assertions_json() -> list[dict]:
    return json.loads((DATA_DIR / "assertions.json").read_text(encoding="utf-8"))


def _save_table(df: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    return path


def _bar_plot(
    labels: list[str],
    values: list[int | float],
    title: str,
    ylabel: str,
    filename: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.tight_layout()
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# A. True sample ceiling
# ---------------------------------------------------------------------------

def analysis_a_sample_ceiling(inventory: pd.DataFrame | None = None) -> dict:
    inventory = inventory if inventory is not None else _load_inventory()

    total = len(inventory)
    evaluable = inventory[inventory["is_evaluable_target"]]
    evaluable_n = len(evaluable)

    by_pair = (
        evaluable.groupby("entity_pair_type", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    by_pair["share_of_evaluable"] = by_pair["count"] / max(evaluable_n, 1)

    summary = pd.DataFrame(
        [
            {"metric": "total_accepted_evidence_items", "count": total},
            {"metric": "pubmed_sourced_items", "count": int(inventory["is_pubmed_source"].sum())},
            {"metric": "evaluable_abstract_two_entity", "count": evaluable_n},
        ]
    )

    _save_table(summary, "evaluable_target_summary.csv")
    _save_table(by_pair, "entity_pair_breakdown.csv")

    if not by_pair.empty:
        _bar_plot(
            by_pair["entity_pair_type"].astype(str).tolist(),
            by_pair["count"].tolist(),
            "Evaluable targets by entity-pair type",
            "Count",
            "entity_pair_distribution.png",
        )

    result = {
        "total_accepted": total,
        "evaluable_n": evaluable_n,
        "by_pair": by_pair.to_dict(orient="records"),
    }

    print("\n=== A. Sample ceiling ===")
    print(f"  total accepted evidence items: {total}")
    print(f"  evaluable abstract-level two-entity targets: {evaluable_n}")
    print(by_pair.to_string(index=False))

    return result


# ---------------------------------------------------------------------------
# B. Positive/negative balance
# ---------------------------------------------------------------------------

def _is_negative_direction(direction: str | float) -> bool:
    if pd.isna(direction):
        return False
    return str(direction).upper() in {"DOES_NOT_SUPPORT"}


def _is_negative_significance(significance: str | float) -> bool:
    if pd.isna(significance):
        return False
    return str(significance).upper() in {
        "NEGATIVE",
        "RESISTANCE",
        "REDUCED_SENSITIVITY",
        "POOR_OUTCOME",
        "ADVERSE_RESPONSE",
    }


def _is_strict_negative(direction: str | float, significance: str | float) -> bool:
    direction_neg = _is_negative_direction(direction)
    sig_neg = not pd.isna(significance) and str(significance).upper() == "NEGATIVE"
    return direction_neg or sig_neg


def analysis_b_balance(inventory: pd.DataFrame | None = None) -> dict:
    inventory = inventory if inventory is not None else _load_inventory()
    evaluable = inventory[inventory["is_evaluable_target"]].copy()

    direction_counts = evaluable["evidence_direction"].value_counts(dropna=False).reset_index()
    direction_counts.columns = ["evidence_direction", "count"]

    significance_counts = evaluable["clinical_significance"].value_counts(dropna=False).reset_index()
    significance_counts.columns = ["clinical_significance", "count"]

    evaluable["is_negative_direction"] = evaluable["evidence_direction"].map(_is_negative_direction)
    evaluable["is_negative_significance"] = evaluable["clinical_significance"].map(_is_negative_significance)
    evaluable["is_strict_negative"] = [
        _is_strict_negative(d, s) for d, s in zip(evaluable["evidence_direction"], evaluable["clinical_significance"])
    ]
    evaluable["is_broad_negative"] = evaluable["is_negative_direction"] | evaluable["is_negative_significance"]

    n = len(evaluable)
    strict_negative_n = int(evaluable["is_strict_negative"].sum())
    broad_negative_n = int(evaluable["is_broad_negative"].sum())
    supports_n = int((evaluable["evidence_direction"] == "SUPPORTS").sum())
    does_not_support_n = int((evaluable["evidence_direction"] == "DOES_NOT_SUPPORT").sum())
    significance_negative_n = int((evaluable["clinical_significance"] == "NEGATIVE").sum())

    balance = pd.DataFrame(
        [
            {"label": "evaluable_total", "count": n},
            {"label": "direction_supports", "count": supports_n},
            {"label": "direction_does_not_support", "count": does_not_support_n},
            {"label": "significance_negative", "count": significance_negative_n},
            {"label": "strict_negative_any", "count": strict_negative_n},
            {"label": "strict_positive_share", "count": round((n - strict_negative_n) / max(n, 1), 4)},
            {"label": "broad_negative_any", "count": broad_negative_n},
            {"label": "resistance_or_adverse_significance", "count": int(
                evaluable["clinical_significance"].isin(["RESISTANCE", "ADVERSE_RESPONSE", "REDUCED_SENSITIVITY"]).sum()
            )},
        ]
    )

    _save_table(direction_counts, "evidence_direction_counts.csv")
    _save_table(significance_counts, "clinical_significance_counts.csv")
    _save_table(balance, "assertion_balance_summary.csv")

    _bar_plot(
        direction_counts["evidence_direction"].astype(str).tolist(),
        direction_counts["count"].tolist(),
        "Evidence direction (evaluable set)",
        "Count",
        "direction_balance.png",
    )

    result = {
        "evaluable_n": n,
        "strict_negative_n": strict_negative_n,
        "strict_positive_share": (n - strict_negative_n) / max(n, 1),
        "does_not_support_n": does_not_support_n,
        "direction_counts": direction_counts.to_dict(orient="records"),
    }

    print("\n=== B. Positive/negative balance ===")
    print(f"  evaluable items: {n}")
    print(f"  DOES_NOT_SUPPORT: {does_not_support_n}")
    print(f"  strict negative (DOES_NOT_SUPPORT or significance=NEGATIVE): {strict_negative_n}")
    print(f"  strict positive share: {result['strict_positive_share']:.3f}")

    return result


# ---------------------------------------------------------------------------
# C. Text–assertion alignment
# ---------------------------------------------------------------------------

def _fetch_pubmed_abstracts(pmids: list[str], batch_size: int = 100) -> dict[str, str]:
    """Fetch abstracts from PubMed for PMIDs missing in CIViC cache."""
    api_key = os.environ.get("NCBI_API_KEY", "")
    abstracts: dict[str, str] = {}
    unique_pmids = sorted({p for p in pmids if p})

    for start in range(0, len(unique_pmids), batch_size):
        batch = unique_pmids[start : start + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if api_key:
            params["api_key"] = api_key

        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=params,
            timeout=60,
        )
        response.raise_for_status()

        root = ET.fromstring(response.text)
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            abstract_el = article.find(".//Abstract")
            if pmid_el is None or abstract_el is None:
                continue
            parts = ["".join(el.itertext()).strip() for el in abstract_el.findall("AbstractText")]
            text = " ".join(p for p in parts if p)
            if text:
                abstracts[pmid_el.text.strip()] = text

        time.sleep(0.12)

    return abstracts


def _build_abstract_lookup(evidence_records: list[dict]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for item in evidence_records:
        source = item.get("source") or {}
        pmid = source.get("citationId")
        if not pmid:
            continue
        lookup[str(pmid)] = {
            "abstract": source.get("abstract") or "",
            "source_id": source.get("id"),
            "title": source.get("title"),
        }
    return lookup


def analysis_c_alignment(inventory: pd.DataFrame | None = None) -> dict:
    inventory = inventory if inventory is not None else _load_inventory()
    evaluable = inventory[inventory["is_evaluable_target"]].copy()
    evidence_records = _load_evidence_json()
    abstract_lookup = _build_abstract_lookup(evidence_records)

    missing_pmids = [
        str(p)
        for p in evaluable["pmid"].dropna().unique()
        if not abstract_lookup.get(str(p), {}).get("abstract")
    ]

    pubmed_fetched = {}
    if missing_pmids:
        print(f"  fetching {len(missing_pmids)} abstracts from PubMed...")
        pubmed_fetched = _fetch_pubmed_abstracts(missing_pmids)

    rows = []
    for _, row in evaluable.iterrows():
        pmid = str(row["pmid"])
        civic_abs = abstract_lookup.get(pmid, {}).get("abstract", "")
        abstract = civic_abs or pubmed_fetched.get(pmid, "")
        abstract_source = "civic" if civic_abs else ("pubmed" if pmid in pubmed_fetched else "missing")

        alignment = check_alignment(
            abstract,
            str(row["head_entity"]),
            str(row["head_type"]),
            str(row["tail_entity"]),
            str(row["tail_type"]),
        )
        rows.append(
            {
                "evidence_id": row["evidence_id"],
                "pmid": pmid,
                "abstract_source": abstract_source,
                "abstract_length": len(abstract),
                **alignment,
            }
        )

    alignment_df = pd.DataFrame(rows)
    summary = summarize_alignment(rows)
    status_counts = alignment_df["alignment_status"].value_counts().reset_index()
    status_counts.columns = ["alignment_status", "count"]

    _save_table(alignment_df, "alignment_details.csv")
    _save_table(status_counts, "abstract_alignment_summary.csv")

    _bar_plot(
        ["both present", "any missing", "head missing", "tail missing"],
        [
            summary["both_mentioned"],
            summary["any_entity_missing"],
            summary["head_missing"],
            summary["tail_missing"],
        ],
        "Abstract–entity alignment (evaluable set)",
        "Count",
        "alignment_rates.png",
    )

    result = {**summary, "status_counts": status_counts.to_dict(orient="records")}

    print("\n=== C. Text–assertion alignment ===")
    print(f"  checked: {summary['n']}")
    print(f"  both entities mentioned: {summary['both_mentioned']} ({summary['both_mentioned_rate']:.3f})")
    print(f"  one or both absent: {summary['any_entity_missing']} ({summary['any_entity_missing_rate']:.3f})")

    return result


# ---------------------------------------------------------------------------
# D. Native-label usability
# ---------------------------------------------------------------------------

def analysis_d_native_labels(inventory: pd.DataFrame | None = None) -> dict:
    inventory = inventory if inventory is not None else _load_inventory()
    evaluable = inventory[inventory["is_evaluable_target"]]

    tables = {}
    for column, filename in [
        ("evidence_type", "evidence_type_counts.csv"),
        ("clinical_significance", "clinical_significance_label_counts.csv"),
        ("evidence_direction", "evidence_direction_label_counts.csv"),
        ("assertion_direction", "assertion_direction_counts.csv"),
        ("assertion_significance", "assertion_significance_counts.csv"),
    ]:
        counts = evaluable[column].value_counts(dropna=False).reset_index()
        counts.columns = [column, "count"]
        counts["share"] = counts["count"] / max(len(evaluable), 1)
        _save_table(counts, filename)
        tables[column] = counts

    cross = (
        evaluable.groupby(["evidence_type", "clinical_significance", "evidence_direction"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    _save_table(cross, "label_cross_tab.csv")

    result = {
        "unique_evidence_types": int(evaluable["evidence_type"].nunique()),
        "unique_significance": int(evaluable["clinical_significance"].nunique()),
        "unique_directions": int(evaluable["evidence_direction"].nunique()),
        "cross_tab_rows": len(cross),
    }

    print("\n=== D. Native-label usability ===")
    print(f"  evidence types: {result['unique_evidence_types']}")
    print(f"  significance values: {result['unique_significance']}")
    print(f"  evidence directions: {result['unique_directions']}")

    return result


# ---------------------------------------------------------------------------
# E. Assertion vs evidence layer
# ---------------------------------------------------------------------------

def analysis_e_assertion_vs_evidence(
    inventory: pd.DataFrame | None = None,
) -> dict:
    inventory = inventory if inventory is not None else _load_inventory()
    assertions = _load_assertions_json()

    assertion_rows = []
    for assertion in assertions:
        eids = assertion.get("evidenceItems") or []
        pmids = sorted(
            {
                str(e.get("source", {}).get("citationId"))
                for e in eids
                if (e.get("source") or {}).get("citationId")
            }
        )
        assertion_rows.append(
            {
                "assertion_id": assertion.get("id"),
                "evidence_items_count": assertion.get("evidenceItemsCount", len(eids)),
                "linked_evidence_items": len(eids),
                "unique_pubmed_abstracts": len(pmids),
                "assertion_type": assertion.get("assertionType"),
                "assertion_direction": assertion.get("assertionDirection"),
            }
        )

    assertion_df = pd.DataFrame(assertion_rows)
    evaluable = inventory[inventory["is_evaluable_target"]]

    summary = pd.DataFrame(
        [
            {"layer": "evidence_items_accepted", "count": len(inventory)},
            {"layer": "evidence_items_evaluable", "count": len(evaluable)},
            {"layer": "assertions_accepted", "count": len(assertion_df)},
            {
                "layer": "evaluable_with_linked_assertion",
                "count": int(evaluable["assertion_id"].notna().sum()),
            },
            {
                "layer": "mean_evidence_per_assertion",
                "count": round(assertion_df["linked_evidence_items"].mean(), 2),
            },
            {
                "layer": "median_evidence_per_assertion",
                "count": float(assertion_df["linked_evidence_items"].median()),
            },
            {
                "layer": "mean_abstracts_per_assertion",
                "count": round(assertion_df["unique_pubmed_abstracts"].mean(), 2),
            },
        ]
    )

    eid_per_assertion = (
        assertion_df["linked_evidence_items"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    eid_per_assertion.columns = ["evidence_items_per_assertion", "assertion_count"]

    _save_table(summary, "layer_comparison.csv")
    _save_table(assertion_df, "assertion_details.csv")
    _save_table(eid_per_assertion, "evidence_per_assertion.csv")

    result = {
        "assertions_n": len(assertion_df),
        "evaluable_n": len(evaluable),
        "mean_evidence_per_assertion": float(assertion_df["linked_evidence_items"].mean()),
        "mean_abstracts_per_assertion": float(assertion_df["unique_pubmed_abstracts"].mean()),
    }

    print("\n=== E. Assertion vs evidence layer ===")
    print(f"  accepted assertions: {result['assertions_n']}")
    print(f"  evaluable evidence items: {result['evaluable_n']}")
    print(f"  mean evidence items per assertion: {result['mean_evidence_per_assertion']:.2f}")
    print(f"  mean unique PMIDs per assertion: {result['mean_abstracts_per_assertion']:.2f}")

    return result


def run_all_analyses() -> dict:
    inventory = _load_inventory()
    return {
        "A": analysis_a_sample_ceiling(inventory),
        "B": analysis_b_balance(inventory),
        "C": analysis_c_alignment(inventory),
        "D": analysis_d_native_labels(inventory),
        "E": analysis_e_assertion_vs_evidence(inventory),
    }


if __name__ == "__main__":
    run_all_analyses()
