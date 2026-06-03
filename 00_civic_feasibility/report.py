"""Generate markdown feasibility report from cached analysis outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from .config import DATA_DIR, OUTPUT_DIR, REPORT_DIR


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / name)


def _pct(n: float) -> str:
    return f"{100 * n:.1f}%"


def _alignment_label(status: str) -> str:
    mapping = {
        "both_present": "Both entities in abstract",
        "head_absent": "One or both entities not in abstract",
        "tail_absent": "One or both entities not in abstract",
        "both_absent": "One or both entities not in abstract",
    }
    return mapping.get(status, status.replace("_", " "))


def generate_report() -> None:
    metadata_path = DATA_DIR / "fetch_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    summary = _read_csv("evaluable_target_summary.csv")
    pair_breakdown = _read_csv("entity_pair_breakdown.csv")
    balance = _read_csv("assertion_balance_summary.csv")
    direction = _read_csv("evidence_direction_counts.csv")
    alignment = _read_csv("abstract_alignment_summary.csv")
    label_cross = _read_csv("label_cross_tab.csv")
    layer = _read_csv("layer_comparison.csv")

    total = int(summary.loc[summary["metric"] == "total_accepted_evidence_items", "count"].iloc[0])
    evaluable = int(summary.loc[summary["metric"] == "evaluable_abstract_two_entity", "count"].iloc[0])
    pubmed_n = int(summary.loc[summary["metric"] == "pubmed_sourced_items", "count"].iloc[0])

    supports_n = int(direction.loc[direction["evidence_direction"] == "SUPPORTS", "count"].sum()) if "SUPPORTS" in direction["evidence_direction"].values else 0
    does_not_support = int(balance.loc[balance["label"] == "direction_does_not_support", "count"].iloc[0])
    strict_positive_share = float(balance.loc[balance["label"] == "strict_positive_share", "count"].iloc[0])

    both_in_abstract = int(alignment.loc[alignment["alignment_status"] == "both_present", "count"].sum())
    alignment_total = int(alignment["count"].sum())
    not_in_abstract = alignment_total - both_in_abstract

    assertions_n = int(layer.loc[layer["layer"] == "assertions_accepted", "count"].iloc[0])
    mean_eid = float(layer.loc[layer["layer"] == "mean_evidence_per_assertion", "count"].iloc[0])
    mean_pmid = float(layer.loc[layer["layer"] == "mean_abstracts_per_assertion", "count"].iloc[0])

    releases = metadata.get("data_releases", [])
    pinned_release = releases[0]["name"] if releases else "unknown"
    fetch_ts = metadata.get("fetch_timestamp", "unknown")

    pair_lines = "\n".join(
        f"| {row.entity_pair_type} | {int(row.count)} | {_pct(row.share_of_evaluable)} |"
        for row in pair_breakdown.itertuples()
    )

    label_lines = "\n".join(
        f"| {row.evidence_type} | {row.clinical_significance} | {row.evidence_direction} | {int(row.count)} |"
        for row in label_cross.head(8).itertuples()
    )

    # Collapse alignment rows to plain wording
    both_row = int(alignment.loc[alignment["alignment_status"] == "both_present", "count"].sum())
    other_rows = alignment[alignment["alignment_status"] != "both_present"]
    other_count = int(other_rows["count"].sum()) if len(other_rows) else 0

    report = f"""# CIViC Feasibility Report (step 00)

Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

## Data provenance

| Field | Value |
| --- | --- |
| API endpoint | `{metadata.get('api_endpoint', 'https://civicdb.org/api/graphql')}` |
| Fetch timestamp (UTC) | {fetch_ts} |
| CIViC release track | `{pinned_release}` |
| Accepted evidence items pulled | {total} |
| Accepted assertions pulled | {assertions_n} |

All data were retrieved live from the CIViC GraphQL API and cached under `projects/project_1/data/00_civic_feasibility/`.

---

## A. Evaluable target inventory

| Metric | Count |
| --- | ---: |
| Accepted evidence items (total) | {total} |
| PubMed-sourced evidence items | {pubmed_n} |
| Evaluable abstract-level two-entity targets | {evaluable} |

### Entity-pair breakdown (evaluable set)

| Entity-pair type | Count | Share of evaluable |
| --- | ---: | ---: |
{pair_lines}

**Interpretation.** CIViC contains **{total}** accepted evidence items, of which **{evaluable}** qualify as PubMed-backed, two-entity targets. The prior study's n=162 is not the database ceiling — it reflects earlier filtering. This inventory defines the full evidence landscape; the **ranking evaluation set** (step 02) uses the abstract-grounded subset below.

---

## B. Positive-assertion dominance (why ranking, not classification)

| Label | Count |
| --- | ---: |
| Evaluable items | {evaluable} |
| `SUPPORTS` direction | {supports_n} |
| `DOES_NOT_SUPPORT` direction | {does_not_support} |
| Share with positive assertion direction | {_pct(strict_positive_share)} |

**Interpretation.** CIViC is dominated by positive clinical assertions ({supports_n}/{evaluable}; {_pct(supports_n / max(evaluable, 1))}). A curation-relevant task is therefore **ranking/triage**: among co-occurring entity pairs in an abstract, can a model rank CIViC-curated positives highly? This study uses ranking from the start — not binary classification with constructed negatives.

---

## C. Abstract-grounded evaluation set definition

| Text-grounded status | Count |
| --- | ---: |
| Both entities in abstract | {both_row} |
| One or both entities not in abstract | {other_count} |

- Abstract-grounded pairs: **{both_in_abstract} / {alignment_total}** ({_pct(both_in_abstract / max(alignment_total, 1))})
- Outside text-grounded task: **{not_in_abstract} / {alignment_total}** ({_pct(not_in_abstract / max(alignment_total, 1))})

**Matching method.** Case-insensitive substring matching on entity strings, with simple surface-form variants (hyphen/space swaps, parenthetical aliases). Abstract text from CIViC Source records when available, with PubMed efetch as fallback.

**Interpretation.** The **abstract-grounded subset** ({both_in_abstract} pairs where both entities appear in the abstract) is the **complete evaluation set definition** for this study. Pairs where an entity does not appear in the abstract are outside the text-grounded task — they are not part of ranking evaluation.

---

## D. Native-label heterogeneity (RQ1 context)

| Evidence type | Clinical significance | Direction | Count |
| --- | --- | --- | ---: |
{label_lines}

**Interpretation.** CIViC native labels (`evidenceType`, `significance`, `evidenceDirection`) span heterogeneous semantic levels. Training-corpus labels (association, mechanism, clinical significance) are not directly commensurable with CIViC clinical assertions — a finding developed further in step 01.

---

## E. Assertion vs evidence layer

| Layer metric | Value |
| --- | ---: |
| Accepted assertions | {assertions_n} |
| Evaluable evidence items | {evaluable} |
| Mean evidence items per assertion | {mean_eid:.2f} |
| Mean unique PMIDs per assertion | {mean_pmid:.2f} |

**Interpretation.** Assertions aggregate multiple evidence items across PMIDs. For **abstract-level ranking**, the **evidence-item layer** (one PubMed abstract per item) is the correct unit.

---

## Design implications

1. **Evaluation unit:** Evidence items linked to a single PubMed abstract.
2. **Sample size:** ~{evaluable} evaluable targets; **{both_in_abstract}** abstract-grounded pairs form the ranking evaluation universe (step 02 freezes the gene–drug and gene–disease subset).
3. **Task framing:** Ranking/triage among co-occurring candidates — not binary classification.
4. **Reproducibility:** Pin methods to fetch timestamp `{fetch_ts}` and release `{pinned_release}`.
"""

    path = REPORT_DIR / "report.md"
    path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {path}")
