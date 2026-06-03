"""Assemble frozen ranking evaluation protocol (targets + metric definitions only)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from .config import (
    FETCH_METADATA_JSON,
    FROZEN_PROTOCOL_JSON,
    RANKING_TARGETS_CSV,
    RECALL_K_VALUES,
    REPORT_DIR,
    SAMPLING_SEED,
)
from .targets import build_ranking_targets


def _load_civic_version() -> dict[str, str]:
    meta = json.loads(FETCH_METADATA_JSON.read_text(encoding="utf-8"))
    releases = meta.get("data_releases", [])
    return {
        "fetch_timestamp": meta.get("fetch_timestamp", "unknown"),
        "civic_release": releases[0]["name"] if releases else "unknown",
        "api_endpoint": meta.get("api_endpoint", "https://civicdb.org/api/graphql"),
    }


def build_protocol() -> dict:
    print("\n=== Step 02: ranking evaluation protocol ===")
    targets, inv_meta = build_ranking_targets()
    targets.to_csv(RANKING_TARGETS_CSV, index=False)

    by_pair = targets["pair_type"].value_counts().to_dict()
    protocol = {
        "version": "02_ranking_protocol_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "kb_relation_ranking",
        "civic_version": _load_civic_version(),
        "sampling_seed": SAMPLING_SEED,
        "metrics": {
            "mrr": "Mean Reciprocal Rank — is the top-ranked candidate a CIViC-curated positive?",
            "recall_at_k": f"Recall at k (k={list(RECALL_K_VALUES)}) — curation triage coverage in top-k slots",
            "auc_pr": "Area under the precision-recall curve — overall ranking quality under class imbalance",
        },
        "statistics": {
            "n_evaluable_ranking_targets": len(targets),
            "n_unique_pmids": int(targets["pmid"].nunique()),
            "targets_by_pair_type": by_pair,
            "abstract_grounded_inventory_total": inv_meta["abstract_grounded_inventory_total"],
            "variant_pairs_excluded_from_evaluation": inv_meta["variant_pairs_excluded"],
        },
        "variant_exclusion_note": (
            "Variant pairs appear in the step-00 full inventory but are NOT evaluation targets: "
            "PubTator3 cannot build variant candidate pools (0% coverage). "
            "Only gene-drug and gene-disease abstract-grounded positives are frozen for ranking."
        ),
        "baseline_note": "Trivial ranking baselines are computed on the real frozen pool in step 03.",
        "targets": targets.to_dict(orient="records"),
    }
    FROZEN_PROTOCOL_JSON.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"  Frozen protocol -> {FROZEN_PROTOCOL_JSON}")

    _write_report(protocol)
    return protocol


def _write_report(protocol: dict) -> None:
    stats = protocol["statistics"]
    pair_lines = "\n".join(
        f"| {pt} | {n} |" for pt, n in stats["targets_by_pair_type"].items()
    )
    report = f"""# Ranking Evaluation Protocol (step 02)

Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

## Rationale

CIViC evidence is dominated by positive clinical assertions (~90% SUPPORTS direction). A curation-relevant evaluation is therefore a **ranking/triage** problem: among co-occurring entity pairs in an abstract, can a model rank CIViC-curated positives highly?

## Frozen evaluation target set

**Evaluable set:** **{stats['n_evaluable_ranking_targets']}** abstract-grounded gene–drug and gene–disease positives across **{stats['n_unique_pmids']}** PMIDs.

| Pair type | Evaluable targets |
| --- | ---: |
{pair_lines}

The step-00 inventory contains **{stats['abstract_grounded_inventory_total']}** abstract-grounded pairs in total. The remaining **{stats['variant_pairs_excluded_from_evaluation']}** variant pairs are **not evaluable** — PubTator3 cannot build variant candidate pools (0% coverage in step 03). They are excluded from all ranking evaluation.

## Metric definitions

| Metric | Rationale |
| --- | --- |
| Mean Reciprocal Rank (MRR) | Is the top-ranked candidate the CIViC-curated one? |
| Recall@k (k=1,3,5) | Curation triage coverage — fraction of curated positives in top-k |
| Area under the precision-recall curve (AUC-PR) | Overall ranking under ~15% positive rate |

**No scores are computed in step 02.** Trivial ranking baselines (random, constant, distance ranker) and tie-handling verification are computed on the real frozen candidate pool in step 03.

## Outputs

- `outputs/02_evaluation_protocol/frozen_protocol.json`
- `outputs/02_evaluation_protocol/ranking_targets.csv`
"""
    path = REPORT_DIR / "report.md"
    path.write_text(report, encoding="utf-8")
    print(f"  Report -> {path}")
