"""Freeze current downstream baseline from existing processed artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .paths import MANIFESTS, PROC, REPORTS, ensure_dirs


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _linkage_level_counts(link_csv: Path) -> Dict[str, int]:
    rows = _read_csv(link_csv)
    if not rows or "linkage_level" not in rows[0]:
        return {}
    return dict(Counter(r["linkage_level"] for r in rows))


def write_baseline_freeze() -> Dict[str, Any]:
    ensure_dirs()
    assert_vol = _read_csv(PROC / "assertion_volume_summary.csv")
    ev_out = _read_csv(PROC / "evidence_outcome_summary.csv")
    exec_log: Dict[str, Any] = {}
    if (MANIFESTS / "execution_log.json").is_file():
        exec_log = json.loads((MANIFESTS / "execution_log.json").read_text(encoding="utf-8"))

    assertion_counts = {r["model_id"]: int(r["raw_assertion_lines"]) for r in assert_vol if r.get("model_id")}

    ev_totals = {r["audit_outcome"]: int(r["count"]) for r in ev_out if "audit_outcome" in r}
    if not ev_totals and ev_out:
        ev_totals = {r.get(list(r.keys())[0], ""): int(r[list(r.keys())[1]]) for r in ev_out}

    linkage_levels = _linkage_level_counts(PROC / "kb_linkage_results.csv")

    freeze: Dict[str, Any] = {
        "freeze_utc_note": "Captured before strengthening enhancement matrix; sources under output data/processed.",
        "scope": "NSCLC precision panel — manifests/scope_definition.json",
        "model_shortlist": ["M015", "M021", "M003", "S002"],
        "extraction_backend": exec_log.get("extraction_backend", "checkpoint"),
        "retrieval_setting": {
            "track_a": "CIViC PMID efetch",
            "track_b": "bounded esearch NSCLC + EGFR/ALK/KRAS retmax=12",
            "max_pmid_fetch": exec_log.get("max_pmid_fetch", 80),
        },
        "proposal_type": "gene_x_drug_comention_per_sentence_plus_gene_disease_lung_proxy",
        "linkage_strictness": "L1_strict_production_rules_in_run_pipeline.link_to_kb",
        "context_type": "abstract_plus_title_sentence_split",
        "assertion_non_negative_counts": assertion_counts,
        "evidence_outcome_totals": ev_totals,
        "linkage_level_counts_estimate": linkage_levels,
        "known_weaknesses": [
            "M015 emitted 0 non-negative assertions on shared gene×drug inventory in first full checkpoint run — conservatism vs proposal space, not placeholder failure.",
            "kb_supported_aligned was 0 under strict linkage + confidence gates — dominated by ambiguity, weak support, KB-absent candidates.",
            "Abstract-only context; no PMC OA requirement in baseline.",
            "Pair inventory gene×drug centric; variant_disease rows rare until model maps VARIANT_GENE.",
        ],
    }

    (MANIFESTS / "current_baseline_freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    md = REPORTS / "current_baseline_freeze.md"
    md.write_text(
        f"""# Current baseline freeze (pre-strengthening matrix)

## Configuration

- **Extraction backend:** `{freeze["extraction_backend"]}`
- **Retrieval:** Track A CIViC PMIDs + bounded Track B esearch; `max_pmid_fetch` = {freeze["retrieval_setting"].get("max_pmid_fetch")}
- **Proposal:** {freeze["proposal_type"]}
- **Linkage:** {freeze["linkage_strictness"]}
- **Context:** {freeze["context_type"]}

## Assertion volume (non-negative retained)

```json
{json.dumps(assertion_counts, indent=2)}
```

## Evidence audit outcomes

```json
{json.dumps(ev_totals, indent=2)}
```

## Known weaknesses (must carry forward)

{chr(10).join("- " + w for w in freeze["known_weaknesses"])}

## Machine-readable

`manifests/current_baseline_freeze.json`

---
""",
        encoding="utf-8",
    )
    return freeze
