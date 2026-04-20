"""Assemble upper_bound_and_bottleneck_report.md and summary + integration note."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .paths import MANIFESTS, PROC, REPORTS, OUT_ROOT, ensure_dirs


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.is_file() else f"_(missing {p.name})_\n"


def compose_reports() -> None:
    ensure_dirs()
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    bfree = _read_text(MANIFESTS / "current_baseline_freeze.json")
    gold = _read_text(MANIFESTS / "goldlite_audit_summary.json")
    bot = _read_text(MANIFESTS / "bottleneck_summary.json")

    sections: List[str] = [
        "# Upper-bound and bottleneck analysis for knowledge-grounded oncology evidence audit\n",
        f"\n*Generated {utc} — output root: `{OUT_ROOT}`*\n",
        "\n## 1. Why the first downstream pass was insufficient\n\n",
        "The initial checkpoint-backed audit run surfaced **valid pipeline behavior** but **weak downstream utility signals**: "
        "**M015** produced **zero** retained non-negative assertions on the shared gene×drug inventory; "
        "**kb_supported_aligned** stayed **0** under strict linkage; ambiguity, weak support, and KB-absent candidate buckets dominated. "
        "Those facts are **not** hidden here — they motivate a **layered** bottleneck and ceiling decomposition rather than a single headline score.\n",
        "\n## 2. Experimental goals\n\n",
        "- **RQ-KA1:** localize loss to retrieval, localization, proposal, classification, or linkage.\n",
        "- **RQ-KA2:** estimate realistic ceilings under oracle pair/sentence and relaxed linkage (bounded).\n",
        "- **RQ-KA3:** map models to **operating profiles** (conservative, surfacing, variant-centric, default, oracle-best).\n",
        "\n## 3. Gold-lite audit set\n\n",
        "```json\n" + (gold if gold.startswith("{") else "{}") + "\n```\n",
        "Detail: `reports/goldlite_audit_construction.md`.\n",
        "\n## 4. Retrieval experiments\n\n",
        _read_text(REPORTS / "retrieval_variant_analysis.md"),
        "\n## 5. Context experiments\n\n",
        _read_text(REPORTS / "context_variant_analysis.md"),
        "\n## 6. Proposal-space experiments\n\n",
        _read_text(REPORTS / "proposal_variant_analysis.md"),
        "\n## 7. Oracle upper-bound experiments\n\n",
        _read_text(REPORTS / "oracle_upper_bound_analysis.md"),
        "\n## 8. Linkage sensitivity\n\n",
        _read_text(REPORTS / "linkage_sensitivity_analysis.md"),
        "\n## 9. Model × setting matrix\n\n",
        "See `reports/tables/model_setting_matrix_results.csv` and `model_operating_profile_analysis.md`.\n",
        _read_text(REPORTS / "model_operating_profile_analysis.md"),
        "\n## 10. Bottleneck attribution\n\n",
        "```json\n" + (bot if bot.startswith("{") else "{}") + "\n```\n",
        _read_text(REPORTS / "bottleneck_attribution_analysis.md"),
        "\n## 11. Final operating profiles\n\n",
        _read_text(REPORTS / "final_downstream_operating_profiles.md"),
        "\n## 12. What this says about model potential\n\n",
        "Compare **oracle O3** macro-F1 (pair+sentence) to **C1_abstract** rows in `context_variant_results.csv`. "
        "A large gap implicates **context/proposal** more than raw checkpoint capacity; flat ceilings under O3 suggest **schema / supervision / label mismatch** for oncology audit phrasing.\n",
        "\n## 13. What remains unresolved\n\n",
        "- No independent expert gold — heuristic S2 remains a **proxy**.\n",
        "- R2 retrieval is **not** a second live crawl.\n",
        "- PMC / full text not required in this pass.\n",
        "\n## 14. Implications for the overall project\n\n",
        "The project gains a **table-driven downstream diagnosis**: default benchmark line (M015) may be **conservative** in this audit formulation while other checkpoints surface more candidates — "
        "policy should **route by operating profile**, not assume one winner on macro-F1 alone.\n",
        "\n## 15. Recommended next step\n\n",
        "1. Optional **live** expanded PubMed retrieval for R2/R4 with rate limits.\n",
        "2. **Stratified manual review** on a subset of gold-lite targets to replace heuristic labels.\n",
        "3. If ceilings stay low, consider **relation-schema audit alignment** (S2_current vs oncology sentence semantics).\n",
        "\n## Appendix: baseline freeze snapshot\n\n",
        "```json\n" + (bfree if bfree.startswith("{") else "{}") + "\n```\n",
    ]

    mainp = REPORTS / "upper_bound_and_bottleneck_report.md"
    mainp.write_text("".join(sections), encoding="utf-8")

    summ = REPORTS / "upper_bound_and_bottleneck_summary.md"
    summ.write_text(
        f"""# Upper-bound & bottleneck pass — compact summary

**Generated:** {utc}

## What was added

- Gold-lite slice + retrieval/context/proposal/oracle/linkage **tables** under `reports/tables/`.
- **Bottleneck attribution** (`bottleneck_attribution_table.csv`) — descriptive ablation shares.
- **Operating profiles** (`final_downstream_operating_profiles.json`).

## Read first

1. `reports/upper_bound_and_bottleneck_report.md`
2. `manifests/bottleneck_summary.json`
3. `data/processed/final_downstream_operating_profiles.json`

## Honesty

No clinical discovery claims; KB-absent objects remain **candidates** only.

---
""",
        encoding="utf-8",
    )

    integ = OUT_ROOT / "integration_note_upper_bound_pass.md"
    integ.write_text(
        f"""# Integration note — upper-bound & bottleneck pass

## For the master report author

- **What changed vs first downstream pass:** added **quantified** decomposition (retrieval proxies, context windows, proposal recall, oracle ceilings, linkage relaxation) with **machine-readable tables**.
- **Oncology-facing strength:** positions the subproject as **audit diagnostics** — where evidence auditing fails under bounded KB + PubMed — not as biomarker discovery.
- **Default model policy:** **M015** remains the **benchmark default**, but downstream audit may route **conservative support** to the **lowest-volume** model on C1 and **surfacing** to **higher-volume** lines — see `final_downstream_operating_profiles.json`.
- **M015 zero-assertion reinterpretation:** with oracle pair+sentence, if M015 **non-zero** macro-F1 appears, the original **0** is primarily **proposal × threshold × context**, not proof the checkpoint is unloaded. If M015 stays lowest under O3 as well, note **classification conservatism** remains plausible.

## Paths

- Canonical: `{OUT_ROOT}/`
- Code: `project_1/knowledge_grounded_evidence_audit/`

---
*Generated {utc}*
""",
        encoding="utf-8",
    )


def mirror_reports_to_code_reports() -> None:
    """Copy select artifacts to project_1/reports/knowledge_grounded_evidence_audit/."""
    kg = Path(__file__).resolve().parent.parent.parent
    mirror = kg.parent / "reports" / "knowledge_grounded_evidence_audit"
    mirror.mkdir(parents=True, exist_ok=True)
    for name in (
        "upper_bound_and_bottleneck_report.md",
        "upper_bound_and_bottleneck_summary.md",
    ):
        src = REPORTS / name
        if src.is_file():
            (mirror / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    integ_src = OUT_ROOT / "integration_note_upper_bound_pass.md"
    if integ_src.is_file():
        (mirror / "integration_note_upper_bound_pass.md").write_text(
            integ_src.read_text(encoding="utf-8"), encoding="utf-8"
        )
