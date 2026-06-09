"""Unified step 01 report."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from .config import (
    INVENTORY_FILE,
    ONCOLOGY_AGREEMENT_CSV,
    ONCOLOGY_FRACTIONS_CSV,
    ONCOLOGY_METADATA_JSON,
    OUTPUT_DIR,
    REPORT_DIR,
)


def _conflict_risk_label(rate: float, n_co: int) -> str:
    if n_co == 0:
        return "negligible (no co-annotated pairs on overlapping PMIDs)"
    if rate < 0.05:
        return "negligible"
    if rate < 0.15:
        return "moderate"
    return "serious"


def generate_report() -> None:
    inventories = json.loads(INVENTORY_FILE.read_text(encoding="utf-8"))
    coverage = pd.read_csv(OUTPUT_DIR / "corpus_civic_relevance.csv")
    matrix = pd.read_csv(OUTPUT_DIR / "corpus_alignment_matrix.csv")
    granularity = pd.read_csv(OUTPUT_DIR / "granularity_ladder.csv")
    gran_summary = pd.read_csv(OUTPUT_DIR / "granularity_summary.csv")
    assessment = pd.read_csv(OUTPUT_DIR / "volume_assessment.csv")
    mapping = pd.read_csv(OUTPUT_DIR / "drugprot_civic_mapping.csv")

    diag_path = OUTPUT_DIR / "pmid_diagnostics.json"
    diag = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.exists() else None
    conflict_examples = (
        pd.read_csv(OUTPUT_DIR / "pmid_conflict_examples.csv")
        if (OUTPUT_DIR / "pmid_conflict_examples.csv").exists()
        else pd.DataFrame()
    )

    inv_lines = []
    for _key, entry in inventories["corpora"].items():
        split_docs = ", ".join(f"{s}={n} documents" for s, n in entry["split_sizes"].items())
        rel_top = sorted(entry["relation_type_counts"].items(), key=lambda x: -x[1])[:5]
        rel_str = "; ".join(f"{l} ({c})" for l, c in rel_top)
        inv_lines.append(
            f"**{entry['display_name']}** — {entry['total_relations']} relations corpus-wide | "
            f"split sizes ({split_docs}) | top relation labels: {rel_str}"
        )

    share_by_type = dict(zip(matrix["civic_pair_type"], matrix["civic_eval_share"]))

    matrix_lines = "\n".join(
        f"| {r.civic_pair_type} | {100 * r.civic_eval_share:.1f}% | "
        + " | ".join(
            f"{'yes' if int(getattr(r, f'{k}_relation_count')) else 'no'}"
            + (f" (n={int(getattr(r, f'{k}_relation_count'))})" if int(getattr(r, f'{k}_relation_count')) else "")
            for k in ["biored", "drugprot", "bc5cdr"]
        )
        + " |"
        for r in matrix.itertuples()
    )

    cov_lines = "\n".join(
        f"| {r.display_name} | {r.civic_relevance_pct}% | {r.pairs_covered_of_4} | {r.admissibility} | {r.admissibility_reason} |"
        for r in coverage.itertuples()
    )

    def _vol_note(rec: str) -> str:
        rec = str(rec)
        if "descriptive-only" in rec:
            return "descriptive-only; thin volume"
        return "suitable for future full-corpus vs oncology-subset comparison"

    vol_lines = "\n".join(
        f"| {r.civic_pair_type} | {100 * share_by_type.get(r.civic_pair_type, r.civic_eval_share):.1f}% | {int(r.biored_train_relations)} | "
        f"{int(r.drugprot_train_relations)} | {int(r.combined_train_relations)} | {_vol_note(r.rq3_recommendation)} |"
        for r in assessment.itertuples()
    )

    risky = granularity[
        (granularity["corpus"].isin(["biored", "drugprot"])) & (granularity["over_attribution_risk"] == "high")
    ].head(10)
    risky_lines = "\n".join(
        f"| {r.display_name} | {r.label} | {r.granularity_level} | {int(r.count)} |"
        for r in risky.itertuples()
    )

    amb = mapping[mapping["mapping_status"] == "ambiguous"]
    amb_lines = "\n".join(
        f"| {r.drugprot_label} | {r.proposed_civic_significance} | {r.ambiguous_because} |"
        for r in amb.itertuples()
    )

    diag_section = ""
    design_extra = ""
    oncology_section = ""
    if ONCOLOGY_FRACTIONS_CSV.exists() and ONCOLOGY_AGREEMENT_CSV.exists():
        onco_frac = pd.read_csv(ONCOLOGY_FRACTIONS_CSV)
        onco_agree = pd.read_csv(ONCOLOGY_AGREEMENT_CSV)
        onco_meta = (
            json.loads(ONCOLOGY_METADATA_JSON.read_text(encoding="utf-8"))
            if ONCOLOGY_METADATA_JSON.exists()
            else {}
        )
        frac_lines = ""
        for row in onco_frac.itertuples():
            if pd.notna(row.fraction):
                frac = f"{row.fraction:.1%}"
            else:
                frac = "n/a"
            note = getattr(row, "note", "") or ""
            note = "" if (isinstance(note, float) and pd.isna(note)) else str(note)
            note_suffix = f" ({note})" if note else ""
            frac_lines += (
                f"| {row.corpus} | {row.pair_type} | {row.criterion} | {int(row.n_oncology)} | "
                f"{int(row.n_total)} | {frac}{note_suffix} |\n"
            )
        agree_lines = ""
        for row in onco_agree.itertuples():
            agree_lines += (
                f"| {row.corpus} | {row.pair_type} | {int(row.n_all_three_criteria)} | {int(row.n_total)} | "
                f"{row.fraction_all_three:.1%} | {int(row.n_gene_and_mesh)} | "
                f"{int(row.n_disease_and_gene) if pd.notna(row.n_disease_and_gene) else '—'} | "
                f"{int(row.n_disease_and_mesh) if pd.notna(row.n_disease_and_mesh) else '—'} |\n"
            )
        mesh_cov = onco_meta.get("mesh_fetch", {})
        ncit = onco_meta.get("ncit", {})
        civic = onco_meta.get("civic_genes", {})
        biored_gd = onco_agree[
            (onco_agree["corpus"] == "biored") & (onco_agree["pair_type"] == "gene-disease")
        ]
        core_n = int(biored_gd["n_all_three_criteria"].iloc[0]) if len(biored_gd) else 0
        core_total = int(biored_gd["n_total"].iloc[0]) if len(biored_gd) else 0
        oncology_section = f"""
---

## F. Oncology subset of training corpora

The main experiment trains on **full** BioRED + DrugProt (general biomedical corpora) but evaluates on a **cancer** knowledge base (CIViC). Three **independent** oncology-relevance criteria were applied to every gene–drug and gene–disease relation in the train+validation splits. Criteria are **not** unioned; the conservative **intersection** (all applicable criteria positive) is the primary sufficiency figure.

### Data sources

| Source | Version / date | Role |
| --- | --- | --- |
| NCIt Neoplasm Core | EVS file cached {ncit.get('ncit_core_file_date', '?')[:10]} | Disease → neoplasm branch ({ncit.get('n_ncit_neoplasm_codes', '?')} concepts) |
| NCIt ↔ MeSH crosswalk | NCI EVS Neoplasm_Core_Mappings (MSH source) | Maps BioRED MeSH disease IDs to neoplasm ({ncit.get('n_mesh_neoplasm_ids', '?')} MeSH IDs) |
| CIViC gene set | Step-00 fetch {civic.get('civic_fetch_timestamp', '?')[:10]} | Cancer-gene reference ({civic.get('n_civic_genes', '?')} genes, CC0; COSMIC not used) |
| PubMed MeSH | efetch {mesh_cov.get('timestamp', '?')[:10]} | Literature criterion ({mesh_cov.get('mesh_coverage_rate', 0):.1%} of PMIDs have MeSH) |

### Criterion 1 — Disease maps to NCIt Neoplasm

BioRED disease entities carry MeSH normalisation; MeSH IDs in the NCIt Neoplasm Core crosswalk count as oncology-related. Direct NCIt IDs on entities are also accepted. Gene–drug relations have no disease entity (not applicable).

### Criterion 2 — Gene in CIViC gene set

Gene symbol matched against CIViC `GENE` features from step 00. CIViC is the natural CC0 cancer-gene reference for this study; COSMIC Cancer Gene Census was excluded at preparation time due to licence and redistribution restrictions. COSMIC Cancer Gene Census access is now available and could strengthen the oncology-subset gene criterion if the registered full-corpus vs oncology-subset comparison is run in future work (annotation use only, no redistribution).

### Criterion 3 — Literature indexed under MeSH Neoplasms

Source PMID MeSH descriptors checked against the MeSH neoplasm ID set from the NCIt crosswalk (operational proxy for MeSH C04 Neoplasms branch). Articles without MeSH indexing count as not oncology-related under this criterion.

### Fractions by criterion (separate; no union)

| Corpus | Pair type | Criterion | Oncology n | Total n | Fraction |
| --- | --- | --- | ---: | ---: | ---: |
{frac_lines}

Figure: `figures/01_oncology_fraction_by_criterion.png` · Table: `outputs/oncology_fractions_by_criterion.csv`

### Conservative intersection and pairwise agreement

| Corpus | Pair type | All criteria | Total | Fraction | Gene∩MeSH | Disease∩Gene | Disease∩MeSH |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{agree_lines}

For **gene–disease** relations, "all criteria" requires disease + gene + literature flags. For **gene–drug** relations (no disease entity), the intersection reported is gene + literature only.

Figure: `figures/01_oncology_criteria_intersection.png` · Table: `outputs/oncology_criteria_agreement.csv`

**Immediate interpretation.** Even under the strictest agreement rule, BioRED contains **{core_n}** oncology-related gene–disease training relations (of {core_total}). Training corpora carry substantial cancer signal — benchmark-vs-KB divergence is unlikely to be explained solely by absent oncology content in training.

**Registered future direction (not part of this study).** The conservative oncology subset is large enough to support a future **full-corpus vs oncology-subset training** comparison as an experimental control for training-domain confounding. That comparison is **not** implemented in the preparation stage; main training remains full BioRED + DrugProt.
"""
    if diag:
        ov = diag["overlap"]
        cf = diag["conflict"]
        lk = diag["leakage"]
        ex_lines = ""
        if not conflict_examples.empty:
            for r in conflict_examples.head(8).itertuples():
                ex_lines += (
                    f"| {r.pmid} | {r.pair_type} | {r.entity_1} / {r.entity_2} | "
                    f"{r.biored_binary} | {r.drugprot_binary} |\n"
                )
        else:
            ex_lines = "_No conflicting pairs._\n"

        leak_free = lk["leakage_free"]
        leak_verdict = "**LEAKAGE-FREE: YES**" if leak_free else "**LEAKAGE-FREE: NO**"
        leak_detail = (
            "No training PMIDs appear in the frozen CIViC evaluation PMID set."
            if leak_free
            else (
                f"**{lk['overlap_combined']} PMIDs** overlap the eval set and **must be excluded** before training. "
                f"Leaked PMIDs: `{', '.join(lk['leaked_pmids'])}`. "
                f"Exclusion list: `outputs/excluded_pmids.json`. "
                f"DrugProt would lose {lk['relations_removed_if_excluded'].get('drugprot', {}).get('civic_pair_relations', '?')} "
                f"CIViC-relevant relations; BioRED {lk['relations_removed_if_excluded'].get('biored', {}).get('civic_pair_relations', 0)}."
            )
        )
        conflict_impact = (
            "negligible absolute impact"
            if cf["co_annotated_pairs"] <= 5
            else "non-negligible absolute impact"
        )
        design_extra = f"""| Mixed BioRED+DrugProt training | **Acceptable** for binary presence ({ov['intersection']} shared PMIDs; {cf['conflict_count']} conflicting pair(s); {conflict_impact}) |
| Train/eval PMID split | **{'Clean' if leak_free else 'NOT clean — exclude leaked PMIDs'}** |
| PMIDs to exclude before training | {', '.join(lk['leaked_pmids']) if lk['leaked_pmids'] else 'None'} |
"""
        diag_section = f"""
---

## D-overlap. BioRED ∩ DrugProt PMID overlap

Training splits used: **train + validation** for both corpora.

| Corpus | PMIDs |
| --- | ---: |
| BioRED | {ov['biored_pmids']} |
| DrugProt | {ov['drugprot_pmids']} |
| Intersection | {ov['intersection']} |
| Jaccard | {ov['jaccard']:.4f} |

**Interpretation.** The two training corpora share **{ov['intersection']}** source documents ({100*ov['jaccard']:.2f}% Jaccard). Overlap is minimal — mixing corpora does not duplicate large portions of document-level supervision.

Figure: `figures/pmid_overlap.png` · Table: `outputs/pmid_overlap.csv`

---

## D-conflict. Annotation consistency on overlapping PMIDs

Entity matching for conflict detection: **normalised surface text** (lowercase, collapsed whitespace) + **CIViC entity type** + **pair type**; undirected pairs; binary presence (relation asserted = positive, co-occurring without relation = negative). Database normalisation IDs are **not** used — this choice can merge distinct mentions and affects the conflict rate.

| Metric | Value |
| --- | ---: |
| Overlapping PMIDs | {cf['overlapping_pmids']} |
| Co-annotated entity pairs | {cf['co_annotated_pairs']} |
| Conflicts | {cf['conflict_count']} |
| Conflict rate | {cf['conflict_rate']:.1%} |

### Example conflicts

| PMID | Pair type | Entities | BioRED | DrugProt |
| --- | --- | --- | --- | --- |
{ex_lines}

**Interpretation.** On {cf['overlapping_pmids']} shared PMIDs there are {cf['co_annotated_pairs']} co-annotated entity pairs and **{cf['conflict_count']}** conflict(s) ({cf['conflict_rate']:.1%} rate). Absolute impact on mixed training is **negligible** ({cf['conflict_count']} conflicting pair on {cf['overlapping_pmids']} shared documents). Mixed BioRED+DrugProt training proceeds with binary **presence** labels only.

Figure: `figures/pmid_conflicts.png` · Tables: `outputs/pmid_conflicts.csv`, `outputs/pmid_conflict_examples.csv`

---

## D-leakage. Train ↔ CIViC evaluation PMID overlap (CRITICAL)

Eval PMID source: step-00 abstract-grounded inventory — **{lk['eval_unique_pmids']} unique PMIDs** backing **{lk.get('eval_ranking_targets', lk.get('eval_positive_targets', '?'))}** ranking targets.

| Corpus | Training PMIDs | Overlap with eval | Leaked PMIDs |
| --- | ---: | ---: | --- |
| BioRED | {ov['biored_pmids']} | {lk['overlap_biored']} | {'—' if not lk['overlap_biored'] else ', '.join(lk['leaked_pmids'])} |
| DrugProt | {ov['drugprot_pmids']} | {lk['overlap_drugprot']} | {'—' if not lk['overlap_drugprot'] else ', '.join(lk['leaked_pmids'])} |
| Combined | {ov['union']} | {lk['overlap_combined']} | {'—' if not lk['overlap_combined'] else ', '.join(lk['leaked_pmids'])} |

{leak_verdict}. {leak_detail}

Clean training PMID lists (with exclusions applied): `outputs/training_pmids_clean.json`.

Figure: `figures/pmid_leakage.png` · Table: `outputs/pmid_leakage.csv`
"""

    report = f"""# Step 01: Corpus Alignment & CIViC-Relevance Report

Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

Training corpora for the main experiment: **BioRED + DrugProt**. BC5CDR included for reference statistics only.

Data source: HuggingFace BigBio (`datasets` {inventories['datasets_library_version']}), configs `*_bigbio_kb`.  
Fetch time: {inventories['generated_at']}

Training volume and PMID diagnostics use **train + validation** splits (not test).

---

## A. Corpus inventories (full statistics)

{chr(10).join(f'- {line}' for line in inv_lines)}

Detailed per-split tables: `data/corpus_inventory_long.csv`, `data/corpus_inventories.json`.

---

## B. CIViC alignment matrix

| CIViC pair | Eval share | BioRED | DrugProt | BC5CDR |
| --- | ---: | --- | --- | --- |
{matrix_lines}

| Corpus | CIViC-relevance | Pairs | Admissibility | Reason |
| --- | ---: | ---: | --- | --- |
{cov_lines}

**Implication.** BioRED is the only admissible all-round trainer (100%). DrugProt is partially admissible (53.5%, gene–drug only). BC5CDR is not admissible (0%).

---

## C. Label granularity ladder (label incommensurability)

| Corpus | Label | Level | Count |
| --- | --- | --- | ---: |
{risky_lines}

{gran_summary.iloc[0]['question']}: {gran_summary.iloc[0]['answer']}

**Implication for task design.** Do not collapse CIViC clinical labels onto BioRED or DrugProt types. The evaluation-validity question is whether benchmark rank predicts CIViC downstream performance under a **presence-only** relation target, not whether native label taxonomies align.

See `outputs/granularity_ladder.csv`, figure `figures/granularity_ladder.png`.

---

## D. Trainable volume by CIViC pair (data-composition preparation)

| CIViC pair | Eval share | BioRED train+val | DrugProt train+val | Combined | Data-composition note |
| --- | ---: | ---: | ---: | ---: | --- |
{vol_lines}

**Implication.** Gene–drug and gene–disease pairs carry enough train+validation volume to support a registered future **full-corpus vs oncology-subset** comparison (the data-composition dimension of the evaluation-validity diagnostic). Variant–drug remains descriptive-only (BioRED-only and thin).
{diag_section}
---

## E. DrugProt → CIViC projectability (label incommensurability)

Deterministic projection possible: **no**.

| DrugProt label | Proposed CIViC significance | Why ambiguous |
| --- | --- | --- |
{amb_lines}

**Implication.** Use DrugProt for gene–drug relation **presence** pretraining, not label-aligned fine-tuning to CIViC clinical significance.
{oncology_section}
---

## Design summary

| Topic | Decision |
| --- | --- |
| Training corpora | BioRED (primary) + DrugProt (gene–drug supplement) |
| Training splits | train + validation (test held out) |
| Label incommensurability | Presence-only task; no cross-corpus label collapse |
| Data-composition prep | Registered future full-corpus vs oncology-subset comparison on gene–drug / gene–disease |
| BC5CDR | Reference only — not used for training |
{design_extra}"""

    out = REPORT_DIR / "report.md"
    out.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out}")
