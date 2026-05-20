# Inventory check — dossier completeness (self-audit)

Generated alongside `MANUSCRIPT_EVIDENCE_DOSSIER.md` (**57** Evidence ID blocks: **E-001** … **E-056**, including **E-002b**).

## Counts

| Metric | Count |
|--------|------:|
| Total evidence entries | **57** |
| **STABLE** (locked analysis JSON / reproducible Phase D artefacts / Phase A+B primary) | **39** |
| **FRAGILE** (wide CI, second-source manuscript numbers, external comparisons under validity gates, or `.tex`-only ICC) | **6** |
| **DEPRECATED** (superseded **claims/framing**, not necessarily wrong numbers) | **1** |
| Entries with numeric claims and **`[PROVENANCE UNKNOWN]`** (Phase **C** batch **E-020–E-030**; optional ICC recomputation **E-056**) | **11** (+ **1** optional) |

### STABLE list (IDs)

E-001, E-002, E-002b, E-004, E-005, E-007, E-008, E-009, E-011, E-012, E-013, E-014, E-015, E-016, E-017, E-018, E-019, E-031, E-032, E-033, E-034, E-035, E-036, E-037, E-038, E-039, E-040, E-041, E-042, E-043, E-044, E-046, E-047, E-048, E-050, E-051, E-052, E-053, E-054

### FRAGILE list (IDs)

E-003 (encoder deltas — verify vs Supplement E), E-006 (encoder-specific H2 — verify vs Supplement F), **E-010** (R_B ratio CI crosses 1), **E-045** (α̂ bootstrap width > 0.95), **E-049** (extended R_B designs), **E-056** (ICC/SD paragraph cited from `.tex` only)

### DEPRECATED list (IDs)

**E-055** — manuscript Methods sentence treating LLM IAA as “lower bound” on human agreement (**superseded** by author κ layer).

### `[PROVENANCE UNKNOWN]` numeric shelf (needs file+key)

**E-020 – E-030** (entire Phase C robustness subsection as specified in Phase 3 brief), plus **E-056** replication trace to ICC generator.

## Gap detection — manuscript quantities not mapped to an E-ID

The following appear in `report/project/sections/*.tex` but are **not** assigned a dedicated Evidence ID in this dossier (merge into supplement tables or add E-057+ on revision):

1. **Schema table FDR-q** for first-row permutation (manuscript: **q=0.024** vs `phase_a_analysis.json` permutation **p=0.015** only) — requires **multiple-testing / FDR output** file (Supplement E tier).
2. **Wilcoxon vs t q-values** for auxiliary H1 contrasts (**PB vs BL**) — only partially used in prose; full triple in `H1_encoder.tests[2]` (**E-008** covers primary PL-centric headline).
3. **BioLink / PL staging marginal** narrative already inside **E-007** — no gap.
4. **Figure-only diagnostics** (e.g. forest plot secondary labels) — cite figures as visual duplicates of **F** JSON.
5. **ρ grid Clopper intervals** for **0.01** and **0.05** besides **0.03** — values live in `rho_sensitivity_*.json` but not duplicated as separate E-IDs (optional add-on).
6. **Cross-metric Pearson/Spearman blocks** in `phase_a_analysis.json` (`cross_metric_correlations`) — omit unless Discussion cites.

**Action:** if Phase 3 prose needs any gap item, mint **E-057+** with the same dossier template.

## File manifest (this dossier)

- `MANUSCRIPT_EVIDENCE_DOSSIER.md`
- `quick_reference_numbers_card.md`
- `provenance_map.md`
- `inventory_check.md` (this file)
- `section_by_section_evidence/abstract_evidence.md`
- `section_by_section_evidence/background_evidence.md`
- `section_by_section_evidence/methods_evidence.md`
- `section_by_section_evidence/results_evidence.md`
- `section_by_section_evidence/discussion_evidence.md`
- `section_by_section_evidence/conclusion_evidence.md`
- `section_by_section_evidence/cover_letter_evidence.md`
