# Table Inventory

Current generated tables:

| Table | Files | Role |
|---|---|---|
| T2 | `table02_phase_b_cell_results.{csv,md}` | Phase B cell-level FT results (190-row aggregate collapsed to cells). |
| T3 | `table03_phase_b_hypothesis_summary.{csv,md}` | H1-H7 and RQ3 summary with paper implications. |
| T4 | `table04_rq_evidence_matrix.{csv,md}` | RQ-level evidence map and limitations. |

The numbering starts at T2 because T1 is the schema/data inventory drafted in
`paper_methods_draft.md` and supported by the Phase A design documents.

Regenerate all current tables with:

```bash
python3.11 report/scripts/build_phase_b_final_assets.py
```
