#!/usr/bin/env python3.11
"""Phase 2A — CIViCmine external baseline (Case C, unfiltered TSV).

Writes JSON + Markdown under ``phase_d_baselines/outputs/`` for supplement
S13.1 and Phase 2D reporting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge_grounded_evidence_audit.analysis.phase_d_baselines.civicmine.civicmine_matching import (
    civicmine_predicted_spair_label,
    index_civicmine_by_pmid,
    load_enriched_eval_targets,
    pair_slot_presence,
    strict_pair_match,
    DEFAULT_CIVICMINE,
)
from knowledge_grounded_evidence_audit.analysis.phase_d_baselines.analysis.pb_civicmine_subset_kb import (
    pb_subset_kb_block_from_covered_targets,
)
from fine_tuning_experiments.schema_exp.eval.schema_expected_label import schema_expected_label_set

SCRIPT = Path(__file__).resolve().parent
OUT_DIR = SCRIPT.parent / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--civicmine-tsv-gz", type=Path, default=DEFAULT_CIVICMINE)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument(
        "--phase-b-runs-root",
        type=Path,
        default=Path("/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b"),
        help="Root that contains PB_PB_FT_* seed directories used for KB subset aggregates.",
    )
    ap.add_argument("--skip-pb-subset-merge", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    by = index_civicmine_by_pmid(args.civicmine_tsv_gz)
    targets = load_enriched_eval_targets()

    covered: list[dict[str, object]] = []
    missing_pmid: list[str] = []
    pmid_no_pair: list[dict[str, object]] = []

    for t in targets:
        rows = by.get(t["pmid"], [])
        if not rows:
            missing_pmid.append(t["target_id"])
            continue
        if any(strict_pair_match(r, t) for r in rows):
            match_rows = [r for r in rows if strict_pair_match(r, t)]
            pick = max(match_rows, key=lambda x: float(x.get("evidencetype_prob") or 0))
            pred = civicmine_predicted_spair_label(pick["evidencetype"], t["pairing_family"])
            civic = {
                "expected_pairing_family": t["pairing_family"],
                "heuristic_gold_s2_label": t["expected_label"],
            }
            exp_set, _ = schema_expected_label_set(civic, "S_pair", "primary", "set_valued")
            hit = int(pred in exp_set)
            covered.append({
                "target_id": t["target_id"],
                "pairing_family": t["pairing_family"],
                "pmid": t["pmid"],
                "civicmine_evidencetype": pick["evidencetype"],
                "pred_spair_label": pred,
                "expected_label_s2": t["expected_label"],
                "expected_set_sv": sorted(exp_set),
                "hit_A_sv_argmax": hit,
                "evidencetype_prob": float(pick.get("evidencetype_prob") or 0.0),
            })
        else:
            a_slot, b_slot = pair_slot_presence(rows, t)
            pmid_no_pair.append({
                "target_id": t["target_id"],
                "pairing_family": t["pairing_family"],
                "pmid": t["pmid"],
                "slot_a_present": bool(a_slot),
                "slot_b_present": bool(b_slot),
                "at_least_one_slot": bool(a_slot or b_slot),
            })

    n162 = len(targets)
    n41 = len(covered)
    acc41 = sum(c["hit_A_sv_argmax"] for c in covered) / n41 if n41 else float("nan")

    breakdown = {
        "n_pmid_no_pair": len(pmid_no_pair),
        "n_at_least_one_entity_slot": sum(1 for p in pmid_no_pair if p["at_least_one_slot"]),
        "n_neither_entity_slot": sum(1 for p in pmid_no_pair if not p["at_least_one_slot"]),
    }

    n_pmid_in_civicmine_any_row = sum(1 for t in targets if by.get(str(t["pmid"]).strip()))

    out = {
        "coverage_rates_unfiltered_relaxed": {
            "strict_entity_pair_cover_162": n41 / max(1, n162),
            "pmid_only_cover_162": n_pmid_in_civicmine_any_row / max(1, n162),
            "strict_count": n41,
            "pmid_only_count": n_pmid_in_civicmine_any_row,
        },
        "n_evaluable_targets": n162,
        "n_civicmine_strict_covered": n41,
        "n_missing_pmid": len(missing_pmid),
        "civicmine_kb_argmax_accuracy_strict41_mean": acc41,
        "civicmine_kb_argmax_accuracy_strict41_note": (
            "Selection-biased: strict coverage follows CIViCmine's realised extraction support. "
            "Compare against PB models only on the matched 41-target denominator (see pb_pubmedbert_kb_on_civicmine_strict_subset)."
        ),
        "pmid_present_no_pair_breakdown": breakdown,
        "covered_targets": covered,
        "pmid_present_no_pair_targets": pmid_no_pair,
    }

    if not args.skip_pb_subset_merge:
        out["pb_pubmedbert_kb_on_civicmine_strict_subset"] = pb_subset_kb_block_from_covered_targets(
            covered, runs_root=args.phase_b_runs_root,
        )

    json_path = args.out_dir / "civicmine_baseline_case_c.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md_path = args.out_dir / "civicmine_baseline_case_c.md"
    md_lines = [
        "# Phase 2A — CIViCmine baseline (Case C, unfiltered TSV)",
        "",
        f"- Evaluable KB targets: **{n162}**",
        f"- Strict entity-pair coverage: **{n41}** ({n41 / n162:.3%})",
        f"- PMID-only coverage (any CIViCmine row sharing the curator PMID): **{n_pmid_in_civicmine_any_row}** "
        f"({n_pmid_in_civicmine_any_row / n162:.3%})",
        f"- CIViCmine KB argmax on strict subset (selection-biased): **{acc41:.4f}** "
        "(use JSON ``pb_pubmedbert_kb_on_civicmine_strict_subset`` for matched PB denominators)",
        "",
    ]
    if not args.skip_pb_subset_merge and "pb_pubmedbert_kb_on_civicmine_strict_subset" in out:
        pb_t2_blk = out["pb_pubmedbert_kb_on_civicmine_strict_subset"].get("PB_T2_on_strict41") or {}
        pv = pb_t2_blk.get("subset_kb_hit_mean_seed_mean")
        if pv is not None:
            md_lines.append(f"- **PB × T2 KB argmax mean (20 seeds, same 41 targets):** **{pv:.4f}**")
        md_lines.append("")

    md_lines.extend([
        "## PMID-present but no entity-pair match (sensitivity analysis for §S13.1)",
        "",
        f"- Count: **{breakdown['n_pmid_no_pair']}**",
        f"- At least one gold slot seen in CIViCmine extractions: **{breakdown['n_at_least_one_entity_slot']}**",
        f"- Neither slot seen: **{breakdown['n_neither_entity_slot']}**",
        "",
        f"Full per-target table: `{json_path.name}`",
        "",
    ])
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
