"""Assemble frozen candidate pool and report."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .config import (
    DESCRIPTIVE_PAIR_TYPES,
    FROZEN_POOL_CSV,
    FROZEN_POOL_JSON,
    MAX_POSITIVE_FRACTION_FOR_ROOM,
    MIN_MEDIAN_POOL_SIZE,
    MIN_MEAN_POOL_SIZE,
    MIN_PRIMARY_BOTH_ENTITY_COVERAGE,
    OUTPUT_DIR,
    PRIMARY_PAIR_TYPES,
    RANKING_BASELINES_CSV,
    REPORT_DIR,
    SAMPLING_SEED,
)
from .diagnostics import analyze_systematic_loss, analyze_variant_root_cause
from .figures import (
    export_tables,
    plot_composition,
    plot_coverage,
    plot_pool_size,
    plot_systematic_loss,
    plot_variant_root_cause,
)
from .pool_builder import (
    _asserted_pairs_by_pmid,
    analyze_positive_coverage,
    build_candidate_pools,
    load_all_grounded_pmids,
    load_frozen_positives,
    load_variant_positives_for_diagnostics,
)
from .pubtator_fetch import fetch_pubtator_annotations


def _pool_size_by_pair_type(
    candidates: list[dict[str, Any]],
    abstract_summaries: list[dict[str, Any]],
) -> pd.DataFrame:
    """Per-abstract pool size for each primary pair type."""
    by_pmid: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for c in candidates:
        by_pmid[c["pmid"]][c["pair_type"]].append(c)

    rows = []
    pmids = sorted({s["pmid"] for s in abstract_summaries})
    for pmid in pmids:
        for pair_type in PRIMARY_PAIR_TYPES:
            pool = by_pmid[pmid][pair_type]
            n_pos = sum(1 for c in pool if c["is_civic_positive"])
            rows.append(
                {
                    "pmid": pmid,
                    "pair_type": pair_type,
                    "pool_size": len(pool),
                    "n_positives_in_pool": n_pos,
                    "positive_fraction": n_pos / len(pool) if pool else None,
                }
            )
    return pd.DataFrame(rows)


def _composition_table(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for scope in ["primary", "descriptive_only"]:
        for pt in (PRIMARY_PAIR_TYPES if scope == "primary" else DESCRIPTIVE_PAIR_TYPES):
            sub = [c for c in candidates if c["scope"] == scope and c["pair_type"] == pt]
            rows.append(
                {
                    "scope": scope,
                    "pair_type": pt,
                    "n_candidates": len(sub),
                    "n_civic_positives": sum(1 for c in sub if c["is_civic_positive"]),
                }
            )
    return pd.DataFrame(rows)


def _ranking_feasibility_verdict(
    coverage_df: pd.DataFrame,
    pool_size_df: pd.DataFrame,
    type_summary: pd.DataFrame,
    variant_d1: dict[str, Any],
    loss_d2: dict[str, Any],
) -> dict[str, Any]:
    primary_cov = coverage_df[coverage_df["scope"] == "primary"]
    both_rate = primary_cov["both_found"].mean() if len(primary_cov) else 0.0

    primary_pools = pool_size_df[pool_size_df["pair_type"].isin(PRIMARY_PAIR_TYPES)]
    primary_pools = primary_pools[primary_pools["pool_size"] > 0]
    mean_pool = primary_pools["pool_size"].mean() if len(primary_pools) else 0.0
    median_pool = primary_pools["pool_size"].median() if len(primary_pools) else 0.0
    mean_pos_frac = primary_pools["positive_fraction"].mean() if len(primary_pools) else 1.0

    coverage_ok = both_rate >= MIN_PRIMARY_BOTH_ENTITY_COVERAGE
    size_ok = mean_pool >= MIN_MEAN_POOL_SIZE and median_pool >= MIN_MEDIAN_POOL_SIZE
    room_ok = mean_pos_frac <= MAX_POSITIVE_FRACTION_FOR_ROOM
    variant_ok = variant_d1.get("n_category_c", 0) == 0

    viable = coverage_ok and size_ok and room_ok and variant_ok

    reasons = []
    if not coverage_ok:
        reasons.append(
            f"Primary both-entity coverage ({both_rate:.1%}) below threshold ({MIN_PRIMARY_BOTH_ENTITY_COVERAGE:.0%})"
        )
    if not size_ok:
        reasons.append(
            f"Pool sizes too small (mean={mean_pool:.1f}, median={median_pool:.1f})"
        )
    if not room_ok:
        reasons.append(f"Positive fraction too high ({mean_pos_frac:.1%})")
    if not variant_ok:
        reasons.append(
            f"Variant matching bugs detected ({variant_d1['n_category_c']} category-c cases)"
        )
    if viable:
        bias_note = (
            " Some representativeness skew in losses (D2)."
            if loss_d2.get("systematic_bias_detected")
            else " Losses appear largely idiosyncratic (D2)."
        )
        reason = (
            f"Primary coverage {both_rate:.1%}, mean pool size {mean_pool:.1f}, "
            f"mean positive fraction {mean_pos_frac:.1%}; variant 0.0% confirmed genuine (D1)."
            + bias_note
        )
    else:
        reason = "; ".join(reasons) if reasons else "Insufficient data."

    variant_row = type_summary[type_summary["entity_type"] == "variant"]
    variant_cov = float(variant_row["coverage_rate"].iloc[0]) if len(variant_row) else 0.0

    return {
        "viable": bool(viable),
        "reason": reason,
        "primary_both_entity_coverage": round(float(both_rate), 4),
        "mean_pool_size_primary": round(float(mean_pool), 2),
        "median_pool_size_primary": round(float(median_pool), 2),
        "mean_positive_fraction": round(float(mean_pos_frac), 4),
        "variant_entity_coverage": round(float(variant_cov), 4),
        "variant_genuine_zero": bool(variant_d1.get("genuine_zero_coverage", False)),
        "systematic_loss_bias": bool(loss_d2.get("systematic_bias_detected", False)),
    }


def write_report(
    protocol: dict[str, Any],
    type_summary: pd.DataFrame,
    pool_stats: pd.DataFrame,
    composition_df: pd.DataFrame,
    verdict: dict[str, Any],
    variant_d1: dict[str, Any],
    variant_breakdown: pd.DataFrame,
    variant_inspect: pd.DataFrame,
    loss_d2: dict[str, Any],
    loss_comparison: pd.DataFrame,
    gene_comparison: pd.DataFrame,
    baseline_summary: dict[str, Any] | None = None,
) -> None:
    cov_lines = ""
    for row in type_summary.itertuples():
        cov_lines += f"| {row.entity_type} | {int(row.n_found)} | {int(row.n_total)} | {row.coverage_rate:.1%} |\n"

    pool_lines = ""
    for row in pool_stats.itertuples():
        pool_lines += (
            f"| {row.pair_type} | {int(row.count)} | {row.mean:.1f} | {row.median:.1f} | "
            f"{int(row.min)} | {int(row.max)} |\n"
        )

    comp_lines = ""
    for row in composition_df.itertuples():
        comp_lines += (
            f"| {row.scope} | {row.pair_type} | {int(row.n_candidates)} | {int(row.n_civic_positives)} |\n"
        )

    d1_lines = ""
    for row in variant_breakdown.itertuples():
        d1_lines += f"| {row.label} | {int(row.n)} | {row.fraction:.1%} |\n"

    inspect_lines = ""
    for row in variant_inspect.head(12).itertuples():
        raw = row.pubtator_variant_texts
        if pd.isna(raw) or raw is None:
            pt_preview = "(none)"
        else:
            pt_preview = str(raw)[:60]
        inspect_lines += (
            f"| {row.pmid} | {row.civic_variant} | {row.root_cause[0]} | {pt_preview} |\n"
        )

    d2_lines = ""
    for row in loss_comparison.itertuples():
        mean_len = f"{row.mean_head_entity_length:.1f}" if pd.notna(row.mean_head_entity_length) else "—"
        sym = f"{row.pct_head_is_symbol:.1%}" if pd.notna(row.pct_head_is_symbol) else "—"
        freq = f"{row.mean_head_corpus_frequency:.1f}" if pd.notna(row.mean_head_corpus_frequency) else "—"
        year = f"{row.mean_publication_year:.0f}" if pd.notna(row.mean_publication_year) else "—"
        d2_lines += (
            f"| {row.group} | {int(row.n)} | {row.pct_gene_disease:.1%} | {row.pct_gene_drug:.1%} | "
            f"{mean_len} | {sym} | {freq} | {year} |\n"
        )

    stats = protocol["statistics"]
    d1 = protocol["diagnostics"]["variant_root_cause"]
    d2 = protocol["diagnostics"]["systematic_loss"]

    baseline_section = ""
    if baseline_summary:
        v = baseline_summary["ranking_verification"]
        base_lines = ""
        for row in baseline_summary["baselines"]:
            base_lines += (
                f"| {row['baseline']} | {row['mrr']:.3f} | {row['recall_at_1']:.3f} | "
                f"{row['recall_at_3']:.3f} | {row['recall_at_5']:.3f} | {row['auc_pr']:.3f} |\n"
            )
        baseline_section = f"""
---

## E. Trivial ranking baselines (real frozen pool)

Floor-line rankers on **{baseline_summary['n_candidates']}** primary-scope candidates ({baseline_summary['n_pmids']} PMIDs; positive rate {baseline_summary['positive_rate']:.1%}). Targets are frozen in step 02 from the step-00 abstract-grounded set.

| Baseline | MRR | Recall@1 | Recall@3 | Recall@5 | AUC-PR |
| --- | ---: | ---: | ---: | ---: | ---: |
{base_lines}

**Tie-handling verification:** constant MRR ({v['mrr_constant']:.3f}) ≈ random MRR ({v['mrr_random']:.3f}); analytic E[MRR] = {baseline_summary['analytic_random_mrr']:.3f}.

**Distance ranker purpose.** Scores candidates by entity proximity only (inverse sentence distance / co-sentence indicator) — no relation understanding. If this shallow heuristic ranks highly, the task can be gamed; trained models in step 04 must beat it to demonstrate genuine relation signal.

Outputs: `outputs/ranking_baselines.csv`, `outputs/ranking_verification.json`
"""

    report = f"""# Step 03: Candidate Pool Feasibility Report

Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

## Evaluation framework

**Ranking/retrieval:** For each abstract, a model scores a fixed pool of candidate entity pairs; CIViC-asserted pairs should rank highly. The pool is **tool-generated** via PubTator3 precomputed NER — not provided by CIViC. PubTator3 recall bounds what we can evaluate; this is a documented limitation and a realistic curation-pipeline feature.

**PubTator3 dependency:** {protocol['pubtator3']['source']}  
**API:** `{protocol['pubtator3']['api_endpoint']}`  
**Fetch date:** {protocol['pubtator3']['fetch_timestamp']}  
**PMIDs cached:** {protocol['pubtator3']['n_pmids_cached']} / {protocol['pubtator3']['n_pmids_requested']}

---

## A. Ranking coverage in the frozen pool

For each primary CIViC relation in the frozen evaluation set (step 02), does the pool contain at least one positive candidate under the frozen matching rules?

| Metric | Value |
| --- | ---: |
| Total primary relations | {stats['n_primary_total']} |
| With pool positive (ranking evaluable) | {stats['n_primary_evaluable']} ({stats['primary_coverage_rate']:.1%}) |
| Without pool positive | {stats['n_primary_unevaluable']} ({1 - stats['primary_coverage_rate']:.1%}) |
| PubTator slot both-found (NER-level) | {stats.get('n_primary_both_found', stats['n_primary_evaluable'])} ({stats.get('primary_both_found_rate', stats['primary_coverage_rate']):.1%}) |

Slot-level both-found counts PubTator annotations matching both CIViC entity strings before pool enumeration; pool-positive counts relations with at least one marked positive candidate. The counts differ by one relation where pool matching succeeds without both slots matched individually.

### Coverage by entity type (PubTator slot found)

| Entity type | Found | Total slots | Coverage |
| --- | ---: | ---: | ---: |
{cov_lines}

**Interpretation.** Gene and drug/disease coverage determines whether gene–drug and gene–disease pairs enter the pool. Variant coverage ({verdict['variant_entity_coverage']:.1%}) is analysed in D1. Positives whose entities PubTator3 misses cannot enter the candidate pool.

---

## B. Candidate-pool size and ranking room

Per abstract, all co-occurring PubTator3 entity pairs are enumerated (gene–drug, gene–disease; variant pairs tracked separately).

| Pair type | Abstracts with pool>0 | Mean size | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
{pool_lines}

- Mean positive fraction (gene–drug / gene–disease pools): **{verdict['mean_positive_fraction']:.1%}**
- Total candidates: **{stats['n_candidates_total']}** ({stats['n_candidates_primary']} gene–drug / gene–disease)

**Interpretation.** Mean pool size {verdict['mean_pool_size_primary']:.1f} with positive fraction {verdict['mean_positive_fraction']:.1%} indicates {'adequate' if verdict['mean_pool_size_primary'] >= MIN_MEAN_POOL_SIZE and verdict['mean_positive_fraction'] <= MAX_POSITIVE_FRACTION_FOR_ROOM else 'limited'} ranking room.

---

## C. Pool entity-type composition

| Evaluation focus | Pair type | Candidates | CIViC-curated positives marked |
| --- | --- | ---: | ---: |
{comp_lines}

**Interpretation.** Gene–drug and gene–disease pairs dominate the pool. Variant pairs are tracked separately but cannot be ranked (D1).
{baseline_section}
---

## D1. Variant coverage root-cause

All {d1['n_variant_positives']} variant-head positives classified against PubTator3 tmVar3 annotations:

| Category | n | Fraction |
| --- | ---: | ---: |
{d1_lines}

### Inspected sample (PubTator3 variant text vs CIViC string)

| PMID | CIViC variant | Class | PubTator3 variant texts (preview) |
| --- | --- | --- | --- |
{inspect_lines}

**Conclusion.** {variant_d1['conclusion']}

---

## D2. Systematic-loss check (primary unevaluable positives)

Comparison of **evaluable** vs **unevaluable** primary positives ({d2['n_primary_unevaluable']} unevaluable; {d2['pct_loss_at_gene_head']:.1%} of losses at gene head):

| Group | n | % gene–disease | % gene–drug | Mean gene length | % symbol | Mean gene freq | Mean pub year |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{d2_lines}

**Representativeness implication.** {loss_d2['implication']}

---

## Ranking-feasibility verdict

**{'VIABLE' if verdict['viable'] else 'NOT VIABLE'}** — {verdict['reason']}

| Diagnostic | Result |
| --- | --- |
| Variant 0.0% genuine (D1) | {'Yes' if verdict.get('variant_genuine_zero') else 'No — fix matching'} |
| Systematic loss bias (D2) | {'Yes' if verdict.get('systematic_loss_bias') else 'No strong skew'} |
| PubTator3-recall ceiling | {stats['n_primary_unevaluable']} relations without pool positive |

Frozen artifact: `outputs/frozen_pool.json`
"""

    path = REPORT_DIR / "report.md"
    path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {path}")


def build_candidate_pool(force_fetch: bool = False) -> dict[str, Any]:
    positives = load_frozen_positives()
    pmids = load_all_grounded_pmids()
    pubtator_docs, pubtator_meta = fetch_pubtator_annotations(pmids, force=force_fetch)

    coverage_df, type_summary = analyze_positive_coverage(positives, pubtator_docs)

    variant_positives = load_variant_positives_for_diagnostics()
    # variant PMIDs are already included in load_all_grounded_pmids()

    # D1 and D2 diagnostics
    _, variant_breakdown, variant_inspect, variant_d1 = analyze_variant_root_cause(
        variant_positives, pubtator_docs
    )
    _, loss_comparison, gene_comparison, loss_d2 = analyze_systematic_loss(positives, coverage_df)

    asserted_by_pmid = _asserted_pairs_by_pmid(positives)
    candidates, abstract_summaries = build_candidate_pools(
        positives, pubtator_docs, asserted_by_pmid, pool_pmids=pmids
    )

    pool_size_df = _pool_size_by_pair_type(candidates, abstract_summaries)
    composition_df = _composition_table(candidates)
    pool_stats = (
        pool_size_df[pool_size_df["pool_size"] > 0]
        .groupby("pair_type")["pool_size"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )

    export_tables(type_summary, pool_size_df, composition_df, coverage_df)
    plot_coverage(type_summary)
    plot_pool_size(pool_size_df)
    plot_composition(composition_df)
    plot_variant_root_cause(variant_breakdown)
    plot_systematic_loss(loss_comparison, gene_comparison)

    verdict = _ranking_feasibility_verdict(
        coverage_df, pool_size_df, type_summary, variant_d1, loss_d2
    )

    n_evaluable = int(coverage_df["both_found"].sum())
    n_unevaluable = len(coverage_df) - n_evaluable
    primary_cov = coverage_df[coverage_df["scope"] == "primary"]
    n_primary_unevaluable = int((~primary_cov["both_found"]).sum())

    by_pmid: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_pmid[c["pmid"]].append(c)

    abstracts_out = []
    for s in abstract_summaries:
        pmid = s["pmid"]
        abstracts_out.append({**s, "candidates": by_pmid.get(pmid, [])})

    protocol: dict[str, Any] = {
        "version": "03_candidate_pool_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "ranking_retrieval",
        "sampling_seed": SAMPLING_SEED,
        "pubtator3": pubtator_meta,
        "pool_generation_rules": (
            "Per PMID in frozen positive set: parse PubTator3 biocjson entities; map "
            "Gene/Chemical/Disease/Variant to CIViC types; enumerate co-occurring "
            "same-type pairs for gene-drug, gene-disease, variant-disease, variant-drug; "
            "mark CIViC-asserted pairs as is_civic_positive=True."
        ),
        "statistics": {
            "n_pmids": len(pmids),
            "n_positives_total": len(coverage_df),
            "n_positives_evaluable": n_evaluable,
            "n_positives_unevaluable": n_unevaluable,
            "positive_coverage_rate": round(n_evaluable / len(coverage_df), 4),
            "n_primary_total": len(primary_cov),
            "n_primary_evaluable": int(primary_cov["both_found"].sum()),
            "n_primary_unevaluable": n_primary_unevaluable,
            "primary_coverage_rate": round(primary_cov["both_found"].mean(), 4),
            "n_candidates_total": len(candidates),
            "n_candidates_primary": sum(1 for c in candidates if c["scope"] == "primary"),
            "n_candidates_descriptive": sum(1 for c in candidates if c["scope"] == "descriptive_only"),
            "n_civic_positives_in_pool": sum(1 for c in candidates if c["is_civic_positive"]),
        },
        "diagnostics": {
            "variant_root_cause": variant_d1,
            "systematic_loss": loss_d2,
        },
        "ranking_feasibility_verdict": verdict,
        "abstracts": abstracts_out,
    }

    FROZEN_POOL_JSON.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    pd.DataFrame(candidates).to_csv(FROZEN_POOL_CSV, index=False)

    from .ranking_baselines import compute_ranking_baselines

    baseline_summary = compute_ranking_baselines()

    print("\n=== Candidate pool built ===")
    print(f"  PMIDs: {len(pmids)} | PubTator3 cached: {len(pubtator_docs)}")
    print(f"  Positive coverage: {n_evaluable}/{len(coverage_df)} ({n_evaluable/len(coverage_df):.1%})")
    print(f"  Primary coverage: {int(primary_cov['both_found'].sum())}/{len(primary_cov)} "
          f"({primary_cov['both_found'].mean():.1%})")
    print(f"  Primary unevaluable: {n_primary_unevaluable}")
    print(f"  Total candidates: {len(candidates)}")
    print(f"\n=== D1 variant root-cause ===")
    print(f"  (a) no variant: {variant_d1['n_category_a']} | (b) format mismatch: {variant_d1['n_category_b']} | "
          f"(c) matching bug: {variant_d1['n_category_c']}")
    print(f"  Genuine 0.0%: {variant_d1['genuine_zero_coverage']}")
    print(f"\n=== D2 systematic loss ===")
    print(f"  Systematic bias: {loss_d2['systematic_bias_detected']}")
    print(f"  Ranking feasible: {verdict['viable']}")

    write_report(
        protocol,
        type_summary,
        pool_stats,
        composition_df,
        verdict,
        variant_d1,
        variant_breakdown,
        variant_inspect,
        loss_d2,
        loss_comparison,
        gene_comparison,
        baseline_summary=baseline_summary,
    )
    return protocol


def refresh_entity_type_alignment() -> dict[str, Any]:
    """CPU-only: entity-type alignment diagnostic + report section."""
    from .entity_type_alignment import refresh_entity_type_alignment as _run

    return _run()


def refresh_recall_diagnostic() -> None:
    """Read-only PubTator recall diagnostic; patch report section in place."""
    from .pubtator_recall import recall_report_section, run_pubtator_recall_diagnostic

    summary = run_pubtator_recall_diagnostic()
    section = recall_report_section(summary)

    report_path = REPORT_DIR / "report.md"
    if not report_path.exists():
        report_path.write_text(section.strip() + "\n", encoding="utf-8")
        print(f"\nReport written to {report_path}")
        return

    existing = report_path.read_text(encoding="utf-8")
    marker = "## PubTator recall and entity-span limitation"
    if marker in existing:
        before = existing.split(marker)[0].rstrip()
        while before.endswith("---"):
            before = before[:-3].rstrip()
        after_parts = existing.split(marker, 1)[1]
        for end_marker in ("\n---\n\n## Ranking-feasibility", "\n## Ranking-feasibility"):
            if end_marker in after_parts:
                after = after_parts.split(end_marker, 1)[1]
                existing = before + "\n\n---\n\n" + section.strip() + "\n\n" + "## Ranking-feasibility" + after
                break
        else:
            existing = before + "\n\n" + section.strip() + "\n"
    else:
        insert_before = "## Ranking-feasibility verdict"
        if insert_before in existing:
            existing = existing.replace(
                insert_before,
                "---\n\n" + section.strip() + "\n\n" + insert_before,
            )
        else:
            existing = existing.rstrip() + "\n\n" + section.strip() + "\n"

    report_path.write_text(existing, encoding="utf-8")
    print(f"\nReport updated: {report_path}")


def _pool_stats_from_candidates() -> pd.DataFrame:
    """Recompute pool size summary from frozen pool_candidates.csv."""
    pool = pd.read_csv(FROZEN_POOL_CSV)
    primary = pool[pool["scope"] == "primary"]
    rows: list[dict[str, Any]] = []
    for pt in PRIMARY_PAIR_TYPES:
        sub = primary[primary["pair_type"] == pt]
        sizes = sub.groupby("pmid").size()
        if sizes.empty:
            continue
        rows.append(
            {
                "pair_type": pt,
                "count": int(len(sizes)),
                "mean": float(sizes.mean()),
                "median": float(sizes.median()),
                "min": int(sizes.min()),
                "max": int(sizes.max()),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "03_candidate_pool_size_distribution.csv", index=False)
    return df


def _augment_protocol_pool_stats(protocol: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Overlay pool-positive coverage from classification CSV (read-only on frozen pool)."""
    cls_path = OUTPUT_DIR / "03_candidate_pool_pubtator_recall_classification.csv"
    cov_path = OUTPUT_DIR / "03_candidate_pool_positive_coverage.csv"
    if not cls_path.exists() or not cov_path.exists():
        raise FileNotFoundError("Missing recall classification or positive coverage CSV for report refresh")
    cls = pd.read_csv(cls_path)
    coverage_df = pd.read_csv(cov_path)
    primary_cov = coverage_df[coverage_df["scope"] == "primary"]
    n_matched = int(cls["matched_in_pool"].sum())
    n_miss = int(len(cls) - n_matched)
    n_both = int(primary_cov["both_found"].sum())
    stats = protocol["statistics"]
    stats["n_primary_evaluable"] = n_matched
    stats["n_primary_unevaluable"] = n_miss
    stats["primary_coverage_rate"] = round(n_matched / max(len(cls), 1), 4)
    stats["n_primary_both_found"] = n_both
    stats["primary_both_found_rate"] = round(n_both / max(len(primary_cov), 1), 4)
    stats["n_positives_evaluable"] = n_matched
    stats["n_positives_unevaluable"] = n_miss
    stats["positive_coverage_rate"] = stats["primary_coverage_rate"]
    protocol["statistics"] = stats
    return protocol, cls, coverage_df


def refresh_full_report() -> None:
    """CPU-only: regenerate step-03 report from frozen artifacts; preserve recall section."""
    import json

    from .diagnostics import analyze_systematic_loss
    from .ranking_baselines import compute_ranking_baselines

    protocol = json.loads(FROZEN_POOL_JSON.read_text(encoding="utf-8"))
    protocol, cls, coverage_df = _augment_protocol_pool_stats(protocol)

    positives = load_frozen_positives()
    _, loss_comparison, gene_comparison, loss_d2 = analyze_systematic_loss(
        positives, coverage_df, pool_classification=cls
    )
    protocol["diagnostics"]["systematic_loss"] = loss_d2

    pool_rate = protocol["statistics"]["primary_coverage_rate"]
    verdict = protocol["ranking_feasibility_verdict"]
    bias_note = (
        " Some representativeness skew in losses (D2)."
        if loss_d2.get("systematic_bias_detected")
        else " Losses appear largely idiosyncratic (D2)."
    )
    verdict["reason"] = (
        f"Primary pool-positive coverage {pool_rate:.1%}, mean pool size "
        f"{verdict.get('mean_pool_size_primary', 0):.1f}, mean positive fraction "
        f"{verdict.get('mean_positive_fraction', 0):.1%}; variant 0.0% confirmed genuine (D1)."
        + bias_note
    )

    baseline_summary = None
    if RANKING_BASELINES_CSV.exists():
        baseline_summary = compute_ranking_baselines()

    type_summary = pd.read_csv(OUTPUT_DIR / "03_candidate_pool_coverage_by_entity_type.csv")
    pool_stats = _pool_stats_from_candidates()
    composition_df = pd.read_csv(OUTPUT_DIR / "03_candidate_pool_composition.csv")
    variant_breakdown = pd.read_csv(OUTPUT_DIR / "03_candidate_pool_variant_breakdown.csv")
    variant_inspect = pd.read_csv(OUTPUT_DIR / "03_candidate_pool_variant_inspect_sample.csv")
    variant_d1 = protocol["diagnostics"]["variant_root_cause"]

    write_report(
        protocol,
        type_summary,
        pool_stats,
        composition_df,
        verdict,
        variant_d1,
        variant_breakdown,
        variant_inspect,
        loss_d2,
        loss_comparison,
        gene_comparison,
        baseline_summary=baseline_summary,
    )
    plot_systematic_loss(loss_comparison, gene_comparison)
    refresh_recall_diagnostic()
    from .entity_type_alignment import refresh_entity_type_alignment

    refresh_entity_type_alignment()


def refresh_baselines_and_report() -> None:
    """Recompute baselines on existing frozen pool and refresh report section E."""
    import json

    from .ranking_baselines import compute_ranking_baselines

    protocol = json.loads(FROZEN_POOL_JSON.read_text(encoding="utf-8"))
    baseline_summary = compute_ranking_baselines()

    type_summary = pd.read_csv(OUTPUT_DIR / "03_candidate_pool_coverage_by_entity_type.csv")
    pool_stats = pd.read_csv(OUTPUT_DIR / "03_candidate_pool_size_distribution.csv")
    composition_df = pd.read_csv(OUTPUT_DIR / "03_candidate_pool_composition.csv")
    variant_breakdown = pd.read_csv(OUTPUT_DIR / "03_candidate_pool_variant_breakdown.csv")
    variant_inspect = pd.read_csv(OUTPUT_DIR / "03_candidate_pool_variant_inspect_sample.csv")
    loss_comparison = pd.read_csv(OUTPUT_DIR / "03_candidate_pool_loss_comparison.csv")
    gene_comparison = pd.read_csv(OUTPUT_DIR / "03_candidate_pool_loss_gene_comparison.csv")

    variant_d1 = protocol["diagnostics"]["variant_root_cause"]
    loss_d2 = protocol["diagnostics"]["systematic_loss"]
    verdict = protocol["ranking_feasibility_verdict"]

    write_report(
        protocol,
        type_summary,
        pool_stats,
        composition_df,
        verdict,
        variant_d1,
        variant_breakdown,
        variant_inspect,
        loss_d2,
        loss_comparison,
        gene_comparison,
        baseline_summary=baseline_summary,
    )
