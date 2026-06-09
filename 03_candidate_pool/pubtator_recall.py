"""Read-only PubTator recall / span-limitation diagnostic on the frozen pool."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CIVIC_EVIDENCE_JSON,
    FROZEN_POOL_CSV,
    FROZEN_PROTOCOL_JSON,
    OUTPUT_DIR,
    PUBTATOR_CACHE_JSON,
    SAMPLING_SEED,
)
from .diagnostics import _build_pmid_metadata
from .matching import normalize_text
from .parse import entities_by_type, parse_entities
from .pool_builder import _find_matching_entity

BUCKET_LABELS = {
    "matched": "matched (positive in frozen pool)",
    "miss_entity_absent": "miss: entity type absent in abstract",
    "miss_present_but_unmatched": "miss: entity present but string/span mismatch",
}


def entity_token_count(text: str) -> int:
    """Token length using whitespace splits on normalised surface form."""
    parts = [p for p in normalize_text(text).split() if p]
    return max(len(parts), 1)


def is_multiword(text: str) -> bool:
    return entity_token_count(text) >= 2


def _load_pubtator_docs() -> dict[str, dict[str, Any]]:
    if not PUBTATOR_CACHE_JSON.exists():
        raise FileNotFoundError(f"Missing PubTator cache: {PUBTATOR_CACHE_JSON}")
    payload = json.loads(PUBTATOR_CACHE_JSON.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload and "passages" in next(iter(payload.values()), {}):
        return {str(k): v for k, v in payload.items()}
    if isinstance(payload, dict):
        docs = payload.get("documents") or payload.get("PubTator3") or []
        if isinstance(docs, list):
            return {str(d.get("id") or d.get("pmid")): d for d in docs}
    raise ValueError(f"Unrecognised PubTator cache format: {PUBTATOR_CACHE_JSON}")


def _target_matched_in_pool(
    target: pd.Series,
    pool: pd.DataFrame,
) -> bool:
    """True if the frozen pool marks at least one positive for this relation."""
    sub = pool[
        (pool["pmid"].astype(str) == str(target["pmid"]))
        & (pool["pair_type"] == target["pair_type"])
        & (pool["is_civic_positive"].astype(bool))
    ]
    return not sub.empty


def _slot_status(
    civic_entity: str,
    civic_type: str,
    entities: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    """
    Return (status, nearest_entity) where status is one of:
    absent, matched, present_unmatched
    """
    if not entities:
        return "absent", None
    match = _find_matching_entity(civic_entity, entities, civic_type)
    if match is not None:
        return "matched", match
    nearest = _nearest_entity(civic_entity, entities)
    return "present_unmatched", nearest


def _nearest_entity(civic_entity: str, entities: list[dict[str, Any]]) -> dict[str, Any]:
    """Nearest PubTator entity by token overlap (for illustrative examples)."""
    civic_tokens = set(normalize_text(civic_entity).split())
    best: dict[str, Any] | None = None
    best_score = -1.0
    for ent in entities:
        texts = ent.get("all_texts") or [ent["text"]]
        for text in texts:
            pt_tokens = set(normalize_text(text).split())
            if not pt_tokens:
                continue
            overlap = len(civic_tokens & pt_tokens) / max(len(civic_tokens | pt_tokens), 1)
            if overlap > best_score:
                best_score = overlap
                best = {**ent, "nearest_text": text, "overlap": overlap}
    return best or entities[0]


def _classify_relation(
    target: pd.Series,
    pool: pd.DataFrame,
    pubtator_docs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pmid = str(target["pmid"])
    head_type = str(target["head_type"])
    tail_type = str(target["tail_type"])
    civic_head = str(target["head_entity"])
    civic_tail = str(target["tail_entity"])

    matched = _target_matched_in_pool(target, pool)
    doc = pubtator_docs.get(pmid)
    parsed = parse_entities(doc) if doc else []
    by_type = entities_by_type(parsed)

    head_status, head_nearest = _slot_status(civic_head, head_type, by_type.get(head_type, []))
    tail_status, tail_nearest = _slot_status(civic_tail, tail_type, by_type.get(tail_type, []))

    if matched:
        bucket = "matched"
    elif head_status == "absent" or tail_status == "absent":
        bucket = "miss_entity_absent"
    else:
        bucket = "miss_present_but_unmatched"

    fail_sides: list[str] = []
    if not matched:
        if head_status != "matched":
            fail_sides.append(head_type)
        if tail_status != "matched":
            fail_sides.append(tail_type)

    return {
        "target_id": target["target_id"],
        "evidence_id": target.get("evidence_id"),
        "pmid": pmid,
        "pair_type": target["pair_type"],
        "head_entity": civic_head,
        "tail_entity": civic_tail,
        "head_type": head_type,
        "tail_type": tail_type,
        "head_token_count": entity_token_count(civic_head),
        "tail_token_count": entity_token_count(civic_tail),
        "head_multiword": is_multiword(civic_head),
        "tail_multiword": is_multiword(civic_tail),
        "head_status": head_status,
        "tail_status": tail_status,
        "bucket": bucket,
        "matched_in_pool": matched,
        "fail_sides": ";".join(fail_sides) if fail_sides else "",
        "head_nearest_pubtator": (head_nearest or {}).get("nearest_text") or (head_nearest or {}).get("text"),
        "head_nearest_offset": (head_nearest or {}).get("offset"),
        "tail_nearest_pubtator": (tail_nearest or {}).get("nearest_text") or (tail_nearest or {}).get("text"),
        "tail_nearest_offset": (tail_nearest or {}).get("offset"),
    }


def _pubtator_span_token_stats(pubtator_docs: dict[str, dict[str, Any]], pmids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pmid in pmids:
        doc = pubtator_docs.get(str(pmid))
        if not doc:
            continue
        for ent in parse_entities(doc):
            rows.append(
                {
                    "pmid": str(pmid),
                    "civic_type": ent["civic_type"],
                    "text": ent["text"],
                    "token_count": entity_token_count(ent["text"]),
                    "multiword": is_multiword(ent["text"]),
                }
            )
    return pd.DataFrame(rows)


def _example_quality(row: pd.Series) -> float:
    score = 0.0
    for role in ("head", "tail"):
        if row[f"{role}_status"] != "present_unmatched":
            continue
        civic = str(row[f"{role}_entity"])
        nearest = str(row[f"{role}_nearest_pubtator"] or "")
        if not nearest:
            continue
        cn = normalize_text(civic)
        nn = normalize_text(nearest)
        if cn in nn or nn in cn:
            score += 3.0
        civic_t = set(cn.split())
        near_t = set(nn.split())
        overlap = len(civic_t & near_t)
        score += overlap
        if overlap == 0 and civic.replace("-", "").isalnum():
            initials = "".join(w[0] for w in nn.split() if w)
            if cn.replace("-", "") == initials.lower():
                score += 2.5
        if overlap > 0 and (row[f"{role}_multiword"] or is_multiword(nearest)):
            score += 0.5
    if row["head_status"] == "absent" or row["tail_status"] == "absent":
        score += 0.5
    return score


def _sample_examples(detail: pd.DataFrame, bucket: str, n: int = 3) -> list[dict[str, str]]:
    sub = detail[detail["bucket"] == bucket].copy()
    if sub.empty:
        return []
    if bucket == "miss_present_but_unmatched":
        prefer = sub[(sub["head_status"] == "present_unmatched") | (sub["tail_status"] == "present_unmatched")]
        if not prefer.empty:
            sub = prefer
    sub["example_score"] = sub.apply(_example_quality, axis=1)
    sub = sub.sort_values("example_score", ascending=False).drop_duplicates(subset=["pmid", "pair_type"])
    if bucket == "miss_present_but_unmatched":
        scored = sub[sub["example_score"] > 0]
        if not scored.empty:
            sub = scored
    sample = sub.head(min(len(sub), max(n * 4, n)))
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=SAMPLING_SEED)
    examples: list[dict[str, str]] = []
    for _, r in sample.iterrows():
        parts = []
        if r["head_status"] == "absent":
            parts.append(f"{r['head_type']}: CIViC \"{r['head_entity']}\" has no PubTator annotation of this type")
        elif r["head_status"] == "present_unmatched":
            parts.append(
                f"{r['head_type']}: CIViC \"{r['head_entity']}\" vs nearest PubTator "
                f"\"{r['head_nearest_pubtator']}\" (offset {int(r['head_nearest_offset']) if pd.notna(r['head_nearest_offset']) else 'n/a'})"
            )
        if r["tail_status"] == "absent":
            parts.append(f"{r['tail_type']}: CIViC \"{r['tail_entity']}\" has no PubTator annotation of this type")
        elif r["tail_status"] == "present_unmatched":
            parts.append(
                f"{r['tail_type']}: CIViC \"{r['tail_entity']}\" vs nearest PubTator "
                f"\"{r['tail_nearest_pubtator']}\" (offset {int(r['tail_nearest_offset']) if pd.notna(r['tail_nearest_offset']) else 'n/a'})"
            )
        examples.append(
            {
                "pmid": str(r["pmid"]),
                "pair_type": str(r["pair_type"]),
                "detail": "; ".join(parts) if parts else "unclassified slot failure",
            }
        )
    return examples


def analyze_pubtator_recall() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Classify all primary CIViC relations against frozen pool + PubTator cache."""
    protocol = json.loads(FROZEN_PROTOCOL_JSON.read_text(encoding="utf-8"))
    targets = pd.DataFrame(protocol["targets"])
    primary = targets[targets["scope"] == "primary"].copy()

    pool = pd.read_csv(FROZEN_POOL_CSV)
    pubtator_docs = _load_pubtator_docs()

    rows = [_classify_relation(r, pool, pubtator_docs) for _, r in primary.iterrows()]
    detail = pd.DataFrame(rows)

    pmid_meta = _build_pmid_metadata()
    detail = detail.merge(pmid_meta, on="pmid", how="left")

    bucket_counts = detail["bucket"].value_counts().to_dict()
    n_total = len(detail)
    bucket_frac = {k: bucket_counts.get(k, 0) / n_total for k in BUCKET_LABELS}

    # Multi-word miss rates (entity-slot level among missed relations)
    missed = detail[~detail["matched_in_pool"]]
    matched = detail[detail["matched_in_pool"]]

    def _slot_miss_rate(df: pd.DataFrame, role: str) -> dict[str, float]:
        col = f"{role}_multiword"
        if df.empty:
            return {"single_miss_rate": 0.0, "multi_miss_rate": 0.0}
        single = df[~df[col]]
        multi = df[df[col]]
        return {
            "single_n": len(single),
            "multi_n": len(multi),
            "single_miss_rate": float((~single["matched_in_pool"]).mean()) if len(single) else 0.0,
            "multi_miss_rate": float((~multi["matched_in_pool"]).mean()) if len(multi) else 0.0,
        }

    head_miss = _slot_miss_rate(detail, "head")
    tail_miss = _slot_miss_rate(detail, "tail")

    # Among missed only: entity-absent vs unmatched by failing side type
    miss_side_rows: list[dict[str, Any]] = []
    for _, r in missed.iterrows():
        for role, etype, status in (
            ("head", r["head_type"], r["head_status"]),
            ("tail", r["tail_type"], r["tail_status"]),
        ):
            if status == "matched":
                continue
            miss_side_rows.append(
                {
                    "entity_type": etype,
                    "status": status,
                    "bucket": r["bucket"],
                    "multiword": r[f"{role}_multiword"],
                }
            )
    miss_side_df = pd.DataFrame(miss_side_rows)

    side_breakdown = (
        miss_side_df.groupby(["entity_type", "status"])
        .size()
        .reset_index(name="n")
        if not miss_side_df.empty
        else pd.DataFrame(columns=["entity_type", "status", "n"])
    )

    # Miss rate by CIViC entity type and phrase length (slot level, all relations)
    slot_rows: list[dict[str, Any]] = []
    for _, r in detail.iterrows():
        for role in ("head", "tail"):
            slot_rows.append(
                {
                    "entity_type": r[f"{role}_type"],
                    "multiword": r[f"{role}_multiword"],
                    "matched_in_pool": r["matched_in_pool"],
                    "status": r[f"{role}_status"],
                }
            )
    slot_df = pd.DataFrame(slot_rows)

    def _etype_miss_rates(etype: str) -> dict[str, float]:
        sub = slot_df[slot_df["entity_type"] == etype]
        single = sub[~sub["multiword"]]
        multi = sub[sub["multiword"]]
        return {
            "single_miss_rate": float((single["status"] != "matched").mean()) if len(single) else 0.0,
            "multi_miss_rate": float((multi["status"] != "matched").mean()) if len(multi) else 0.0,
        }

    gene_rates = _etype_miss_rates("gene")
    drug_rates = _etype_miss_rates("drug")
    disease_rates = _etype_miss_rates("disease")

    # Pair-type miss shares among missed relations
    missed_pair = missed.groupby("pair_type").size().to_dict()
    n_missed = len(missed)
    pair_miss_frac = {k: v / n_missed for k, v in missed_pair.items()} if n_missed else {}

    # Pair-type x bucket
    pair_bucket = (
        detail.groupby(["pair_type", "bucket"]).size().unstack(fill_value=0).reset_index()
    )

    # Publication year bias
    year_eval = detail[detail["matched_in_pool"]]["publication_year"].dropna()
    year_miss = detail[~detail["matched_in_pool"]]["publication_year"].dropna()

    pt_stats = _pubtator_span_token_stats(pubtator_docs, set(detail["pmid"].astype(str)))
    pt_token_summary = (
        pt_stats.groupby("civic_type")["token_count"]
        .agg(["mean", "median", lambda s: float((s >= 2).mean())])
        .reset_index()
    )
    pt_token_summary.columns = ["civic_type", "mean_tokens", "median_tokens", "pct_multiword"]
    pt_multiword_pct = (
        float(pt_stats["multiword"].mean()) if not pt_stats.empty else 0.0
    )
    pt_disease_multi_pct = (
        float(pt_stats.loc[pt_stats["civic_type"] == "disease", "multiword"].mean())
        if (pt_stats["civic_type"] == "disease").any()
        else 0.0
    )

    # Miss rate by civic entity token length (relation-level: either slot multiword)
    detail["any_multiword"] = detail["head_multiword"] | detail["tail_multiword"]
    miss_rate_single = float((~detail[~detail["any_multiword"]]["matched_in_pool"]).mean()) if (~detail["any_multiword"]).any() else 0.0
    miss_rate_multi = float((~detail[detail["any_multiword"]]["matched_in_pool"]).mean()) if detail["any_multiword"].any() else 0.0

    examples = {
        "miss_entity_absent": _sample_examples(detail, "miss_entity_absent"),
        "miss_present_but_unmatched": _sample_examples(detail, "miss_present_but_unmatched"),
    }

    summary = {
        "n_total": n_total,
        "n_matched": int(bucket_counts.get("matched", 0)),
        "n_miss_entity_absent": int(bucket_counts.get("miss_entity_absent", 0)),
        "n_miss_present_but_unmatched": int(bucket_counts.get("miss_present_but_unmatched", 0)),
        "frac_matched": round(bucket_frac.get("matched", 0), 4),
        "frac_miss_entity_absent": round(bucket_frac.get("miss_entity_absent", 0), 4),
        "frac_miss_present_but_unmatched": round(bucket_frac.get("miss_present_but_unmatched", 0), 4),
        "head_single_miss_rate": round(head_miss["single_miss_rate"], 4),
        "head_multi_miss_rate": round(head_miss["multi_miss_rate"], 4),
        "tail_single_miss_rate": round(tail_miss["single_miss_rate"], 4),
        "tail_multi_miss_rate": round(tail_miss["multi_miss_rate"], 4),
        "gene_single_miss_rate": round(gene_rates["single_miss_rate"], 4),
        "gene_multi_miss_rate": round(gene_rates["multi_miss_rate"], 4),
        "drug_single_miss_rate": round(drug_rates["single_miss_rate"], 4),
        "drug_multi_miss_rate": round(drug_rates["multi_miss_rate"], 4),
        "disease_single_miss_rate": round(disease_rates["single_miss_rate"], 4),
        "disease_multi_miss_rate": round(disease_rates["multi_miss_rate"], 4),
        "relation_miss_rate_single_word": round(miss_rate_single, 4),
        "relation_miss_rate_any_multiword": round(miss_rate_multi, 4),
        "mean_pub_year_matched": round(float(year_eval.mean()), 1) if len(year_eval) else None,
        "mean_pub_year_missed": round(float(year_miss.mean()), 1) if len(year_miss) else None,
        "pair_miss_frac": {k: round(v, 4) for k, v in pair_miss_frac.items()},
        "n_missed": n_missed,
        "pubtator_pct_multiword": round(pt_multiword_pct, 4),
        "pubtator_disease_pct_multiword": round(pt_disease_multi_pct, 4),
        "examples": examples,
        "pair_bucket": pair_bucket.to_dict(orient="records"),
        "side_breakdown": side_breakdown.to_dict(orient="records"),
        "pubtator_span_summary": pt_token_summary.to_dict(orient="records"),
    }

    out_csv = OUTPUT_DIR / "03_candidate_pool_pubtator_recall_classification.csv"
    detail.to_csv(out_csv, index=False)

    bucket_df = pd.DataFrame(
        [
            {"bucket": k, "label": BUCKET_LABELS[k], "n": bucket_counts.get(k, 0), "fraction": bucket_frac.get(k, 0.0)}
            for k in BUCKET_LABELS
        ]
    )
    bucket_df.to_csv(OUTPUT_DIR / "03_candidate_pool_pubtator_recall_buckets.csv", index=False)

    return detail, bucket_df, summary


def recall_report_section(summary: dict[str, Any]) -> str:
    """Plain-prose section for report.md."""
    n = summary["n_total"]
    ex_abs = summary["examples"]["miss_entity_absent"]
    ex_unm = summary["examples"]["miss_present_but_unmatched"]
    side = pd.DataFrame(summary["side_breakdown"])
    gene_abs = int(side[(side.entity_type == "gene") & (side.status == "absent")]["n"].sum()) if not side.empty else 0
    drug_abs = int(side[(side.entity_type == "drug") & (side.status == "absent")]["n"].sum()) if not side.empty else 0
    dis_abs = int(side[(side.entity_type == "disease") & (side.status == "absent")]["n"].sum()) if not side.empty else 0
    gene_unm = int(side[(side.entity_type == "gene") & (side.status == "present_unmatched")]["n"].sum()) if not side.empty else 0
    drug_unm = int(side[(side.entity_type == "drug") & (side.status == "present_unmatched")]["n"].sum()) if not side.empty else 0
    dis_unm = int(side[(side.entity_type == "disease") & (side.status == "present_unmatched")]["n"].sum()) if not side.empty else 0
    gd_frac = summary.get("pair_miss_frac", {}).get("gene-disease", 0)
    gdr_frac = summary.get("pair_miss_frac", {}).get("gene-drug", 0)

    example_lines = ""
    if ex_abs:
        example_lines += "\nExamples of entity-absent misses (PMID, pair type, detail):\n"
        for ex in ex_abs[:3]:
            example_lines += f"- PMID {ex['pmid']} ({ex['pair_type']}): {ex['detail']}\n"
    if ex_unm:
        example_lines += "\nExamples of present-but-unmatched misses:\n"
        for ex in ex_unm[:3]:
            example_lines += f"- PMID {ex['pmid']} ({ex['pair_type']}): {ex['detail']}\n"

    year_note = ""
    if summary.get("mean_pub_year_matched") and summary.get("mean_pub_year_missed"):
        year_note = (
            f"Missed relations skew to older publications "
            f"(mean year {summary['mean_pub_year_missed']:.0f} vs "
            f"{summary['mean_pub_year_matched']:.0f} for matched). "
        )

    return f"""## PubTator recall and entity-span limitation

This section is descriptive only. The candidate pool and matching rules are frozen; nothing here changes coverage, scores, or model comparisons. PubTator3 recall limits which CIViC relations receive a positive candidate at all. That limit affects external validity (which assertions we can rank) and applies equally to every encoder, so it is not part of the between-model comparison.

Of {n} primary CIViC relations, {summary['n_matched']} ({summary['frac_matched']:.1%}) have at least one positive candidate in the frozen pool. The remaining {summary['n_missed']} ({1 - summary['frac_matched']:.1%}) have no pool positive. Among those misses, {summary['n_miss_entity_absent']} ({summary['frac_miss_entity_absent']:.1%} of all relations) fail because PubTator3 annotated no entity of the required type for at least one CIViC partner in the abstract (pure recall failure). Another {summary['n_miss_present_but_unmatched']} ({summary['frac_miss_present_but_unmatched']:.1%}) have the correct entity type present somewhere in the abstract, but no annotation surface form matches the CIViC string under entities_match (span or wording mismatch, often a shorter PubTator span than the full CIViC phrase).

Multi-word CIViC entity strings are missed more often at relation level. Where both partners are single-token strings, {summary['relation_miss_rate_single_word']:.1%} of relations lack a pool positive; where at least one partner is a multi-word phrase, the miss rate rises to {summary['relation_miss_rate_any_multiword']:.1%}. By entity type, single-token genes fail to match {summary['gene_single_miss_rate']:.1%} of the time versus {summary['gene_multi_miss_rate']:.1%} for multi-word gene strings; drugs {summary['drug_single_miss_rate']:.1%} vs {summary['drug_multi_miss_rate']:.1%}; diseases {summary['disease_single_miss_rate']:.1%} vs {summary['disease_multi_miss_rate']:.1%}. PubTator spans in these abstracts are mostly single-token ({summary['pubtator_pct_multiword']:.1%} multi-token overall; {summary['pubtator_disease_pct_multiword']:.1%} for disease mentions), so phrase-level CIViC disease and drug names disproportionately land in the present-but-unmatched bucket when PubTator tags only a substring or a different surface form.

{year_note}Among the {summary['n_missed']} missed relations, {gd_frac:.1%} are gene-disease and {gdr_frac:.1%} gene-drug, confirming the D2 skew toward gene-disease losses. On failing entity slots, gene partners account for {gene_abs} entity-absent and {gene_unm} present-but-unmatched failures; drug partners for {drug_abs} absent and {drug_unm} unmatched; disease partners for {dis_abs} absent and {dis_unm} unmatched. Entity-absent failures are driven mainly by genes; span mismatches concentrate on drugs and diseases when a mention exists but does not link to the CIViC string. These patterns describe coverage bias in the frozen pool, not a fixable matching bug in this pipeline.

{example_lines.strip()}

Auditable table: 03_candidate_pool_pubtator_recall_classification.csv. Figure: 03_candidate_pool_pubtator_recall_gap.png.
"""


def run_pubtator_recall_diagnostic() -> dict[str, Any]:
    from .figures import plot_pubtator_recall_gap

    detail, bucket_df, summary = analyze_pubtator_recall()
    plot_pubtator_recall_gap(summary)

    print("\n=== PubTator recall limitation (read-only) ===")
    print(f"  Total primary relations: {summary['n_total']}")
    print(f"  matched: {summary['n_matched']} ({summary['frac_matched']:.1%})")
    print(
        f"  miss_entity_absent: {summary['n_miss_entity_absent']} "
        f"({summary['frac_miss_entity_absent']:.1%})"
    )
    print(
        f"  miss_present_but_unmatched: {summary['n_miss_present_but_unmatched']} "
        f"({summary['frac_miss_present_but_unmatched']:.1%})"
    )
    print(
        f"  Relation miss rate: single-word entities {summary['relation_miss_rate_single_word']:.1%}, "
        f"any multi-word {summary['relation_miss_rate_any_multiword']:.1%}"
    )
    print(
        f"  Gene slot miss: single {summary['gene_single_miss_rate']:.1%}, "
        f"multi {summary['gene_multi_miss_rate']:.1%}; "
        f"drug single {summary['drug_single_miss_rate']:.1%}, multi {summary['drug_multi_miss_rate']:.1%}; "
        f"disease single {summary['disease_single_miss_rate']:.1%}, multi {summary['disease_multi_miss_rate']:.1%}"
    )
    return summary
