"""Coverage-gap diagnostics: variant root-cause (D1) and systematic-loss check (D2)."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from .config import CIVIC_EVIDENCE_JSON, OUTPUT_DIR, SAMPLING_SEED
from .matching import entities_match, normalize_text
from .parse import entities_by_type, parse_entities
from .pool_builder import _find_matching_entity

VARIANT_ROOT_CAUSE_LABELS = {
    "a_no_variant_annotated": "(a) PubTator3 annotated no variant in abstract",
    "b_format_mismatch": "(b) PubTator3 annotated variant(s) but form does not match CIViC string",
    "c_matching_bug": "(c) Matching-code bug (PubTator3 variant present and should link)",
}


def _raw_variant_annotations(doc: dict[str, Any] | None) -> list[dict[str, Any]]:
    """All Variant/Mutation annotations from biocjson (tmVar3 output), unfiltered."""
    if not doc:
        return []
    out: list[dict[str, Any]] = []
    for passage in doc.get("passages") or []:
        for ann in passage.get("annotations") or []:
            infons = ann.get("infons") or {}
            if infons.get("type") not in ("Variant", "Mutation"):
                continue
            out.append(
                {
                    "text": (ann.get("text") or "").strip(),
                    "identifier": infons.get("identifier") or infons.get("normalized_id"),
                    "pubtator_type": infons.get("type"),
                }
            )
    return out


def classify_variant_root_cause(
    civic_variant: str,
    doc: dict[str, Any] | None,
) -> tuple[str, list[str], bool]:
    """
    Classify a variant positive into (a), (b), or (c).
    Returns root_cause_code, pubtator_variant_texts, matched_by_code.
    """
    parsed = parse_entities(doc) if doc else []
    by_type = entities_by_type(parsed)
    variant_entities = by_type.get("variant", [])
    raw = _raw_variant_annotations(doc)
    raw_texts = sorted({r["text"] for r in raw if r["text"]})

    matched = _find_matching_entity(civic_variant, variant_entities, "variant") is not None
    if matched:
        return "c_matching_bug", raw_texts, True

    if not raw:
        return "a_no_variant_annotated", raw_texts, False

    return "b_format_mismatch", raw_texts, False


def analyze_variant_root_cause(
    positives: pd.DataFrame,
    pubtator_docs: dict[str, dict[str, Any]],
    sample_inspect_n: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    D1: Classify all variant-head positives; retain an inspected sample with PubTator texts.
    """
    variant_pos = positives[positives["head_type"] == "variant"].copy()
    rows: list[dict[str, Any]] = []

    for _, r in variant_pos.iterrows():
        pmid = str(r["pmid"])
        doc = pubtator_docs.get(pmid)
        cause, pt_texts, matched = classify_variant_root_cause(str(r["head_entity"]), doc)
        rows.append(
            {
                "target_id": r["target_id"],
                "evidence_id": r.get("evidence_id"),
                "pmid": pmid,
                "pair_type": r["pair_type"],
                "civic_variant": r["head_entity"],
                "tail_entity": r["tail_entity"],
                "root_cause": cause,
                "root_cause_label": VARIANT_ROOT_CAUSE_LABELS[cause],
                "n_pubtator_variants_in_abstract": len(pt_texts),
                "pubtator_variant_texts": "; ".join(pt_texts[:8]) + ("..." if len(pt_texts) > 8 else ""),
                "matched_by_code": matched,
            }
        )

    detail_df = pd.DataFrame(rows)
    breakdown = (
        detail_df.groupby("root_cause")
        .size()
        .reset_index(name="n")
        .assign(label=lambda d: d["root_cause"].map(VARIANT_ROOT_CAUSE_LABELS))
    )
    breakdown["fraction"] = breakdown["n"] / breakdown["n"].sum()

    # Stratified sample for manual inspection table in report
    inspect_parts = []
    for cause in ["a_no_variant_annotated", "b_format_mismatch", "c_matching_bug"]:
        sub = detail_df[detail_df["root_cause"] == cause]
        if sub.empty:
            continue
        n_take = min(len(sub), max(5, sample_inspect_n // 3))
        inspect_parts.append(sub.sample(n=n_take, random_state=SAMPLING_SEED))
    inspect_df = pd.concat(inspect_parts, ignore_index=True) if inspect_parts else detail_df.head(0)

    n_c = int((detail_df["root_cause"] == "c_matching_bug").sum())
    n_total = len(detail_df)
    n_a = int((detail_df["root_cause"] == "a_no_variant_annotated").sum())
    n_b = int((detail_df["root_cause"] == "b_format_mismatch").sum())

    if n_c > 0:
        conclusion = (
            f"{n_c}/{n_total} variant positives are recoverable matching bugs (category c); "
            "matching logic should be fixed before freezing."
        )
        genuine_zero = False
    elif n_total > 0 and n_a + n_b == n_total:
        conclusion = (
            f"Variant coverage 0.0% is genuine: {n_a}/{n_total} ({n_a/n_total:.1%}) have no PubTator3 "
            f"variant annotation; {n_b}/{n_total} ({n_b/n_total:.1%}) have tmVar3 variants whose surface "
            "forms do not match CIViC strings (e.g. CIViC 'Fusion' vs PubTator3 'V600E'). "
            "No category-(c) matching bugs detected. Variant pairs remain descriptive-only and unevaluable."
        )
        genuine_zero = True
    else:
        conclusion = "Insufficient variant positives to classify."
        genuine_zero = False

    summary = {
        "n_variant_positives": n_total,
        "n_category_a": n_a,
        "n_category_b": n_b,
        "n_category_c": n_c,
        "fraction_a": round(n_a / max(n_total, 1), 4),
        "fraction_b": round(n_b / max(n_total, 1), 4),
        "fraction_c": round(n_c / max(n_total, 1), 4),
        "genuine_zero_coverage": genuine_zero,
        "conclusion": conclusion,
        "inspect_sample_n": len(inspect_df),
    }

    detail_df.to_csv(OUTPUT_DIR / "03_candidate_pool_variant_root_cause.csv", index=False)
    breakdown.to_csv(OUTPUT_DIR / "03_candidate_pool_variant_breakdown.csv", index=False)
    inspect_df.to_csv(OUTPUT_DIR / "03_candidate_pool_variant_inspect_sample.csv", index=False)

    return detail_df, breakdown, inspect_df, summary


def _is_gene_symbol(name: str) -> bool:
    """Heuristic: short token without spaces, often all-caps HGNC symbol."""
    s = name.strip()
    if not s or " " in s:
        return False
    if len(s) <= 6 and s.isupper():
        return True
    if len(s) <= 10 and re.match(r"^[A-Z0-9]+$", s):
        return True
    return len(s) <= 4


def _build_pmid_metadata() -> pd.DataFrame:
    records = json.loads(CIVIC_EVIDENCE_JSON.read_text(encoding="utf-8"))
    rows = []
    for item in records:
        source = item.get("source") or {}
        pmid = str(source.get("citationId") or "")
        if not pmid:
            continue
        rows.append(
            {
                "pmid": pmid,
                "publication_year": source.get("publicationYear"),
            }
        )
    meta = pd.DataFrame(rows).drop_duplicates("pmid")
    return meta


def analyze_systematic_loss(
    positives: pd.DataFrame,
    coverage_df: pd.DataFrame,
    pool_classification: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    D2: Compare primary positives with vs without pool coverage on interpretable features.
    When pool_classification is supplied, evaluable = matched_in_pool (ranking coverage).
    """
    primary = coverage_df[coverage_df["scope"] == "primary"].copy()
    primary["pmid"] = primary["pmid"].astype(str)
    if pool_classification is not None:
        primary = primary.merge(
            pool_classification[["target_id", "matched_in_pool", "head_status", "tail_status"]],
            on="target_id",
            how="left",
        )
        primary["matched_in_pool"] = primary["matched_in_pool"].fillna(False).astype(bool)
    pmid_meta = _build_pmid_metadata()
    primary = primary.merge(pmid_meta, on="pmid", how="left")

    gene_freq = (
        primary[primary["head_type"] == "gene"]["head_entity"]
        .value_counts()
        .to_dict()
    )

    slot_rows: list[dict[str, Any]] = []
    for _, r in primary.iterrows():
        if pool_classification is not None:
            evaluable = bool(r["matched_in_pool"])
            if evaluable:
                loss_side = "none"
                missed_entity = None
                missed_type = None
            elif r.get("head_status") == "absent":
                loss_side = "head"
                missed_entity = r["head_entity"]
                missed_type = r["head_type"]
            elif r.get("tail_status") == "absent":
                loss_side = "tail"
                missed_entity = r["tail_entity"]
                missed_type = r["tail_type"]
            else:
                loss_side = "head"
                missed_entity = r["head_entity"]
                missed_type = r["head_type"]
        elif r["both_found"]:
            loss_side = "none"
            missed_entity = None
            missed_type = None
            evaluable = True
        elif not r["head_found"]:
            loss_side = "head"
            missed_entity = r["head_entity"]
            missed_type = r["head_type"]
            evaluable = False
        elif not r["tail_found"]:
            loss_side = "tail"
            missed_entity = r["tail_entity"]
            missed_type = r["tail_type"]
            evaluable = False
        else:
            loss_side = "both"
            missed_entity = r["head_entity"]
            missed_type = r["head_type"]
            evaluable = False

        if pool_classification is None and r["both_found"]:
            evaluable = True
        elif pool_classification is None:
            evaluable = bool(r["both_found"])
        gene_head = str(r["head_entity"]) if r["head_type"] == "gene" else None

        slot_rows.append(
            {
                "target_id": r["target_id"],
                "pmid": r["pmid"],
                "pair_type": r["pair_type"],
                "evaluable": bool(evaluable),
                "loss_side": loss_side,
                "missed_entity": missed_entity,
                "missed_entity_type": missed_type,
                "head_entity": r["head_entity"],
                "head_type": r["head_type"],
                "head_entity_length": len(str(r["head_entity"])),
                "head_is_symbol": _is_gene_symbol(str(r["head_entity"])) if r["head_type"] == "gene" else None,
                "head_corpus_frequency": gene_freq.get(str(r["head_entity"]), 0) if r["head_type"] == "gene" else None,
                "tail_entity_length": len(str(r["tail_entity"])),
                "publication_year": r.get("publication_year"),
            }
        )

    char_df = pd.DataFrame(slot_rows)
    char_df.to_csv(OUTPUT_DIR / "03_candidate_pool_loss_characteristics.csv", index=False)

    evaluable = char_df[char_df["evaluable"]]
    unevaluable = char_df[~char_df["evaluable"]]

    def _summarize(sub: pd.DataFrame, label: str) -> dict[str, Any]:
        gene_sub = sub[sub["head_type"] == "gene"]
        return {
            "group": label,
            "n": len(sub),
            "pct_gene_disease": (sub["pair_type"] == "gene-disease").mean() if len(sub) else 0.0,
            "pct_gene_drug": (sub["pair_type"] == "gene-drug").mean() if len(sub) else 0.0,
            "mean_head_entity_length": gene_sub["head_entity_length"].mean() if len(gene_sub) else None,
            "pct_head_is_symbol": gene_sub["head_is_symbol"].mean() if len(gene_sub) else None,
            "mean_head_corpus_frequency": gene_sub["head_corpus_frequency"].mean() if len(gene_sub) else None,
            "median_head_corpus_frequency": gene_sub["head_corpus_frequency"].median() if len(gene_sub) else None,
            "mean_publication_year": sub["publication_year"].mean() if sub["publication_year"].notna().any() else None,
            "pct_loss_at_head": (sub["loss_side"] == "head").mean() if len(sub) else 0.0,
        }

    comparison = pd.DataFrame([_summarize(evaluable, "evaluable"), _summarize(unevaluable, "unevaluable")])

    # Gene-head-only comparison (where gene is the missed entity or present)
    gene_eval = evaluable[evaluable["head_type"] == "gene"]
    gene_uneval = unevaluable[unevaluable["loss_side"] == "head"]

    gene_comparison = pd.DataFrame(
        [
            {
                "group": "evaluable_gene_heads",
                "n": len(gene_eval),
                "mean_entity_length": gene_eval["head_entity_length"].mean(),
                "pct_symbol": gene_eval["head_is_symbol"].mean(),
                "mean_corpus_frequency": gene_eval["head_corpus_frequency"].mean(),
                "median_corpus_frequency": gene_eval["head_corpus_frequency"].median(),
            },
            {
                "group": "missed_gene_heads",
                "n": len(gene_uneval),
                "mean_entity_length": gene_uneval["head_entity_length"].mean(),
                "pct_symbol": gene_uneval["head_is_symbol"].mean(),
                "mean_corpus_frequency": gene_uneval["head_corpus_frequency"].mean(),
                "median_corpus_frequency": gene_uneval["head_corpus_frequency"].median(),
            },
        ]
    )
    gene_comparison.to_csv(OUTPUT_DIR / "03_candidate_pool_loss_gene_comparison.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "03_candidate_pool_loss_comparison.csv", index=False)

    # Representativeness assessment
    n_uneval = len(unevaluable)
    head_loss_rate = (unevaluable["loss_side"] == "head").mean() if n_uneval else 0.0
    missed_genes = unevaluable[unevaluable["loss_side"] == "head"]
    eval_genes = evaluable[evaluable["head_type"] == "gene"]

    biases: list[str] = []
    if len(missed_genes) and len(eval_genes):
        if missed_genes["head_is_symbol"].mean() < eval_genes["head_is_symbol"].mean() - 0.1:
            biases.append("missed genes are less often short symbols (more full names / long strings)")
        elif missed_genes["head_is_symbol"].mean() > eval_genes["head_is_symbol"].mean() + 0.1:
            biases.append("missed genes are more often short symbols")
        if missed_genes["head_corpus_frequency"].median() < eval_genes["head_corpus_frequency"].median() * 0.5:
            biases.append("missed genes tend to be rarer in the corpus (lower frequency)")
        if (unevaluable["pair_type"] == "gene-disease").mean() > 0.65:
            biases.append("losses concentrate in gene–disease pairs")
        eval_year = evaluable["publication_year"].dropna()
        uneval_year = unevaluable["publication_year"].dropna()
        if len(eval_year) and len(uneval_year):
            if uneval_year.mean() < eval_year.mean() - 5:
                biases.append(
                    f"unevaluable positives skew to older publications "
                    f"(mean {uneval_year.mean():.0f} vs {eval_year.mean():.0f})"
                )

    if not biases:
        implication = (
            "Losses do not strongly concentrate in a single interpretable subgroup; "
            "PubTator3 misses appear largely idiosyncratic per abstract rather than systematically "
            "excluding one gene class. The evaluable primary set is moderately representative, "
            "with the caveat that all losses are PubTator3-recall-dependent."
        )
        systematic = False
    else:
        implication = (
            "Some systematic skew detected: "
            + "; ".join(biases)
            + ". The evaluable set may over-represent entities PubTator3 reliably tags."
        )
        systematic = True

    summary = {
        "n_primary_evaluable": len(evaluable),
        "n_primary_unevaluable": n_uneval,
        "pct_loss_at_gene_head": round(float(head_loss_rate), 4),
        "systematic_bias_detected": systematic,
        "bias_signals": biases,
        "implication": implication,
    }

    return char_df, comparison, gene_comparison, summary
