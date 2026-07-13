"""Part 3: qualitative error analysis from folder-11 CIViC scores."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from shared.inference import load_scores_jsonl
from shared.metrics_ranking import _rank_within_abstracts
from shared.models import MODELS

from .config import (
    QUAL_ERROR_CASES_CSV,
    QUAL_ERROR_FLAGGED_CSV,
    QUAL_ERROR_PATTERNS_CSV,
    QUAL_ERROR_SUMMARY_CSV,
    ENRICHED_POOL_CACHE,
    POOL_SIZE_BY_ABSTRACT_CSV,
    PUBMED_RECALL_CSV,
    R11_SCORES_DIR,
    resolve_checkpoint_model_id,
)
from .pool_cache import load_enriched_pool

REPRESENTATIVE_SEED = 42
N_FLAGGED_MANUAL = 24


def _sentence_spans(abstract: str) -> list[tuple[int, int, str]]:
    if not abstract or not isinstance(abstract, str):
        return []
    spans: list[tuple[int, int, str]] = []
    for m in re.finditer(r"[^.!?]+[.!?]?", abstract):
        spans.append((m.start(), m.end(), m.group(0).strip()))
    return spans


def _offset_sentence(abstract: str, offset: float | int | None) -> str:
    if abstract is None or offset is None or (isinstance(offset, float) and np.isnan(offset)):
        return ""
    off = int(offset)
    for start, end, sent in _sentence_spans(abstract):
        if start <= off < end:
            return sent
    return ""


def _entity_sentences(abstract: str, head_off, tail_off) -> str:
    hs = _offset_sentence(abstract, head_off)
    ts = _offset_sentence(abstract, tail_off)
    if hs and ts and hs == ts:
        return hs
    parts = [p for p in (hs, ts) if p]
    return " | ".join(parts)


def _is_multiword(name: str) -> bool:
    return bool(name) and (" " in str(name).strip() or "-" in str(name).strip())


def _abstract_support_label(subset: str) -> str:
    if subset in ("easy_co_sentence", "hard_cross_sentence"):
        return "abstract_supported"
    return "abstract_unsupported"


def _load_merged_scores(seed: int = REPRESENTATIVE_SEED) -> pd.DataFrame:
    """Median score across nine encoders at one seed (stated representative design)."""
    parts: list[pd.DataFrame] = []
    for spec in MODELS:
        path = R11_SCORES_DIR / resolve_checkpoint_model_id(spec.model_id) / f"seed_{seed}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing folder-11 scores: {path}")
        parts.append(load_scores_jsonl(path))
    all_s = pd.concat(parts, ignore_index=True)
    agg = (
        all_s.groupby("candidate_id", as_index=False)
        .agg(
            score=("score", "median"),
            pmid=("pmid", "first"),
            pair_type=("pair_type", "first"),
            head_entity=("head_entity", "first"),
            tail_entity=("tail_entity", "first"),
            head_offset=("head_offset", "first"),
            tail_offset=("tail_offset", "first"),
            abstract=("abstract", "first"),
            label_civic_curated_positive=("label_civic_curated_positive", "first"),
        )
    )
    return agg


def _attach_metadata(scores: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "candidate_id",
        "subset",
        "sentence_distance",
        "head_entity",
        "tail_entity",
        "head_offset",
        "tail_offset",
        "abstract",
    ]
    pool_sub = pool[meta_cols].drop_duplicates("candidate_id")
    out = scores.merge(pool_sub, on="candidate_id", how="left", suffixes=("", "_pool"))
    for col in ("head_entity", "tail_entity", "head_offset", "tail_offset", "abstract"):
        out[col] = out[col].fillna(out.get(f"{col}_pool"))
    return out


def extract_error_cases(seed: int = REPRESENTATIVE_SEED) -> pd.DataFrame:
    scores = _load_merged_scores(seed)
    pool = load_enriched_pool()
    scores = _attach_metadata(scores, pool)

    pool_sizes = pd.read_csv(POOL_SIZE_BY_ABSTRACT_CSV)
    gd_sizes = pool_sizes[pool_sizes["pair_type"] == "gene-disease"][["pmid", "pool_size"]].copy()
    gd_sizes["pmid"] = gd_sizes["pmid"].astype(str)
    pub_years = pd.read_csv(PUBMED_RECALL_CSV)[["pmid", "publication_year"]].drop_duplicates("pmid")
    pub_years["pmid"] = pub_years["pmid"].astype(str)

    ranked = _rank_within_abstracts(scores)
    ranked["pmid"] = ranked["pmid"].astype(str)
    ranked = ranked.merge(gd_sizes, on="pmid", how="left")
    ranked = ranked.merge(pub_years, on="pmid", how="left")

    cases: list[dict] = []

    for pmid, grp in ranked.groupby("pmid", sort=False):
        positives = grp[grp["label_civic_curated_positive"]]
        if positives.empty:
            continue
        worst_pos = positives.loc[positives["rank"].idxmax()]
        cases.append(
            {
                "case_type": "missed_positive",
                "candidate_id": worst_pos["candidate_id"],
                "pmid": pmid,
                "pair_type": worst_pos["pair_type"],
                "rank_in_pool": int(worst_pos["rank"]),
                "pool_size": int(len(grp)),
                "score": float(worst_pos["score"]),
                "sentence_distance": worst_pos.get("sentence_distance"),
                "subset": worst_pos.get("subset", "unknown"),
                "head_entity": worst_pos["head_entity"],
                "tail_entity": worst_pos["tail_entity"],
                "head_multiword": _is_multiword(worst_pos["head_entity"]),
                "tail_multiword": _is_multiword(worst_pos["tail_entity"]),
                "publication_year": worst_pos.get("publication_year"),
                "entity_sentences": _entity_sentences(
                    worst_pos.get("abstract", ""),
                    worst_pos.get("head_offset"),
                    worst_pos.get("tail_offset"),
                ),
                "support_label": _abstract_support_label(str(worst_pos.get("subset", "unknown"))),
            }
        )

        top_neg = grp[~grp["label_civic_curated_positive"]].nsmallest(1, "rank")
        if not top_neg.empty:
            row = top_neg.iloc[0]
            cases.append(
                {
                    "case_type": "false_high",
                    "candidate_id": row["candidate_id"],
                    "pmid": pmid,
                    "pair_type": row["pair_type"],
                    "rank_in_pool": int(row["rank"]),
                    "pool_size": int(len(grp)),
                    "score": float(row["score"]),
                    "sentence_distance": row.get("sentence_distance"),
                    "subset": row.get("subset", "unknown"),
                    "head_entity": row["head_entity"],
                    "tail_entity": row["tail_entity"],
                    "head_multiword": _is_multiword(row["head_entity"]),
                    "tail_multiword": _is_multiword(row["tail_entity"]),
                    "publication_year": row.get("publication_year"),
                    "entity_sentences": _entity_sentences(
                        row.get("abstract", ""),
                        row.get("head_offset"),
                        row.get("tail_offset"),
                    ),
                    "support_label": _abstract_support_label(str(row.get("subset", "unknown"))),
                }
            )

    df = pd.DataFrame(cases)
    df["error_class"] = np.where(
        df["case_type"] == "missed_positive",
        np.where(df["support_label"] == "abstract_supported", "model_error", "abstract_unsupported"),
        "false_high",
    )
    df["flag_for_manual_read"] = False
    return df


def summarize_patterns(missed: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pos = missed[missed["case_type"] == "missed_positive"].copy()
    n = len(pos)
    n_unsup = int((pos["support_label"] == "abstract_unsupported").sum())
    frac_unsup = float(n_unsup / n) if n else np.nan

    genuine = pos[pos["support_label"] == "abstract_supported"].copy()
    patterns: list[dict] = []

    def _rate(col, cond):
        if genuine.empty:
            return np.nan
        return float(cond.mean())

    patterns.extend(
        [
            {
                "pattern": "cross_sentence_hard",
                "rate_in_genuine_errors": _rate(
                    "subset", genuine["subset"] == "hard_cross_sentence"
                ),
                "n_genuine": len(genuine),
            },
            {
                "pattern": "multiword_entity",
                "rate_in_genuine_errors": _rate(
                    "head_multiword",
                    genuine["head_multiword"] | genuine["tail_multiword"],
                ),
                "n_genuine": len(genuine),
            },
            {
                "pattern": "publication_year_before_2010",
                "rate_in_genuine_errors": _rate(
                    "publication_year",
                    genuine["publication_year"].fillna(9999) < 2010,
                ),
                "n_genuine": len(genuine),
            },
            {
                "pattern": "gene_disease_pair",
                "rate_in_genuine_errors": _rate(
                    "pair_type", genuine["pair_type"] == "gene-disease"
                ),
                "n_genuine": len(genuine),
            },
        ]
    )
    pat_df = pd.DataFrame(patterns)

    summary = {
        "n_missed_positives": n,
        "n_abstract_unsupported": n_unsup,
        "frac_abstract_unsupported": frac_unsup,
        "n_genuine_model_errors": int(len(genuine)),
        "representative_seed": REPRESENTATIVE_SEED,
        "score_aggregation": "median across nine encoders at representative seed",
    }
    return pat_df, summary


def flag_manual_sample(genuine: pd.DataFrame, n: int = N_FLAGGED_MANUAL) -> pd.DataFrame:
    if genuine.empty:
        return genuine
    genuine = genuine.copy()
    genuine["stratum"] = genuine["subset"].astype(str)
    per_stratum = max(1, n // genuine["stratum"].nunique())
    flagged = []
    for _, sub in genuine.groupby("stratum"):
        flagged.append(sub.nsmallest(per_stratum, "rank_in_pool"))
    out = pd.concat(flagged).head(n)
    return out


def run_qualitative_errors() -> dict[str, Any]:
    cases = extract_error_cases()
    missed = cases[cases["case_type"] == "missed_positive"]
    patterns, summary = summarize_patterns(cases)
    genuine = missed[missed["support_label"] == "abstract_supported"].copy()
    flagged = flag_manual_sample(genuine)
    cases.loc[cases["candidate_id"].isin(flagged["candidate_id"]), "flag_for_manual_read"] = True

    cases.to_csv(QUAL_ERROR_CASES_CSV, index=False)
    flagged.to_csv(QUAL_ERROR_FLAGGED_CSV, index=False)
    patterns.to_csv(QUAL_ERROR_PATTERNS_CSV, index=False)
    pd.DataFrame([summary]).to_csv(QUAL_ERROR_SUMMARY_CSV, index=False)

    print("\n=== Part 3b: Abstract-unsupported proportion (missed positives) ===")
    print(
        f"  Missed positives (worst-ranked CIViC+ per abstract): {summary['n_missed_positives']}"
    )
    print(
        f"  Abstract-unsupported: {summary['n_abstract_unsupported']} "
        f"({summary['frac_abstract_unsupported']:.1%})"
    )
    print(f"  Genuine model errors (abstract-supported): {summary['n_genuine_model_errors']}")

    print("\n=== Part 3c: Systematic failure modes (genuine errors only) ===")
    for _, r in patterns.iterrows():
        print(f"  {r['pattern']}: {float(r['rate_in_genuine_errors']):.1%} of genuine errors")

    print(f"\n  Flagged for manual reading: {len(flagged)} cases -> {QUAL_ERROR_FLAGGED_CSV.name}")

    return {
        "cases": cases,
        "patterns": patterns,
        "summary": summary,
        "flagged": flagged,
    }
