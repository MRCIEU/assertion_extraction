#!/usr/bin/env python3.11
"""RQ3 exploratory encoder × KB-metric interaction analysis.

This is explicitly *post-lock exploratory* (Appendix B.25), not part of the
confirmatory H1-H7/FDR tier.  It strengthens RQ3 by asking whether the KB
surfacing audit metric changes encoder conclusions, rather than only reporting
three separate KB columns descriptively.

Model fitted on Phase B full-FT main rows only:

    KB_value ~ encoder + schedule + kb_metric + encoder:kb_metric

where schedule is a blocking/nuisance factor.  The reported SS shares are
partial sums of squares from nested least-squares fits:

    SS(term | all other terms) = SSE(reduced without term) - SSE(full)

This keeps the script dependency-light and makes the interaction effect an
effect-size audit rather than a new confirmatory hypothesis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # scipy is present in the experiment environment; keep graceful fallback.
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


MAIN_ENCODERS = ("PB", "BL", "PL")
SCHEDULES = ("T1B", "T1F", "T2")
METRICS = (
    ("kb_hit_A_setvalued", "KB_hit_A"),
    ("kb_pmass_B_setvalued", "KB_pmass_B"),
    ("kb_auc_C_setvalued", "KB_auc_C"),
)


@dataclass(frozen=True)
class LongRow:
    run_id: str
    encoder: str
    schedule: str
    seed: int
    metric: str
    value: float


def _float_or_none(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except Exception:
        return None
    return f if math.isfinite(f) else None


def load_long_rows(path: Path, include_rb_profile: bool = False) -> tuple[list[LongRow], list[dict[str, Any]]]:
    """Return long-format rows for main FT cells plus optional RB profile rows."""
    long_rows: list[LongRow] = []
    rb_profile_rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("update") != "FT":
                continue
            encoder = row.get("encoder", "")
            if encoder not in MAIN_ENCODERS and encoder != "RB":
                continue
            schedule = row.get("schedule", "")
            if schedule not in SCHEDULES:
                continue
            try:
                seed = int(row.get("seed", ""))
            except Exception:
                continue
            if encoder == "RB":
                if include_rb_profile:
                    rb_profile_rows.append(row)
                continue
            for col, label in METRICS:
                val = _float_or_none(row.get(col))
                if val is None:
                    continue
                long_rows.append(LongRow(
                    run_id=row.get("run_id", ""),
                    encoder=encoder,
                    schedule=schedule,
                    seed=seed,
                    metric=label,
                    value=val,
                ))
    return long_rows, rb_profile_rows


def _dummy(values: list[str], levels: tuple[str, ...], prefix: str) -> tuple[np.ndarray, list[str]]:
    """Reference-coded dummy matrix excluding the first level."""
    cols: list[np.ndarray] = []
    names: list[str] = []
    for level in levels[1:]:
        cols.append(np.array([1.0 if v == level else 0.0 for v in values]))
        names.append(f"{prefix}[{level}]")
    if not cols:
        return np.empty((len(values), 0)), names
    return np.column_stack(cols), names


def design_matrix(rows: list[LongRow]) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    n = len(rows)
    enc_values = [r.encoder for r in rows]
    sched_values = [r.schedule for r in rows]
    metric_values = [r.metric for r in rows]

    intercept = np.ones((n, 1))
    enc_x, enc_names = _dummy(enc_values, MAIN_ENCODERS, "encoder")
    sched_x, sched_names = _dummy(sched_values, SCHEDULES, "schedule")
    metric_levels = tuple(label for _col, label in METRICS)
    metric_x, metric_names = _dummy(metric_values, metric_levels, "kb_metric")

    blocks = [intercept, enc_x, sched_x, metric_x]
    names = ["intercept"] + enc_names + sched_names + metric_names
    term_cols: dict[str, list[int]] = {
        "encoder": list(range(1, 1 + len(enc_names))),
        "schedule_block": list(range(1 + len(enc_names), 1 + len(enc_names) + len(sched_names))),
        "kb_metric": list(range(1 + len(enc_names) + len(sched_names),
                                1 + len(enc_names) + len(sched_names) + len(metric_names))),
    }

    interaction_cols: list[np.ndarray] = []
    interaction_names: list[str] = []
    for enc_i, enc_name in enumerate(enc_names):
        for met_i, met_name in enumerate(metric_names):
            interaction_cols.append(enc_x[:, enc_i] * metric_x[:, met_i])
            interaction_names.append(f"{enc_name}:{met_name}")
    if interaction_cols:
        start = len(names)
        blocks.append(np.column_stack(interaction_cols))
        names.extend(interaction_names)
        term_cols["encoder_x_kb_metric"] = list(range(start, start + len(interaction_names)))
    else:
        term_cols["encoder_x_kb_metric"] = []

    x = np.column_stack(blocks)
    return x, names, term_cols


def fit_sse(x: np.ndarray, y: np.ndarray) -> tuple[float, int, int]:
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    rank = int(np.linalg.matrix_rank(x))
    return float(np.dot(resid, resid)), rank, len(y) - rank


def partial_ss_table(rows: list[LongRow]) -> dict[str, Any]:
    y = np.array([r.value for r in rows], dtype=float)
    x, names, term_cols = design_matrix(rows)
    sse_full, rank_full, df_resid = fit_sse(x, y)
    ss_total = float(np.dot(y - y.mean(), y - y.mean()))

    terms: dict[str, Any] = {}
    for term, cols in term_cols.items():
        keep = [i for i in range(x.shape[1]) if i not in set(cols)]
        sse_reduced, _rank_reduced, _df_reduced = fit_sse(x[:, keep], y)
        ss = max(0.0, sse_reduced - sse_full)
        df = len(cols)
        ms = ss / df if df else float("nan")
        mse = sse_full / df_resid if df_resid > 0 else float("nan")
        f_stat = ms / mse if mse and math.isfinite(mse) and mse > 0 else None
        p_value = (
            float(scipy_stats.f.sf(f_stat, df, df_resid))
            if scipy_stats is not None and f_stat is not None and df > 0 and df_resid > 0
            else None
        )
        terms[term] = {
            "df": df,
            "partial_ss": ss,
            "partial_ss_share_of_corrected_total": ss / ss_total if ss_total else None,
            "F": f_stat,
            "p_value_descriptive": p_value,
        }

    return {
        "n_long_rows": len(rows),
        "n_original_runs": len({r.run_id for r in rows}),
        "model": "KB_value ~ encoder + schedule + kb_metric + encoder:kb_metric",
        "note": (
            "Exploratory partial-SS audit. schedule is a nuisance/blocking factor. "
            "p-values are descriptive and not included in the confirmatory FDR tier."
        ),
        "columns": names,
        "rank_full": rank_full,
        "df_residual": df_resid,
        "sse_full": sse_full,
        "corrected_ss_total": ss_total,
        "terms": terms,
    }


def mean_profile(rows: list[LongRow]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_encoder: dict[str, list[LongRow]] = defaultdict(list)
    for r in rows:
        grouped[(r.encoder, r.metric)].append(r.value)
        by_encoder[r.encoder].append(r)

    profiles: dict[str, Any] = {}
    for enc in MAIN_ENCODERS:
        metric_stats: dict[str, Any] = {}
        means: dict[str, float] = {}
        for _col, metric in METRICS:
            vals = grouped.get((enc, metric), [])
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
            mean = float(arr.mean())
            means[metric] = mean
            metric_stats[metric] = {
                "n": len(vals),
                "mean": mean,
                "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "se": se,
                "ci95_normal": [mean - 1.96 * se, mean + 1.96 * se],
            }
        if means:
            ranking = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
            profiles[enc] = {
                "metrics": metric_stats,
                "metric_ranking_high_to_low": ranking,
                "metric_spread_max_minus_min": ranking[0][1] - ranking[-1][1],
            }
    return profiles


def schedule_profile(rows: list[LongRow]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        grouped[(r.encoder, r.schedule, r.metric)].append(r.value)
    out: dict[str, Any] = {}
    for enc in MAIN_ENCODERS:
        out[enc] = {}
        for sched in SCHEDULES:
            out[enc][sched] = {}
            for _col, metric in METRICS:
                vals = grouped.get((enc, sched, metric), [])
                if not vals:
                    continue
                arr = np.array(vals, dtype=float)
                out[enc][sched][metric] = {
                    "n": len(vals),
                    "mean": float(arr.mean()),
                    "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                }
    return out


def analyze(input_csv: Path) -> dict[str, Any]:
    rows, rb_profile_rows = load_long_rows(input_csv, include_rb_profile=True)
    if not rows:
        raise SystemExit(f"No Phase B FT main rows found in {input_csv}")
    return {
        "analysis": "RQ3 exploratory encoder × KB-metric interaction",
        "status": "exploratory_post_lock_not_confirmatory",
        "input_csv": str(input_csv),
        "main_encoders": list(MAIN_ENCODERS),
        "kb_metrics": [label for _col, label in METRICS],
        "n_long_rows": len(rows),
        "n_main_runs": len({r.run_id for r in rows}),
        "n_rb_reference_rows_excluded_from_anova": len(rb_profile_rows),
        "anova_partial_ss": partial_ss_table(rows),
        "encoder_metric_profiles": mean_profile(rows),
        "encoder_schedule_metric_profiles": schedule_profile(rows),
        "interpretation_guardrail": (
            "Use the encoder_x_kb_metric partial-SS share to assess whether audit "
            "formulation changes encoder-level KB conclusions. In the observed run, "
            "this interaction is weak; report the result as exploratory evidence "
            "against a strong encoder × audit-formulation interaction, not as a "
            "confirmatory test."
        ),
    }


def _fmt_float(v: Any, digits: int = 4) -> str:
    if v is None:
        return "NA"
    try:
        f = float(v)
    except Exception:
        return str(v)
    return f"{f:.{digits}f}"


def render_markdown(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# RQ3 Exploratory Encoder × KB-Metric Interaction")
    lines.append("")
    lines.append(f"Input: `{result['input_csv']}`")
    lines.append("")
    lines.append(
        f"Rows: {result['n_main_runs']} Phase B FT main runs × "
        f"{len(result['kb_metrics'])} KB metrics = {result['n_long_rows']} long rows."
    )
    lines.append("")
    lines.append("## Partial-SS Audit")
    lines.append("")
    lines.append("| Term | df | partial SS share | F | descriptive p |")
    lines.append("|---|---:|---:|---:|---:|")
    for term, d in result["anova_partial_ss"]["terms"].items():
        lines.append(
            f"| {term} | {d['df']} | "
            f"{_fmt_float(d['partial_ss_share_of_corrected_total'])} | "
            f"{_fmt_float(d['F'])} | {_fmt_float(d['p_value_descriptive'], 6)} |"
        )
    lines.append("")
    lines.append("> Exploratory only: p-values are descriptive and not part of the confirmatory FDR tier.")
    lines.append("")
    lines.append("## Encoder × KB Metric Means")
    lines.append("")
    lines.append("| Encoder | KB_hit_A | KB_pmass_B | KB_auc_C | metric spread | ranking |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for enc, prof in result["encoder_metric_profiles"].items():
        metrics = prof["metrics"]
        ranking = " > ".join(k for k, _v in prof["metric_ranking_high_to_low"])
        lines.append(
            f"| {enc} | "
            f"{_fmt_float(metrics.get('KB_hit_A', {}).get('mean'))} | "
            f"{_fmt_float(metrics.get('KB_pmass_B', {}).get('mean'))} | "
            f"{_fmt_float(metrics.get('KB_auc_C', {}).get('mean'))} | "
            f"{_fmt_float(prof['metric_spread_max_minus_min'])} | {ranking} |"
        )
    lines.append("")
    lines.append("## Interpretation Guardrail")
    lines.append("")
    lines.append(result["interpretation_guardrail"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    result = analyze(args.input)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2))
    args.out_md.write_text(render_markdown(result))

    interaction = result["anova_partial_ss"]["terms"]["encoder_x_kb_metric"]
    print(f"RQ3 exploratory rows: {result['n_long_rows']}")
    print(
        "encoder_x_kb_metric partial SS share: "
        f"{interaction['partial_ss_share_of_corrected_total']:.4f}"
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
