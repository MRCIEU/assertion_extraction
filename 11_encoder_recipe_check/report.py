"""Human-readable report for encoder recipe check."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import (
    FALLBACK_LR,
    FALLBACK_RUN_KEY,
    FALLBACK_SEED,
    FALLBACK_WARMUP_RATIO,
    OUTPUT_DIR,
    PRIMARY_SEED,
    REPORT_DIR,
    REPO_ROOT,
)


def _table(df: pd.DataFrame, floats: set[str] | None = None) -> str:
    floats = floats or set()
    if df.empty:
        return "_No data._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in floats or isinstance(v, float):
                cells.append(f"{float(v):.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    degenerate: pd.DataFrame,
    grid_rows: list[dict],
    vs_group: pd.DataFrame,
    warmup_df: pd.DataFrame,
    others_min: float,
    others_max: float,
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "report.md"

    primary = [r for r in grid_rows if r["seed"] == PRIMARY_SEED and not r.get("bad_seed_guard")]
    guard_rows = [r for r in grid_rows if r.get("bad_seed_guard")]
    best_primary = max(primary, key=lambda x: x["benchmark_f1"]) if primary else None
    best_any = max(grid_rows, key=lambda x: x["benchmark_f1"]) if grid_rows else None
    r10_deberta = float(
        vs_group.loc[vs_group["source"] == "round1_mean_old", "benchmark_f1"].iloc[0]
    )

    lines = [
        "# Encoder recipe check: is the fixed Round-1 recipe fair to DeBERTa?",
        "",
        "Round 1 trained every encoder with learning rate 2e-5, no warmup, and validation-F1 "
        "checkpoint selection. That recipe came from a small sweep on three warmup-friendly models. "
        "DeBERTa was not in that sweep. Its Round-1 mean benchmark F1 was far below the other eight encoders. "
        "This diagnostic trains DeBERTa only on a small learning-rate and warmup grid, using the same "
        "training data and the same self-measured BioRED benchmark protocol as Round 1. "
        "No CIViC or knowledge-base evaluation appears anywhere in this step.",
        "",
        "## Step 0: degenerate Round-1 runs",
        "",
    ]

    if degenerate.empty:
        lines.append("_No degenerate runs listed in Round-1 outputs._")
    else:
        for _, d in degenerate.iterrows():
            deb = "yes" if d.get("is_deberta") or d["model_id"] == "deberta_base" else "no"
            lines.append(
                f"- **{d['model_id']}**, seed **{int(d['seed'])}**: validation or benchmark F1 collapsed to zero. "
                f"DeBERTa: **{deb}**."
            )
        n_deb = int((degenerate["model_id"] == "deberta_base").sum())
        lines.append("")
        lines.append(
            f"Both listed runs are DeBERTa ({n_deb} of {len(degenerate)}). "
            "They are identified here only; not re-run in this folder."
        )

    lines.extend(["", "## Step 1: DeBERTa recipe grid", ""])
    grid_df = pd.DataFrame(
        [
            {
                "learning_rate": r["lr"],
                "warmup": r["warmup_label"],
                "seed": r["seed"],
                "best_val_f1_epoch": r["best_epoch_val_f1"],
                "validation_F1_at_best": r["best_val_f1"],
                "benchmark_F1": r["benchmark_f1"],
                "bad_seed_guard": "yes" if r.get("bad_seed_guard") else "no",
            }
            for r in sorted(grid_rows, key=lambda x: (x["run_key"], x["seed"]))
        ]
    )
    lines.append(_table(grid_df, floats={"learning_rate", "validation_F1_at_best", "benchmark_F1"}))

    if guard_rows:
        lines.extend(
            [
                "",
                "**Bad-seed guard:** one or more primary-seed runs looked degenerate, so seeds 43 and 44 "
                "were run for those recipe combinations. See the table rows marked bad_seed_guard = yes.",
            ]
        )

    lines.extend(["", "## Step 2: placement against the eight-encoder Round-1 group", ""])
    lines.append(
        f"The eight non-DeBERTa Round-1 encoder means span benchmark F1 **{others_min:.3f}** "
        f"(BERT-base) to **{others_max:.3f}** (BioLinkBERT-base)."
    )
    lines.append(f"Round-1 DeBERTa mean under the old recipe: **{r10_deberta:.3f}**.")

    if best_primary:
        lines.append(
            f"Best DeBERTa on the primary grid (seed {PRIMARY_SEED}): **{best_primary['benchmark_f1']:.3f}** "
            f"with learning rate {best_primary['lr']:.0e} and warmup **{best_primary['warmup_label']}**."
        )
    if best_any and best_primary and best_any["benchmark_f1"] > best_primary["benchmark_f1"]:
        lines.append(
            f"Best DeBERTa across all grid seeds: **{best_any['benchmark_f1']:.3f}** "
            f"(seed {best_any['seed']}, {best_any['run_key']})."
        )

    if not warmup_df.empty:
        lines.extend(["", "### Warmup contrast at each learning rate (seed 42)", ""])
        lines.append(_table(warmup_df, floats={"benchmark_f1_none", "benchmark_f1_warmup_10pct", "delta_warmup_minus_none"}))

    lines.extend(
        [
            "",
            "## Descriptive reading (no decision)",
            "",
        ]
    )

    if best_primary:
        bp = best_primary["benchmark_f1"]
        if bp >= 0.70:
            zone = "near the lower end of the eight-encoder band (around 0.70 and above, close to BERT-base at 0.725)."
        elif bp >= 0.65:
            zone = "in a middle zone (roughly 0.65 to 0.70), below most encoders but above the old DeBERTa mean."
        else:
            zone = "still low relative to the group (roughly below 0.65), similar to or only modestly above the old 0.554 mean."

        lines.append(
            f"Under the best grid recipe on seed {PRIMARY_SEED}, DeBERTa benchmark F1 is **{bp:.3f}**, "
            f"which sits {zone}"
        )

    if not warmup_df.empty:
        for _, w in warmup_df.iterrows():
            d = w["delta_warmup_minus_none"]
            direction = "higher" if d > 0.02 else ("lower" if d < -0.02 else "similar")
            lines.append(
                f"At learning rate {w['lr']:.0e}, adding 10% linear warmup vs none is **{direction}** "
                f"for benchmark F1 (delta {d:+.3f}); best epoch moves from {int(w['best_epoch_none'])} "
                f"to {int(w['best_epoch_warmup'])}."
            )

    lines.extend(
        [
            "",
            "If warmup lifts DeBERTa toward the other encoders while holding learning rate fixed, "
            "the Round-1 gap may partly reflect recipe mismatch rather than a fixed capability floor. "
            "If all four recipes leave DeBERTa well below the group, a genuine floor or instability remains plausible. "
            "**Whether to re-run the full Round-1 matrix or rewrite the shared recipe is left to the investigator.**",
            "",
            "### Figures",
            "",
            "- Validation curves: `figures/11_encoder_recipe_check/11_val_curves_grid.png`",
            "- Encoder benchmark strip: `figures/11_encoder_recipe_check/11_encoder_benchmark_strip.png`",
            "",
            "### Optional fallback (not run automatically)",
            "",
            f"If every grid point stays degenerate or below ~0.65, you may launch one extra run: "
            f"learning rate {FALLBACK_LR:.0e}, warmup 10%, seed {FALLBACK_SEED}.",
            "",
            "```bash",
            f"source ~/miniforge3/etc/profile.d/conda.sh && conda activate hf-hpc",
            f"export REPO={REPO_ROOT}",
            f"export OUTPUT_ROOT=${{REPO}}/../projects/project_1",
            f"cd ${{REPO}}/11_encoder_recipe_check",
            f"python run.py --train-fallback-only",
            "```",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report -> {path}")
    return path
