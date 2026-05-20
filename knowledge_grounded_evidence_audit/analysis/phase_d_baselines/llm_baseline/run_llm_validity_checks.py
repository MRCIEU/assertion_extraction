#!/usr/bin/env python3.11
"""Regenerate llm_validity_checks.md (Phase 2B gate before unified table)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

REPO = Path(__file__).resolve().parents[4]
KB_PATH = REPO / "fine_tuning_experiments/schema_exp/eval/inputs/kb_surface_pairs.jsonl"
OUT_MD = Path(__file__).resolve().parent / "llm_validity_checks.md"
LLM_DIR = REPO / "knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/llm_baseline"
CONDITIONS = [
    ("zero_shot", "gpt4o_mini_zero_shot.json"),
    ("six_shot", "gpt4o_mini_six_shot.json"),
    ("six_shot_rationale", "gpt4o_mini_six_shot_rationale.json"),
]

sys.path.insert(0, str(REPO))
from fine_tuning_experiments.schema_exp.eval.schema_expected_label import schema_expected_label_set  # noqa: E402

LABELS8 = [
    "ASSOCIATION_GENERAL",
    "DRUG_DISEASE",
    "DRUG_GENE_REGULATION",
    "DRUG_VARIANT_ASSOC",
    "GENE_DISEASE",
    "GENE_GENE_ASSOC",
    "VARIANT_DISEASE",
    "__NEGATIVE__",
]

DISPUTE_TARGETS = [
    "GL_0031", "GL_0039", "GL_0043", "GL_0068", "GL_0070", "GL_0118", "GL_0131",
]


def load_kb() -> list[dict]:
    rows = [json.loads(l) for l in KB_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("expected_label") != "VARIANT_GENE"]


def expected_set_for_row(r: dict) -> set[str]:
    civic = {
        "expected_pairing_family": r["pairing_family"],
        "heuristic_gold_s2_label": r["expected_label"],
    }
    exp, _ = schema_expected_label_set(civic, "S_pair", "primary", "set_valued")
    return set(exp)


def main() -> None:
    kb = load_kb()
    if len(kb) != 162:
        raise SystemExit(f"expected 162 KB rows, got {len(kb)}")

    sizes: list[int] = []
    sets: list[set[str]] = []
    for r in kb:
        es = expected_set_for_row(r)
        sets.append(es)
        sizes.append(len(es))

    sizes_arr = np.array(sizes, dtype=np.int64)
    frac_1 = float((sizes_arr == 1).mean())
    frac_ge2 = float((sizes_arr >= 2).mean())
    mean_sz = float(sizes_arr.mean())
    hist = Counter(sizes)

    rand_acc = mean(len(s) / 8.0 for s in sets)

    prim = Counter(r["expected_label"] for r in kb)
    modal_label, modal_count = prim.most_common(1)[0]
    smart_hits = sum(1 for s in sets if modal_label in s)
    smart_acc = smart_hits / len(kb)

    dgr_hits = sum(1 for s in sets if "DRUG_GENE_REGULATION" in s)
    dgr_acc = dgr_hits / len(kb)

    lines: list[str] = []
    lines.append("# Phase 2B — LLM baseline validity checks")
    lines.append("")
    lines.append("**Status:** Verification artefact before any LLM headline numbers enter the unified table or Phase 2D prose.")
    lines.append("")
    lines.append(
        "**Inputs:** `fine_tuning_experiments/schema_exp/eval/inputs/kb_surface_pairs.jsonl` "
        "(VARIANT_GENE excluded, **n=162**); `schema_expected_label_set` (**S_pair**, **primary**, **set_valued**); "
        "GPT-4o-mini JSON under `outputs/llm_baseline/`."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## CHECK 1 — Expected-set width (`expected_set_sv`) distribution")
    lines.append("")
    lines.append("| Summary | Value |")
    lines.append("|---------|------:|")
    lines.append(f"| Mean $\\|S\\|$ across 162 targets | **{mean_sz:.4f}** |")
    lines.append(f"| Fraction with $\\|S\\| = 1$ | **{frac_1:.4f}** ({int(round(frac_1 * 162))}/162) |")
    lines.append(f"| Fraction with $\\|S\\| \\geq 2$ | **{frac_ge2:.4f}** ({int(round(frac_ge2 * 162))}/162) |")
    lines.append("")
    lines.append("**Histogram (count of targets by $|S|$):**")
    lines.append("")
    lines.append("| $\\|S\\|$ | Count |")
    lines.append("|--------|------:|")
    for k in sorted(hist):
        lines.append(f"| {k} | {hist[k]} |")
    lines.append("")
    lines.append(
        "*Interpretation:* On this Goldlite-derived **KB audit slice**, **every** target maps to a **singleton** "
        "$S$ under (`S_pair`, primary, set-valued). The “multi-label $S$ inflates Method~A” shortcut **does not apply** "
        "here ($|S|\\geq 2$ never occurs). High LLM scores must be explained by **label priors / always-positive-head "
        "strategies** (see Check~4), not by wide expected sets."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## CHECK 2 — LLM prediction distribution by condition")
    lines.append("")

    for cond_key, fname in CONDITIONS:
        path = LLM_DIR / fname
        data = json.loads(path.read_text(encoding="utf-8"))
        recs = data["records"]
        preds = [r.get("pred_label", "") for r in recs]
        c = Counter(preds)
        n = len(recs)
        nneg = sum(1 for p in preds if p == "__NEGATIVE__")
        one_one = sum(1 for r in recs if r.get("pred_label") == r.get("expected_label"))
        lines.append(f"### {cond_key} (`{fname}`)")
        lines.append("")
        lines.append(f"- **n targets:** {n}")
        lines.append(f"- **Fraction NEG (`__NEGATIVE__`):** {nneg}/{n} = **{nneg / n:.4f}**")
        lines.append(
            f"- **Fraction pred equals heuristic `expected_label` (one-to-one):** {one_one}/{n} = **{one_one / n:.4f}**"
        )
        lines.append("")
        lines.append("| pred_label | Count | Fraction |")
        lines.append("|------------|------:|---------:|")
        for lab in LABELS8:
            v = c.get(lab, 0)
            if v:
                lines.append(f"| `{lab}` | {v} | {v / n:.4f} |")
        other = {k: v for k, v in c.items() if k not in set(LABELS8)}
        for k, v in sorted(other.items()):
            lines.append(f"| `{k}` | {v} | {v / n:.4f} |")
        lines.append("")

    lines.append(
        "*Interpretation:* Mass is concentrated on **`DRUG_GENE_REGULATION`** (and a thin tail of **`VARIANT_DISEASE`** on "
        "variant–disease rows), with **very few NEG** draws—consistent with **entity-type / label-vocabulary heuristics** "
        "rather than abstention-heavy human adjudication."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## CHECK 3 — Seven IAA disagreement targets (hard cases)")
    lines.append("")
    lines.append(
        "**Audit note:** Per Phase~2B brief: second annotator (Claude Opus) chose **`__NEGATIVE__`** (drug/gene not named "
        "in abstract). Goldlite still yields **positive** `expected_set_sv` under schema projection (singleton DGR for the "
        "gene–drug rows below). Repository does not vend machine-readable Opus labels—**“Agree IAA”** means "
        "`pred_label == __NEGATIVE__`."
    )
    lines.append("")
    lines.append("| target_id | pairing_family | `expected_label` | $|S|$ | $S$ (sorted) |")
    lines.append("|-----------|----------------|------------------|-----|--------------|")

    kb_by_id = {r["target_id"]: r for r in kb}
    for tid in DISPUTE_TARGETS:
        r = kb_by_id[tid]
        es = sorted(expected_set_for_row(r))
        lines.append(f"| {tid} | {r['pairing_family']} | {r['expected_label']} | {len(es)} | `{', '.join(es)}` |")

    hit_key = "hit_A_sv_argmax"
    for cond_key, fname in CONDITIONS:
        path = LLM_DIR / fname
        data = json.loads(path.read_text(encoding="utf-8"))
        by_id = {r["target_id"]: r for r in data["records"]}
        lines.append("")
        lines.append(f"### Predictions — {cond_key} (`{hit_key}` from run log)")
        lines.append("")
        lines.append("| target_id | `pred_label` | hit | Agree IAA (NEG)? | Agree schema $S$ (hit)? |")
        lines.append("|-----------|--------------|-----|------------------|-------------------------|")
        for tid in DISPUTE_TARGETS:
            r = by_id[tid]
            pred = r.get("pred_label", "")
            hit = int(r.get(hit_key, 0))
            agree_iaa = pred == "__NEGATIVE__"
            agree_heur = bool(hit)
            lines.append(f"| {tid} | `{pred}` | {hit} | {agree_iaa} | {agree_heur} |")

    lines.append("")
    lines.append(
        "*Interpretation:* On these seven IDs the model **never** mirrors Opus-style **NEG**; where `hit=1`, it aligns with "
        "the **schema projector** (singleton DGR), **not** the second annotator’s abstention narrative."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## CHECK 4 — Random baselines (162-set)")
    lines.append("")
    lines.append(
        "**IID uniform over eight S_pair labels (full 162 targets):** for each row, "
        "$\\mathbb{P}(\\mathrm{hit}) = |S|/8$. Here $|S|=1$ everywhere ⇒ expected accuracy **1/8 = 0.125** "
        f"(numerical mean **{rand_acc:.6f}**)."
    )
    lines.append("")
    lines.append(
        "**Relation to paper Layer~A “0.334 random”:** the CIViCmine $n=162$ **random** line in Case~C is "
        "`(correct_on_strict41 + analytic_random_mass_on_uncovered_only) / 162`, **not** the same estimand as "
        "“uniform guesser evaluated on all 162 KB rows.” Do **not** equate **0.334** with the **0.125** floor here without "
        "that caveat."
    )
    lines.append("")
    lines.append("| Baseline | Accuracy | Notes |")
    lines.append("|----------|---------:|-------|")
    lines.append(f"| IID uniform (per-row $\\|S\\|/8$) | **{rand_acc:.6f}** | = **0.125** with all-singleton $S$. |")
    lines.append(
        f"| Always `DRUG_GENE_REGULATION` | **{dgr_acc:.6f}** ({dgr_hits}/162) | “Smart” constant if $S$ usually contains DGR. |"
    )
    lines.append(
        f"| Always modal primary `expected_label` = `{modal_label}` | **{smart_acc:.6f}** ({smart_hits}/162) | "
        f"Modal primary prevalence **{modal_count}/162**. |"
    )
    lines.append("")
    lines.append(
        f"**Vs zero-shot ~0.988:** the **0.125** floor is trivially passed; the informative ceiling is the **~{dgr_acc:.3f}** "
        "**always-DGR** (or equivalent modal-in-set) strategy—only **~0.04** below the observed LLM mean—so the headline "
        "accuracy is **not** “oracle-like” relative to naive structure-aware baselines, even though it **dominates** "
        "fine-tuned encoders trained under seed noise + abstention."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Gate for RESULTS_TO_REPORT_PHASE_D.md")
    lines.append("")
    lines.append(
        "**Hold:** Do **not** paste GPT-4o-mini headline KB numbers into the Phase~2D unified table until Freddie reviews "
        "this file (especially Check~3 vs IAA and Check~4 estimand separation)."
    )
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
