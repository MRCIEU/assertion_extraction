#!/usr/bin/env python3.11
"""Phase 2B — GPT-4o-mini KB-surface baseline (3 prompt conditions, 162 targets).

Conditions:
  * zero-shot
  * 6-shot (frozen exemplars from kb_surface_pairs.jsonl)
  * 6-shot + rationale instructions (longer output allowance)

Outputs per-condition JSON under phase_d_baselines/outputs/llm_baseline/.
Honours BUDGET_USD via environment (default 1.00); aborts before the next API
call if projected spend would exceed cap.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI

REPO = Path(__file__).resolve().parents[4]
KB_PATH = REPO / "fine_tuning_experiments/schema_exp/eval/inputs/kb_surface_pairs.jsonl"
CIVIC_JSON = (
    REPO / "knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/"
    "civicmine_baseline_case_c.json"
)
OUT_DIR = REPO / "knowledge_grounded_evidence_audit/analysis/phase_d_baselines/outputs/llm_baseline"

from fine_tuning_experiments.schema_exp.eval.schema_expected_label import (  # noqa: E402
    schema_expected_label_set,
)

Condition = Literal["zero_shot", "six_shot", "six_shot_rationale"]

ALLOWED = (
    "DRUG_DISEASE",
    "DRUG_GENE_REGULATION",
    "GENE_DISEASE",
    "VARIANT_DISEASE",
    "ASSOCIATION_GENERAL",
    "__NEGATIVE__",
)


BASE_PROMPT = """You are a biomedical curator labelling drug-gene-variant-disease assertions
extracted from PubMed abstracts. You will be given an abstract and two
pre-identified entities. Your task is to choose ONE family-level relation
label that best describes the relationship the abstract asserts between
the two entities.

Allowed labels (choose exactly one):
- DRUG_DISEASE
- DRUG_GENE_REGULATION
- GENE_DISEASE
- VARIANT_DISEASE
- ASSOCIATION_GENERAL
- __NEGATIVE__

Output STRICT JSON only, no prose, no markdown:
{{"label": "<one of the allowed labels>", "confidence": <float 0 to 1>,
 "rationale": "<one sentence>"}}

---
Abstract:
{abstract}

Entity A: "{ent_a}" (type: {type_a})
Entity B: "{ent_b}" (type: {type_b})

Your label:
"""


def load_kb_targets() -> list[dict[str, Any]]:
    rows = [json.loads(l) for l in KB_PATH.read_text().splitlines() if l.strip()]
    return [r for r in rows if r.get("expected_label") != "VARIANT_GENE"]


def entity_fields(r: dict[str, Any]) -> tuple[str, str, str, str]:
    fam = r["pairing_family"]
    if fam == "gene_drug":
        return r["head_text"], r["tail_text"], "drug", "gene"
    return r["head_text"], r["tail_text"], "variant_alteration", "disease"


def pick_fewshot_examples() -> list[dict[str, Any]]:
    rows = load_kb_targets()
    by_fam: dict[str, list[dict[str, Any]]] = {"gene_drug": [], "variant_disease": []}
    for r in rows:
        fam = r["pairing_family"]
        if fam in by_fam:
            by_fam[fam].append(r)
    # Deterministic: first three per family by target_id sort
    out: list[dict[str, Any]] = []
    for fam in ("gene_drug", "variant_disease"):
        sub = sorted(by_fam[fam], key=lambda x: x["target_id"])[:3]
        out.extend(sub)
    return out


def fewshot_block(examples: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for ex in examples:
        ea, eb, ta, tb = entity_fields(ex)
        abstract = ex["text"].split("[SEP]", 1)[-1].strip()
        parts.append(
            BASE_PROMPT.format(abstract=abstract, ent_a=ea, ent_b=eb, type_a=ta, type_b=tb).strip()
        )
        parts.append('{"label": "%s", "confidence": 0.9, "rationale": "embedded exemplar."}'
                     % ex["expected_label"])
    return "\n\n".join(parts)


def build_user_message(
    r: dict[str, Any], *, condition: Condition, shot_prefix: str,
) -> str:
    ea, eb, ta, tb = entity_fields(r)
    abstract = r["text"].split("[SEP]", 1)[-1].strip()
    core = BASE_PROMPT.format(abstract=abstract, ent_a=ea, ent_b=eb, type_a=ta, type_b=tb)
    if condition == "zero_shot":
        return core
    if condition == "six_shot_rationale":
        core = core.replace(
            '"rationale": "<one sentence>"',
            '"rationale": "<two to four sentences explaining entity alignment>"',
        )
    return shot_prefix + "\n\n" + core


_JSON_RE = re.compile(r"\{[^{}]+\}")


def parse_json_obj(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if m:
            return json.loads(m.group(0))
        raise


@dataclass
class SpendTracker:
    budget: float
    spent: float = 0.0

    def add(self, *, inp: int, out: int) -> None:
        # gpt-4o-mini (2025): $0.15 / 1M in, $0.60 / 1M out
        self.spent += inp * 0.15e-6 + out * 0.60e-6

    def allow(self, est_cost: float) -> bool:
        return (self.spent + est_cost) <= self.budget


def score_row(pred_label: str, r: dict[str, Any]) -> int:
    civic = {
        "expected_pairing_family": r["pairing_family"],
        "heuristic_gold_s2_label": r["expected_label"],
    }
    exp, _ = schema_expected_label_set(civic, "S_pair", "primary", "set_valued")
    return int(pred_label in exp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=("zero_shot", "six_shot", "six_shot_rationale", "all"),
                    default="all")
    ap.add_argument("--budget-usd", type=float, default=float(os.environ.get("BUDGET_USD", "1.0")))
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds between calls (rate limit).")
    ap.add_argument("--max-targets", type=int, default=0, help="Debug cap (0 = all 162).")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY required", file=sys.stderr)
        sys.exit(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    covered_ids = set()
    if CIVIC_JSON.exists():
        data = json.loads(CIVIC_JSON.read_text())
        covered_ids = {c["target_id"] for c in data.get("covered_targets", [])}

    tracker = SpendTracker(budget=args.budget_usd)
    client = OpenAI()
    shot_examples = pick_fewshot_examples()
    shot_prefix = fewshot_block(shot_examples)

    conds: list[Condition]
    if args.condition == "all":
        conds = ["zero_shot", "six_shot", "six_shot_rationale"]
    else:
        conds = [args.condition]  # type: ignore[assignment]

    targets = load_kb_targets()
    if args.max_targets:
        targets = targets[: args.max_targets]

    for cond in conds:
        records: list[dict[str, Any]] = []
        hits162 = 0
        hits41 = 0
        n41 = 0
        for i, r in enumerate(targets):
            est = 0.0015  # conservative cushion per call
            if not tracker.allow(est):
                print(f"HALT: budget cap {tracker.budget:.2f}; spent {tracker.spent:.4f}", file=sys.stderr)
                break
            messages = [
                {"role": "system", "content": "You output valid JSON only for relation classification."},
                {"role": "user", "content": build_user_message(r, condition=cond, shot_prefix=shot_prefix)},
            ]
            max_tokens = 350 if cond == "six_shot_rationale" else 120
            rsp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
            )
            usage = rsp.usage
            if usage:
                tracker.add(inp=int(usage.prompt_tokens or 0), out=int(usage.completion_tokens or 0))
            raw = (rsp.choices[0].message.content or "").strip()
            try:
                obj = parse_json_obj(raw)
                pred = str(obj.get("label", "")).strip()
            except Exception as exc:  # noqa: BLE001
                pred = "__PARSE_ERROR__"
                obj = {"error": str(exc), "raw": raw[:500]}
            ok = pred in ALLOWED
            hit = score_row(pred, r) if ok else 0
            hits162 += hit
            if r["target_id"] in covered_ids:
                n41 += 1
                hits41 += hit
            records.append({
                "target_id": r["target_id"],
                "pairing_family": r["pairing_family"],
                "pmid": r["pmid"],
                "expected_label": r["expected_label"],
                "pred_label": pred,
                "parse_ok": ok,
                "hit_A_sv_argmax": hit,
                "raw_response": raw[:2000],
            })
            if args.sleep:
                time.sleep(args.sleep)

        out_path = OUT_DIR / f"gpt4o_mini_{cond}.json"
        summary = {
            "condition": cond,
            "n_targets": len(records),
            "kb_hit_mean_162": hits162 / len(records) if records else float("nan"),
            "kb_hit_mean_41": hits41 / n41 if n41 else float("nan"),
            "n41_evaluable": n41,
            "usd_spent_estimate": tracker.spent,
            "fewshot_target_ids": [ex["target_id"] for ex in shot_examples],
        }
        out_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
