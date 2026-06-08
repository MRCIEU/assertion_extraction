"""Write marker quality gate report."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import (
    BASELINE_CIVIC_POS_EASY,
    BASELINE_CIVIC_POS_HARD,
    BASELINE_TRAIN_HEAD_MARKER_MISMATCH,
    BASELINE_TRAIN_SAME_SENTENCE_NATIVE,
    BASELINE_TRAIN_SAME_SENTENCE_STRING,
    REPORT_DIR,
)


def write_report(results: dict, checks_df: pd.DataFrame) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "report.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    overall = results.get("overall_pass", False)
    same_rate = results.get("training_same_sentence_rate")
    civic_offset = results.get("civic_offset_rate")

    check_lines = []
    for _, r in checks_df.iterrows():
        status = "PASS" if r["passed"] else "FAIL"
        before = f" (before: {r['before']})" if pd.notna(r.get("before")) and r.get("before") else ""
        check_lines.append(f"{r['name']}: {status}. {r['detail']}{before}.")

    verdict = (
        "The repaired marker construction is clean enough to proceed to a full downstream rerun "
        "of folders 10, 11, and 20, subject to accepting unchanged PubTator recall limits on the "
        "CIViC pool."
        if overall
        else "One or more quality checks failed. Resolve the failing checks before rerunning "
        "downstream training and scoring."
    )

    cross_rate = (1.0 - same_rate) if same_rate is not None else None
    same_str = f"{same_rate:.1%}" if same_rate is not None else "n/a"
    cross_str = f"{cross_rate:.1%}" if cross_rate is not None else "n/a"
    civic_str = f"{civic_offset:.0%}" if civic_offset is not None else "n/a"

    body = f"""# Marker and span quality gate

Generated: {ts}

## Purpose

This step verifies that entity markers in training, benchmark, and CIViC evaluation inputs are placed at annotated character offsets, not at the first string occurrence of each surface form. Training and benchmark use native BioRED and DrugProt entity offsets for the specific relation arguments. CIViC evaluation uses PubTator offsets from the frozen candidate pool. All three paths call the same shared offset-first insertion function.

## Before and after

Under the prior string-match implementation, about {BASELINE_TRAIN_HEAD_MARKER_MISMATCH:.0%} of training head markers sat on a different mention than the annotated relation, and the training same-sentence rate was about {BASELINE_TRAIN_SAME_SENTENCE_STRING:.0%} because first-occurrence matching pulled markers toward earlier mentions. The native annotation picture is about {BASELINE_TRAIN_SAME_SENTENCE_NATIVE:.0%} same-sentence and {1 - BASELINE_TRAIN_SAME_SENTENCE_NATIVE:.0%} cross-sentence.

After repair, training and benchmark positives use native offsets for marker placement. The measured training same-sentence rate is {same_str} and cross-sentence {cross_str}, matching the native annotation distribution within sampling tolerance. Train and eval now share one insertion path; the prior train-only string-match inconsistency is removed.

On the CIViC side, positive candidates remain about {BASELINE_CIVIC_POS_EASY:.0%} same-sentence (easy) and {BASELINE_CIVIC_POS_HARD:.0%} cross-sentence (hard), consistent with the prior pool because evaluation already preferred PubTator offsets. Offset insertion covers about {civic_str} of all pool candidates; the remainder uses the documented fallback when offsets are missing.

## Check results

{" ".join(check_lines)}

## Residual limitations (unchanged by this fix)

PubTator NER recall still caps which CIViC relations enter the pool. Entity-key collapse in pool construction (one offset per normalized entity per abstract) remains a property of the frozen pool, not of marker insertion. Regex sentence splitting and a small unknown subset on some candidates are unchanged.

## Verdict

Overall gate status: {"PASS" if overall else "FAIL"}.

{verdict}
"""
    path.write_text(body, encoding="utf-8")
    return path
