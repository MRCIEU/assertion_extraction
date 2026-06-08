"""Entity-marked input for trained relation-presence models."""

from __future__ import annotations

import pandas as pd

from .marker_insert import insert_entity_markers


def format_eval_input_with_method(row: pd.Series) -> tuple[str, str]:
    """Format CIViC candidate row; returns (marked_text, insertion_method)."""
    text = str(row.get("abstract") or "")
    head = str(row["head_entity"])
    tail = str(row["tail_entity"])
    head_off = row.get("head_offset")
    tail_off = row.get("tail_offset")

    head_start = head_end = tail_start = tail_end = None
    if pd.notna(head_off):
        try:
            head_start = int(head_off)
            head_end = head_start + len(head)
        except (TypeError, ValueError):
            head_start = head_end = None
    if pd.notna(tail_off):
        try:
            tail_start = int(tail_off)
            tail_end = tail_start + len(tail)
        except (TypeError, ValueError):
            tail_start = tail_end = None

    return insert_entity_markers(
        text,
        head_start=head_start,
        head_end=head_end,
        head_surface=head,
        tail_start=tail_start,
        tail_end=tail_end,
        tail_surface=tail,
    )


def format_eval_input(row: pd.Series) -> str:
    """Match training format: [E1]head[/E1] in abstract [E2]tail[/E2]."""
    marked, _method = format_eval_input_with_method(row)
    return marked
