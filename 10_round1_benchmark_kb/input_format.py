"""Entity-marked input for trained relation-presence models."""

from __future__ import annotations

import pandas as pd


def format_eval_input(row: pd.Series) -> str:
    """Match training format: [E1]head[/E1] in abstract [E2]tail[/E2]."""
    text = str(row.get("abstract") or "")
    head = str(row["head_entity"])
    tail = str(row["tail_entity"])
    head_off = row.get("head_offset")
    tail_off = row.get("tail_offset")

    if pd.notna(head_off) and pd.notna(tail_off) and text:
        try:
            hs, he = int(head_off), int(head_off) + len(head)
            ts, te = int(tail_off), int(tail_off) + len(tail)
            if 0 <= hs < he <= len(text) and 0 <= ts < te <= len(text):
                if hs > ts:
                    out = text[:hs] + f"[E1]{text[hs:he]}[/E1]" + text[he:]
                    ts2 = ts + len("[E1]") + len("[/E1]")
                    te2 = te + len("[E1]") + len("[/E1]")
                    out = out[:ts2] + f"[E2]{out[ts2:te2]}[/E2]" + out[te2:]
                    return out
                out = text[:ts] + f"[E2]{text[ts:te]}[/E2]" + text[te:]
                hs2 = hs + len("[E2]") + len("[/E2]")
                he2 = he + len("[E2]") + len("[/E2]")
                out = out[:hs2] + f"[E1]{out[hs2:he2]}[/E1]" + out[he2:]
                return out
        except (TypeError, ValueError):
            pass

    if head in text and tail in text:
        marked = text.replace(head, f"[E1]{head}[/E1]", 1)
        marked = marked.replace(tail, f"[E2]{tail}[/E2]", 1)
        return marked
    return f"[E1]{head}[/E1] {text} [E2]{tail}[/E2]"
