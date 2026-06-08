"""Offset-first [E1]/[E2] marker insertion shared by train, benchmark, and CIViC eval."""

from __future__ import annotations

from typing import Any


def bigbio_doc_text(doc: dict[str, Any]) -> str:
    parts: list[str] = []
    for passage in doc.get("passages") or []:
        text = passage.get("text")
        if isinstance(text, list):
            parts.extend(str(t) for t in text if t)
        elif text:
            parts.append(str(text))
    return " ".join(parts).strip()


def bigbio_entity_surface(entity: dict[str, Any]) -> str:
    text = entity.get("text") or [""]
    if isinstance(text, list):
        return str(text[0]) if text else ""
    return str(text)


def bigbio_entity_span(entity: dict[str, Any]) -> tuple[int, int, str] | None:
    """Return (start, end, surface) from a BigBio entity record (first annotated mention)."""
    offsets = entity.get("offsets")
    if not offsets:
        return None
    first = offsets[0]
    if not isinstance(first, (list, tuple)) or len(first) != 2:
        return None
    start, end = int(first[0]), int(first[1])
    surface = bigbio_entity_surface(entity)
    return start, end, surface


def _valid_span(text: str, start: int | None, end: int | None) -> bool:
    if start is None or end is None:
        return False
    try:
        start_i, end_i = int(start), int(end)
    except (TypeError, ValueError):
        return False
    return 0 <= start_i < end_i <= len(text)


def insert_entity_markers(
    text: str,
    *,
    head_start: int | None = None,
    head_end: int | None = None,
    head_surface: str = "",
    tail_start: int | None = None,
    tail_end: int | None = None,
    tail_surface: str = "",
) -> tuple[str, str]:
    """
    Insert [E1] and [E2] around the annotated character spans.

    Returns (marked_text, method) where method is one of:
      offset, fallback_string, fallback_wrap
    """
    head_ok = _valid_span(text, head_start, head_end)
    tail_ok = _valid_span(text, tail_start, tail_end)

    if head_ok and tail_ok:
        hs, he = int(head_start), int(head_end)
        ts, te = int(tail_start), int(tail_end)
        spans = [(hs, he, "E1"), (ts, te, "E2")]
        spans.sort(key=lambda x: x[0], reverse=True)
        out = text
        for start, end, tag in spans:
            open_tag = f"[{tag}]"
            close_tag = f"[/{tag}]"
            out = out[:start] + open_tag + out[start:end] + close_tag + out[end:]
        return out, "offset"

    head = head_surface or ""
    tail = tail_surface or ""
    if head and tail and head in text and tail in text:
        marked = text.replace(head, f"[E1]{head}[/E1]", 1)
        marked = marked.replace(tail, f"[E2]{tail}[/E2]", 1)
        return marked, "fallback_string"

    return f"[E1]{head}[/E1] {text} [E2]{tail}[/E2]", "fallback_wrap"


def format_marked_pair(
    text: str,
    head_entity: dict[str, Any],
    tail_entity: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """
    Mark a training/benchmark pair using BigBio entity offsets.

    Returns (marked_text, method, span_metadata).
    """
    head = bigbio_entity_span(head_entity)
    tail = bigbio_entity_span(tail_entity)
    meta: dict[str, Any] = {}
    if head:
        meta["head_offset"] = head[0]
        meta["head_end"] = head[1]
    if tail:
        meta["tail_offset"] = tail[0]
        meta["tail_end"] = tail[1]

    marked, method = insert_entity_markers(
        text,
        head_start=head[0] if head else None,
        head_end=head[1] if head else None,
        head_surface=head[2] if head else bigbio_entity_surface(head_entity),
        tail_start=tail[0] if tail else None,
        tail_end=tail[1] if tail else None,
        tail_surface=tail[2] if tail else bigbio_entity_surface(tail_entity),
    )
    meta["marker_method"] = method
    return marked, method, meta
