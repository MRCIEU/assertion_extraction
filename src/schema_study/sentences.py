"""Deterministic sentence boundaries for title and abstract text.

The splitter is rule-based. It does not load a statistical or neural model.
A sentence boundary is a period, question mark, or exclamation mark that is
followed by a sentence start, and that is not a decimal point, an initial, or
a listed biomedical abbreviation. Title and abstract are segmented separately,
so the stored title–abstract separator is not placed inside a sentence.
Offsets use the joined document coordinate system. The original characters
are not rewritten.
"""

from __future__ import annotations

SENTENCE_SEGMENTATION_VERSION = "abbreviation_decimal_v1"
BIOMEDICAL_SEGMENTER_CANDIDATE = {
    "name": "scispacy_en_core_sci_sm",
    "status": "not_executed",
    "reason": "The segmenter depends on a neural model asset. That asset was not downloaded, and neural inference is not part of this preparation.",
    "agreement_is_not_correctness": True,
}
TERMINAL_PUNCTUATION_VERSION = "terminal_punctuation_v1"

ABBREVIATIONS = frozenset(
    {
        "dr", "mr", "mrs", "ms", "prof", "jr", "sr", "vs", "etc", "al", "fig", "figs",
        "eq", "eqs", "ref", "refs", "no", "nos", "vol", "pp", "ca", "cf", "approx",
        "min", "max", "spp", "sp", "var", "st", "gen", "inc", "ltd", "co", "dept",
        "univ", "assoc", "natl", "suppl", "ed", "eds", "trans", "rev", "viz", "esp",
    }
)
NEIGHBOUR_SENTENCES = 1


def segment_document(document: dict) -> list[dict]:
    """Return sentence records for one prepared document."""
    text = document["text"]
    title_end = int(document["title_char_end"])
    abstract_start = int(document["abstract_char_start"])
    separator = document["separator"]
    if text[title_end:abstract_start] != separator:
        raise ValueError(f"{document['document_id']} separator is not at the recorded position")
    rows = []
    rows.extend(_section_sentences(document, "title", text[:title_end], 0))
    rows.extend(_section_sentences(document, "abstract", text[abstract_start:], abstract_start))
    for row in rows:
        if text[row["char_start"]:row["char_end"]] != row["text"]:
            raise ValueError(f"Sentence {row['sentence_id']} is not recoverable from its offsets")
    return rows


def _section_sentences(document: dict, section: str, text: str, origin: int) -> list[dict]:
    spans = _boundaries(text)
    rows = []
    for index, (start, end) in enumerate(spans):
        rows.append(
            {
                "sentence_id": f"{document['document_id']}:{section}:{index:04d}",
                "document_id": str(document["document_id"]),
                "split": document["split"],
                "section": section,
                "char_start": origin + start,
                "char_end": origin + end,
                "text": text[start:end],
            }
        )
    return rows


def _boundaries(text: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    index = 0
    length = len(text)
    while index < length:
        if text[index] in ".!?" and _is_boundary(text, index):
            end = index + 1
            if end > start:
                spans.append((start, end))
            next_start = end
            while next_start < length and text[next_start] in " \t":
                next_start += 1
            start = next_start
            index = next_start
            continue
        index += 1
    if start < length:
        spans.append((start, length))
    return [(left, right) for left, right in spans if left < right]


def _is_boundary(text: str, index: int) -> bool:
    character = text[index]
    if character == "." and index > 0 and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit():
        return False
    if character == "." and index + 1 < len(text) and (text[index + 1].isalpha() or text[index + 1] == "."):
        return False
    token = _token_before(text, index)
    if character == "." and (token.lower() in ABBREVIATIONS or (len(token) == 1 and token.isalpha())):
        return False
    following = _next_significant(text, index)
    if following is None:
        return True
    nxt = text[following]
    if nxt.islower():
        return False
    return nxt.isupper() or nxt.isdigit() or nxt in "(\"'"


def _token_before(text: str, index: int) -> str:
    end = index
    while end > 0 and text[end - 1] in ".)":
        if text[end - 1] == "." and end - 1 != index:
            break
        end -= 1
    start = end
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in "-"):
        start -= 1
    return text[start:end]


def _next_significant(text: str, index: int) -> int | None:
    cursor = index + 1
    while cursor < len(text) and text[cursor] in " \t\"')]}":
        cursor += 1
    if cursor >= len(text):
        return None
    return cursor


def segmentation_diagnostics(text: str) -> dict:
    """Count boundaries that the rules deliberately do not make."""
    abbreviations = 0
    initials = 0
    decimals = 0
    index = 0
    while index < len(text):
        if text[index] == "." and index > 0 and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit():
            decimals += 1
        if text[index] in ".!?" and not _is_boundary(text, index):
            token = _token_before(text, index)
            following = _next_significant(text, index)
            if following is not None and text[following].isupper():
                if token.lower() in ABBREVIATIONS:
                    abbreviations += 1
                elif len(token) == 1 and token.isalpha():
                    initials += 1
        index += 1
    return {
        "decimal_points": decimals,
        "abbreviation_before_capital": abbreviations,
        "initial_before_capital": initials,
    }


def terminal_punctuation_sentences(document: dict) -> list[dict]:
    """Split on terminal punctuation without the abbreviation list.

    Decimal points remain inside a sentence. This is a comparison method, not
    a replacement for the recorded sentence version.
    """
    text = document["text"]
    title_end = int(document["title_char_end"])
    abstract_start = int(document["abstract_char_start"])
    rows = []
    rows.extend(_terminal_section(document, "title", text[:title_end], 0))
    rows.extend(_terminal_section(document, "abstract", text[abstract_start:], abstract_start))
    return rows


def _terminal_section(document: dict, section: str, text: str, origin: int) -> list[dict]:
    spans = []
    start = 0
    index = 0
    while index < len(text):
        if text[index] in ".!?" and _terminal_boundary(text, index):
            spans.append((start, index + 1))
            next_start = index + 1
            while next_start < len(text) and text[next_start] in " \t":
                next_start += 1
            start = next_start
            index = next_start
            continue
        index += 1
    if start < len(text):
        spans.append((start, len(text)))
    rows = []
    for position, (left, right) in enumerate(span for span in spans if span[0] < span[1]):
        rows.append(
            {
                "sentence_id": f"{document['document_id']}:{section}:{position:04d}:{TERMINAL_PUNCTUATION_VERSION}",
                "document_id": str(document["document_id"]),
                "split": document["split"],
                "section": section,
                "char_start": origin + left,
                "char_end": origin + right,
                "text": text[left:right],
                "segmentation_version": TERMINAL_PUNCTUATION_VERSION,
            }
        )
    return rows


def _terminal_boundary(text: str, index: int) -> bool:
    if text[index] == "." and index > 0 and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit():
        return False
    following = _next_significant(text, index)
    if following is None:
        return True
    nxt = text[following]
    return nxt.isupper() or nxt.isdigit() or nxt in "(\"'["


def mention_sentences(mention: dict, sentences: list[dict]) -> list[str]:
    """Sentence identifiers whose spans overlap a mention."""
    start = int(mention["char_start"])
    end = int(mention["char_end"])
    return [
        row["sentence_id"]
        for row in sentences
        if start < int(row["char_end"]) and end > int(row["char_start"])
    ]
