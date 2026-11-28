"""target_anchored_v1 contexts and entity-level views.

The original-text window is shared by every entity view. Each budget check uses
the maximum encoded length across the requested views. Relation labels are not
an input to this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

MARKER_START = {"H": "<<H>>", "T": "<<T>>"}
MARKER_END = {"H": "<</H>>", "T": "<</T>>"}
GAP_MARKER = "<<GAP>>"
STRATEGY = "target_anchored_v1"
OVERLAP_REPRESENTATION = "interleaved_boundary_tags_v1"


def type_token(name: str) -> str:
    return f"<<TYPE:{name}>>"


SPECIAL_MARKERS = [
    "<<H>>",
    "<</H>>",
    "<<T>>",
    "<</T>>",
    "<<GAP>>",
    "<<TYPE:Molecular_Entity>>",
    "<<TYPE:GeneOrGeneProduct>>",
    "<<TYPE:DiseaseOrPhenotypicFeature>>",
    "<<TYPE:ChemicalEntity>>",
    "<<TYPE:SequenceVariant>>",
    "<<TYPE:OrganismTaxon>>",
    "<<TYPE:CellLine>>",
    "<<TYPE:CHEMICAL>>",
    "<<TYPE:GENE>>",
]


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class MentionView:
    mention_id: str
    start: int
    end: int
    role: str
    entity_type: str
    displayed_type: str | None = None
    displayed_by_level: dict | None = None


class TextEncoder(Protocol):
    def tokenize(self, text: str) -> list[TokenSpan]:
        """Original-text tokens with character offsets into `text`."""

    def encoded_length(self, text: str) -> int:
        """Length including model special tokens and any study markers."""


class WhitespaceEncoder:
    """Deterministic encoder for synthetic tests. Not a model tokenizer."""

    def __init__(self, token_weights: dict[str, int] | None = None) -> None:
        self.token_weights = token_weights or {}

    def tokenize(self, text: str) -> list[TokenSpan]:
        return [TokenSpan(match.start(), match.end(), match.group()) for match in re.finditer(r"\S+", text)]

    def encoded_length(self, text: str) -> int:
        residual = text
        count = 2
        for marker in sorted(SPECIAL_MARKERS, key=len, reverse=True):
            found = residual.count(marker)
            count += found * self.token_weights.get(marker, 1)
            residual = residual.replace(marker, " ")
        for token in re.findall(r"\S+", residual):
            count += self.token_weights.get(token, 1)
        return count


def mapped_type(
    entity_type: str,
    level: str,
    merge: dict[str, str] | None = None,
    displayed: str | None = None,
) -> str | None:
    """E0 hides the type. A deeper label replaces it. A missing deeper label keeps the native type."""
    if level == "E0":
        return None
    if displayed:
        return displayed
    if level not in {"E1", "E2", "native"}:
        return entity_type
    return (merge or {}).get(entity_type, entity_type)


def render_marked_text(
    text: str,
    mentions: list[MentionView],
    *,
    level: str,
    merge: dict[str, str] | None = None,
) -> str:
    """Insert role markers without repeating overlapping target text."""
    opens: dict[int, list[MentionView]] = {}
    closes: dict[int, list[MentionView]] = {}
    for mention in mentions:
        opens.setdefault(mention.start, []).append(mention)
        closes.setdefault(mention.end, []).append(mention)
    if not text and not mentions:
        return ""
    positions = sorted(set(opens) | set(closes) | {0, len(text)})
    pieces: list[str] = []
    for index, pos in enumerate(positions):
        for role in ("T", "H"):
            for mention in sorted(closes.get(pos, []), key=lambda item: item.mention_id):
                if mention.role == role:
                    pieces.append(MARKER_END[role])
        for role in ("H", "T"):
            for mention in sorted(opens.get(pos, []), key=lambda item: item.mention_id):
                if mention.role != role:
                    continue
                pieces.append(MARKER_START[role])
                by_level = getattr(mention, "displayed_by_level", None) or {}
                override = by_level.get(level)
                if override is None and level not in {"E0", "E1", "E2", "native"}:
                    override = getattr(mention, "displayed_type", None)
                shown = mapped_type(mention.entity_type, level, merge, override)
                if shown is not None:
                    pieces.append(type_token(shown))
        if index + 1 < len(positions):
            pieces.append(text[pos:positions[index + 1]])
    return "".join(pieces)


def _overlapping(tokens: list[TokenSpan], start: int, end: int) -> tuple[int, int]:
    indexes = [index for index, token in enumerate(tokens) if token.start < end and token.end > start]
    if not indexes:
        raise ValueError(f"No original token overlaps character span {start}:{end}")
    return indexes[0], indexes[-1] + 1


def _covers_mention(tokens: list[TokenSpan], left: int, right: int, mention: MentionView) -> bool:
    return tokens[left].start <= mention.start and tokens[right - 1].end >= mention.end


def _expand_to_whole_mentions(
    tokens: list[TokenSpan],
    left: int,
    right: int,
    mentions: list[MentionView],
) -> tuple[int, int] | None:
    current = (left, right)
    changed = True
    while changed:
        changed = False
        char_start = tokens[current[0]].start
        char_end = tokens[current[1] - 1].end
        for mention in mentions:
            overlaps = mention.start < char_end and mention.end > char_start
            if overlaps and not _covers_mention(tokens, current[0], current[1], mention):
                grown = _overlapping(tokens, min(char_start, mention.start), max(char_end, mention.end))
                if grown != current:
                    current = grown
                    changed = True
    for mention in mentions:
        char_start = tokens[current[0]].start
        char_end = tokens[current[1] - 1].end
        overlaps = mention.start < char_end and mention.end > char_start
        if overlaps and not _covers_mention(tokens, current[0], current[1], mention):
            return None
    return current


def _render_views(
    text: str,
    fragments: list[tuple[int, int]],
    mentions: list[MentionView],
    views: dict[str, dict | None],
) -> dict[str, str]:
    rendered = {}
    for level, merge in views.items():
        parts = []
        for start, end in fragments:
            included = [mention for mention in mentions if mention.start >= start and mention.end <= end]
            parts.append(render_marked_text(text[start:end], _shift(included, start), level=level, merge=merge))
        rendered[level] = GAP_MARKER.join(parts) if len(parts) > 1 else parts[0]
    return rendered


def _shift(mentions: list[MentionView], start: int) -> list[MentionView]:
    return [
        MentionView(
            mention.mention_id,
            mention.start - start,
            mention.end - start,
            mention.role,
            mention.entity_type,
            mention.displayed_type,
            mention.displayed_by_level,
        )
        for mention in mentions
    ]


def _lengths(rendered: dict[str, str], encoder: TextEncoder) -> dict[str, int]:
    return {level: encoder.encoded_length(text) for level, text in rendered.items()}


def _fits(lengths: dict[str, int], budget: int) -> bool:
    return max(lengths.values()) <= budget


def build_target_anchored_context(
    text: str,
    mentions: list[MentionView],
    encoder: TextEncoder,
    views: dict[str, dict | None],
    *,
    budget: int = 512,
) -> dict:
    """Select one original-text window for all supplied entity views.

    `views` maps a level name to the type-merge table used while rendering.
    E0 passes None and receives no type token.
    """
    if not mentions or {mention.role for mention in mentions} != {"H", "T"} and not (
        any(mention.role == "H" for mention in mentions) and any(mention.role == "T" for mention in mentions)
    ):
        raise ValueError("A context requires at least one H mention and one T mention")
    tokens = encoder.tokenize(text)
    if not tokens:
        return _anomaly("empty_document", mentions, budget)
    full_fragments = [(0, len(text))]
    full_rendered = _render_views(text, full_fragments, mentions, views)
    full_lengths = _lengths(full_rendered, encoder)
    if _fits(full_lengths, budget):
        return _record("full", full_fragments, mentions, full_rendered, full_lengths, budget, anchor_ids=_anchor_ids(mentions))

    anchors = _choose_anchors(tokens, mentions)
    if anchors is None:
        return _anomaly("anchor_not_aligned", mentions)
    left_mention, right_mention, cover_left, cover_right = anchors
    covered = _expand_to_whole_mentions(tokens, cover_left, cover_right, mentions)
    if covered is None:
        return _dual_or_anomaly(text, tokens, mentions, encoder, views, budget, left_mention, right_mention)
    cover_left, cover_right = covered
    cover_chars = _char_fragments(tokens, [(cover_left, cover_right)])
    cover_lengths = _lengths(_render_views(text, cover_chars, mentions, views), encoder)
    if not _fits(cover_lengths, budget):
        return _dual_or_anomaly(text, tokens, mentions, encoder, views, budget, left_mention, right_mention)

    left, right = cover_left, cover_right
    prefer_left = True
    while True:
        moved = False
        order = ("left", "right") if prefer_left else ("right", "left")
        for side in order:
            proposal = _propose(tokens, left, right, side, mentions)
            if proposal is None:
                continue
            proposal_chars = _char_fragments(tokens, [proposal])
            proposal_lengths = _lengths(_render_views(text, proposal_chars, mentions, views), encoder)
            if _fits(proposal_lengths, budget):
                left, right = proposal
                prefer_left = side != "left"
                moved = True
                break
        if not moved:
            break
    fragments = _char_fragments(tokens, [(left, right)])
    rendered = _render_views(text, fragments, mentions, views)
    lengths = _lengths(rendered, encoder)
    return _record(
        "single",
        fragments,
        mentions,
        rendered,
        lengths,
        budget,
        anchor_ids=[left_mention.mention_id, right_mention.mention_id],
    )


def _choose_anchors(tokens: list[TokenSpan], mentions: list[MentionView]):
    heads = [mention for mention in mentions if mention.role == "H"]
    tails = [mention for mention in mentions if mention.role == "T"]
    best = None
    for head in heads:
        for tail in tails:
            try:
                left, right = _overlapping(tokens, min(head.start, tail.start), max(head.end, tail.end))
            except ValueError:
                continue
            key = (right - left, left, right, head.mention_id, tail.mention_id)
            if best is None or key < best[0]:
                best = (key, head, tail, left, right)
    if best is None:
        return None
    _, head, tail, left, right = best
    if head.start <= tail.start:
        return head, tail, left, right
    return tail, head, left, right


def _propose(tokens, left, right, side, mentions):
    if side == "left":
        if left == 0:
            return None
        proposal = (left - 1, right)
    else:
        if right == len(tokens):
            return None
        proposal = (left, right + 1)
    return _expand_to_whole_mentions(tokens, proposal[0], proposal[1], mentions)


def _dual_or_anomaly(text, tokens, mentions, encoder, views, budget, left_mention, right_mention):
    """Grow two anchor fragments in a fixed order and merge them when they meet."""
    try:
        left_range = _overlapping(tokens, left_mention.start, left_mention.end)
        right_range = _overlapping(tokens, right_mention.start, right_mention.end)
    except ValueError:
        return _anomaly("anchor_not_aligned", mentions, budget)
    left_range = _expand_to_whole_mentions(tokens, left_range[0], left_range[1], [left_mention])
    right_range = _expand_to_whole_mentions(tokens, right_range[0], right_range[1], [right_mention])
    if left_range is None or right_range is None:
        return _anomaly("anchor_exceeds_token", mentions, budget)
    ranges = _merge_ranges([left_range, right_range])
    if not _ranges_fit(text, tokens, ranges, mentions, encoder, views, budget):
        record = _anomaly("budget_exceeded", mentions, budget)
        record["anchor_mention_ids"] = [left_mention.mention_id, right_mention.mention_id]
        return record
    cycle = (("left", 0), ("right", 0), ("left", 1), ("right", 1))
    cursor = 0
    consecutive_failures = 0
    while consecutive_failures < len(cycle) and len(ranges) > 1:
        side, which = cycle[cursor % len(cycle)]
        cursor += 1
        if which >= len(ranges):
            consecutive_failures += 1
            continue
        current = ranges[which]
        proposal = _propose(tokens, current[0], current[1], side, mentions)
        if proposal is None:
            consecutive_failures += 1
            continue
        trial = list(ranges)
        trial[which] = proposal
        trial = _merge_ranges(trial)
        if _ranges_fit(text, tokens, trial, mentions, encoder, views, budget):
            ranges = trial
            consecutive_failures = 0
        else:
            consecutive_failures += 1
    fragments = _char_fragments(tokens, ranges)
    rendered = _render_views(text, fragments, mentions, views)
    lengths = _lengths(rendered, encoder)
    mode = "single" if len(fragments) == 1 else "dual"
    return _record(
        mode,
        fragments,
        mentions,
        rendered,
        lengths,
        budget,
        anchor_ids=[left_mention.mention_id, right_mention.mention_id],
    )


def _ranges_fit(text, tokens, ranges, mentions, encoder, views, budget) -> bool:
    fragments = _char_fragments(tokens, ranges)
    return _fits(_lengths(_render_views(text, fragments, mentions, views), encoder), budget)


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(ranges)
    merged = [ordered[0]]
    for left, right in ordered[1:]:
        prev_left, prev_right = merged[-1]
        if left <= prev_right:
            merged[-1] = (prev_left, max(prev_right, right))
        else:
            merged.append((left, right))
    return merged


def _char_fragments(tokens: list[TokenSpan], ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(tokens[left].start, tokens[right - 1].end) for left, right in ranges]


def _anchor_ids(mentions: list[MentionView]) -> list[str]:
    heads = sorted((mention for mention in mentions if mention.role == "H"), key=lambda item: (item.start, item.end))
    tails = sorted((mention for mention in mentions if mention.role == "T"), key=lambda item: (item.start, item.end))
    return [heads[0].mention_id, tails[0].mention_id]


def _record(mode, fragments, mentions, rendered, lengths, budget, anchor_ids) -> dict:
    return {
        "strategy": STRATEGY,
        "overlap_representation": OVERLAP_REPRESENTATION,
        "mode": mode,
        "fragments": [{"char_start": start, "char_end": end} for start, end in fragments],
        "anchor_mention_ids": anchor_ids,
        "marked_mention_ids": [mention.mention_id for mention in mentions if _mention_in_fragments(mention, fragments)],
        "views": rendered,
        "encoded_length_by_view": lengths,
        "max_encoded_length": max(lengths.values()),
        "budget": budget,
        "anomaly": None,
    }


def _mention_in_fragments(mention: MentionView, fragments: list[tuple[int, int]]) -> bool:
    return any(mention.start >= start and mention.end <= end for start, end in fragments)


def _anomaly(code: str, mentions: list[MentionView], budget: int = 512) -> dict:
    return {
        "strategy": STRATEGY,
        "overlap_representation": OVERLAP_REPRESENTATION,
        "mode": "anomaly",
        "fragments": [],
        "anchor_mention_ids": _anchor_ids(mentions) if mentions else [],
        "marked_mention_ids": [],
        "views": {},
        "encoded_length_by_view": {},
        "max_encoded_length": None,
        "budget": budget,
        "anomaly": code,
    }


def load_biomedbert_tokenizer(tokenizer_dir: str, extra_tokens: list[str] | None = None):
    """Load the local WordPiece tokenizer. This does not load model weights."""
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir, local_files_only=True)
    tokens = extra_tokens or SPECIAL_MARKERS
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    return tokenizer


class BiomedBertEncoder:
    """Budget encoder backed by the fixed BiomedBERT tokenizer."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, text: str) -> list[TokenSpan]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )
        spans = []
        for start, end in encoded["offset_mapping"]:
            if start == end:
                continue
            spans.append(TokenSpan(int(start), int(end), text[int(start):int(end)]))
        return spans

    def encoded_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True))
