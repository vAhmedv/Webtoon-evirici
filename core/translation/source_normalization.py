"""Conservative source-case normalization used only by translation providers.

OCR output is retained verbatim in ``TranslationOutputItem.source``.  This module
only prepares ordinary ALL-CAPS English prose for terminology matching and the
translation model.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from core.translation.series_profile import SeriesProfile


_STRUCTURED_TOKEN_RE = re.compile(
    r"(?:\b(?:ZFO|VRMMO|HP|MP)\b|\bLV\.\s*\d+\b|"
    r"\[[^\]\r\n]+\]|〈[^〉\r\n]+〉|《[^》\r\n]+》)",
    re.IGNORECASE,
)
_EXPLICIT_LABEL_RE = re.compile(
    r"^\s*(?:PASSIVE\s+SKILL|UNIQUE\s+TRAIT|SKILL|ABILITY|TITLE)"
    r"\s*(?:ACQUIRED|AVAILABLE)?\s*:",
    re.IGNORECASE,
)
_WATERMARK_OR_CREDIT_RE = re.compile(
    r"(?:\b[a-z0-9_-]+\.(?:com|net|org|site|io|xyz|info)\b|"
    r"\b(?:scanlation|scans|translator|translation|discord\.gg)\b)",
    re.IGNORECASE,
)
_SFX_RE = re.compile(
    r"^[\s.!?~'-]*(?:AH+|OH+|UH+|UM+|GAH+|BOOM|BANG|CLANG|SWOOSH|"
    r"WHOOSH|THUD|THUMP|CRASH|RUMBLE|GRR+|SIGH|GULP|GASP|HAHA|HEH)"
    r"[\s.!?~'-]*$",
    re.IGNORECASE,
)
_SYSTEM_UI_RE = re.compile(r"^\s*(?:\[[^\]]+\]|〈[^〉]+〉|《[^》]+》)\s*$")
_ALPHA_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_SENTENCE_START_RE = re.compile(r"(^|[.!?]+[\s\"'“”‘’(\[]+)([a-z])")


def _is_all_caps_english(text: str) -> bool:
    cased = [char for char in text if char.isalpha() and char.isascii()]
    return bool(cased) and all(not char.islower() for char in cased)


def _is_ordinary_prose(text: str) -> bool:
    """Return whether an ALL-CAPS line is safe to sentence-case."""
    words = _ALPHA_WORD_RE.findall(text)
    if len(words) < 3:
        return False
    if _EXPLICIT_LABEL_RE.search(text):
        return False
    if _WATERMARK_OR_CREDIT_RE.search(text) or _SFX_RE.fullmatch(text):
        return False
    if _SYSTEM_UI_RE.fullmatch(text):
        return False
    return any(char in text for char in ".?!,") or len(words) >= 4


def _canonical_source_spans(
    profile: SeriesProfile | None,
    approved_terms: Mapping[str, str] | None,
) -> list[tuple[str, str]]:
    spans: dict[str, str] = {}
    if profile:
        for source_name, display_name in profile.known_names.items():
            canonical = display_name.strip() or source_name.strip().title()
            spans[source_name.strip().casefold()] = canonical
        for source_term in profile.glossary:
            spans[source_term.strip().casefold()] = source_term.strip()
    for source_term in approved_terms or {}:
        spans[source_term.strip().casefold()] = source_term.strip()
    return sorted(spans.items(), key=lambda pair: len(pair[0]), reverse=True)


def normalize_translation_source_case(
    source: str,
    profile: SeriesProfile | None = None,
    approved_terms: Mapping[str, str] | None = None,
) -> str:
    """Sentence-case safe ALL-CAPS prose without mutating the stored source.

    Known names, approved English glossary keys, acronyms, and structured tokens
    are restored after casing with case-insensitive boundary matching.  Unknown
    term-only labels and non-prose lines are deliberately left untouched.
    """
    if not source or not _is_all_caps_english(source) or not _is_ordinary_prose(source):
        return source

    protected: dict[str, str] = {}
    working = source

    def protect_match(match: re.Match[str], canonical: str | None = None) -> str:
        token = f"\u0000WTCASE{len(protected):04d}\u0000"
        protected[token] = canonical if canonical is not None else match.group(0)
        return token

    for folded_source, canonical in _canonical_source_spans(profile, approved_terms):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(folded_source)}(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        working = pattern.sub(lambda m, value=canonical: protect_match(m, value), working)

    working = _STRUCTURED_TOKEN_RE.sub(lambda m: protect_match(m), working)
    working = working.lower()
    working = _SENTENCE_START_RE.sub(lambda m: m.group(1) + m.group(2).upper(), working)
    working = re.sub(r"(?<![A-Za-z])i(?=(?:['’](?:m|d|ll|ve))?\b)", "I", working)

    # Conservative name cues recover unknown names without title-casing arbitrary
    # prose: "Captain X", "Young Master X", and "My name is X".
    working = re.sub(
        r"\b(captain|young master|master)\s+([a-z][a-z'-]*)",
        lambda m: m.group(1).capitalize() + " " + m.group(2).capitalize(),
        working,
        flags=re.IGNORECASE,
    )
    working = re.sub(
        r"\bmy name is\s+([a-z][a-z'-]*)(?:\s+([a-z][a-z'-]*))?(?=[.!?,])",
        lambda m: "My name is "
        + m.group(1).capitalize()
        + ((" " + m.group(2).capitalize()) if m.group(2) else ""),
        working,
        flags=re.IGNORECASE,
    )

    for token, value in protected.items():
        working = working.replace(token.lower(), value)
    return working
