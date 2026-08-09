"""Application-Level Terminology Protection and Named-Term Detection.

Protects approved glossary terms and unapproved named abilities/skills/titles
at the application layer before calling TranslateGemma, and restores canonical forms afterwards.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from providers.translation.base import TranslationItem


# Regular expression patterns for high-confidence named-term detection
NAMED_TERM_PATTERNS = [
    re.compile(r"\b(it's|is)\s+called\s+([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)", re.IGNORECASE),
    re.compile(r"\bactivate\s+([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)", re.IGNORECASE),
    re.compile(r"\b(used|uses|using)\s+([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)", re.IGNORECASE),
    re.compile(r"\blearned\s+([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)", re.IGNORECASE),
    re.compile(r"^([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+)\s+(is|allows|keeps|allows|requires|remains|failed|cooldown)", re.MULTILINE),
    re.compile(r"(?:PASSIVE SKILL|TITLE|UNIQUE TRAIT|SKILL|ABILITY)\s*(?:ACQUIRED|AVAILABLE)?:?\s*([A-Z0-9\s]{3,30})", re.IGNORECASE),
]

EXCLUDED_WORDS = {
    "THE", "A", "AN", "AND", "OR", "BUT", "NOT", "IF", "THEN", "WHEN", "WHERE", "WHY",
    "HOW", "WHAT", "WHO", "MY", "YOUR", "HIS", "HER", "ITS", "OUR", "THEIR", "THIS", "THAT",
    "IT", "HE", "SHE", "THEY", "WE", "YOU", "I", "CAN", "COULD", "WOULD", "SHOULD", "MUST",
}


def detect_named_terms_in_items(items: list[TranslationItem]) -> set[str]:
    """Scan clean English source items in a batch/chapter to detect named terms.

    Establishes chapter-wide source-side consistency for unknown named abilities/skills/titles.
    """
    detected: set[str] = set()

    for item in items:
        source = item.source.strip()
        if not source:
            continue

        for pattern in NAMED_TERM_PATTERNS:
            for match in pattern.finditer(source):
                # Target the captured term group
                group_idx = len(match.groups())
                term_str = match.group(group_idx).strip(" .?!,:;\"'")

                # Validate multi-word or single TitleCase word
                words = term_str.split()
                if not words:
                    continue

                # Ignore trivial common words
                if len(words) == 1 and words[0].upper() in EXCLUDED_WORDS:
                    continue

                if all(w[0].isupper() for w in words if w and w[0].isalpha()):
                    detected.add(term_str)

    return detected


def protect_source_text(
    source_text: str,
    approved_terms: dict[str, str],
    detected_named_terms: set[str],
) -> tuple[str, dict[str, tuple[str, str, bool]]]:
    """Prepare source text for TranslateGemma by replacing terms with protected token forms.

    Returns (protected_source_text, placeholder_map).
    placeholder_map maps `protected_token` -> (original_text, target_base_text, is_approved).
    """
    protected_text = source_text
    placeholder_map: dict[str, tuple[str, str, bool]] = {}

    # Sort terms by length descending to prevent substring collisions
    all_targets: list[tuple[str, str, bool]] = []

    for src_k, tgt_v in approved_terms.items():
        all_targets.append((src_k, tgt_v, True))

    for named_t in detected_named_terms:
        # Avoid overriding explicit approved terms
        if not any(named_t.upper() == k.upper() for k in approved_terms):
            all_targets.append((named_t, named_t, False))

    all_targets.sort(key=lambda x: len(x[0]), reverse=True)

    for orig_src, target_val, is_approved in all_targets:
        if not orig_src.strip():
            continue

        # Case-insensitive word boundary match
        pattern = re.compile(re.escape(orig_src), re.IGNORECASE)

        def replacer(match: re.Match) -> str:
            matched_str = match.group(0)
            # Create a protected token using underscores
            # e.g., "Secret Realm" -> "Secret_Realm" or "Gizli Diyar" -> "Gizli_Diyar"
            base_repr = target_val if is_approved else matched_str
            token = base_repr.replace(" ", "_")
            placeholder_map[token] = (matched_str, target_val, is_approved)
            return token

        if pattern.search(protected_text):
            protected_text = pattern.sub(replacer, protected_text)

    return protected_text, placeholder_map


def restore_protected_translation(
    translated_text: str,
    placeholder_map: dict[str, tuple[str, str, bool]],
) -> str:
    """Restore underscore tokens back to canonical space-separated Turkish/English terms.

    Preserves attached Turkish grammatical suffixes (e.g. `Gizli_Diyar'a` -> `Gizli Diyar'a`).
    """
    restored = translated_text

    # Sort placeholders by token length descending
    sorted_placeholders = sorted(placeholder_map.items(), key=lambda x: len(x[0]), reverse=True)

    for token, (_, target_val, _) in sorted_placeholders:
        if token in restored:
            restored = restored.replace(token, target_val)

    return restored


def validate_protected_terms(
    restored_translation: str,
    placeholder_map: dict[str, tuple[str, str, bool]],
) -> list[str]:
    """Verify that all approved terms' target bases appear in the restored translation.

    Returns warnings list (e.g., ["approved_term_missing"]).
    """
    warnings: list[str] = []

    for token, (_, target_val, is_approved) in placeholder_map.items():
        if is_approved:
            # Check if canonical target base is present in translation
            # e.g. "Gizli Diyar" in "Gizli Diyar'a girdik."
            if target_val.lower() not in restored_translation.lower():
                warnings.append("approved_term_missing")

    return warnings
