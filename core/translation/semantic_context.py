"""Selective local-context semantic disambiguation with deterministic safety gates.

This module never translates text. It validates a resolver's minimally clarified
English target before an experiment may pass that target to the normal translation
pipeline. Production translation defaults are intentionally outside this module.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable


DEFAULT_SEMANTIC_RESOLVER_MIN_CONFIDENCE = 0.85
NORMAL_CONTEXT_REGION_TYPES = frozenset({"dialogue", "narration"})

_WH_QUESTION_WORDS = {
    "who": "who",
    "whom": "who",
    "whose": "whose",
    "what": "what",
    "why": "why",
    "how": "how",
    "when": "when",
    "where": "where",
    "which": "which",
}
_YES_NO_AUXILIARIES = frozenset(
    {
        "am",
        "are",
        "is",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
    }
)
_NEGATION_PATTERN = re.compile(
    r"\b(?:not|never|no|nobody|nothing|neither|nor|without|cannot)\b|n['’]t\b",
    re.IGNORECASE,
)
_DIGIT_NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?(?:st|nd|rd|th)?\b", re.IGNORECASE)
_NUMBER_WORDS = frozenset(
    {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
    }
)
_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_SENTINEL_PATTERN = re.compile(r"__WTTERM\d+__", re.IGNORECASE)
_TITLE_ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b"
)
_TURKISH_CHARACTER_PATTERN = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
_TURKISH_SUFFIX_PATTERN = re.compile(
    r"\b[A-Za-zçğıöşü]+(?:abilir|ebilir|acak|ecek|mış|miş|muş|müş|ıyor|iyor|uyor|üyor)\b",
    re.IGNORECASE,
)
_COMMON_TURKISH_WORDS = frozenset(
    {
        "ama",
        "ben",
        "bir",
        "biz",
        "bu",
        "bunu",
        "bunun",
        "çünkü",
        "değil",
        "daha",
        "gibi",
        "için",
        "ile",
        "kim",
        "nasıl",
        "neden",
        "olan",
        "olarak",
        "sadece",
        "sen",
        "şu",
        "ve",
    }
)
_ALLOWED_TRANSLATION_RISK_TYPES = frozenset(
    {
        "lexical_sense",
        "phrasal_verb",
        "pronoun_reference",
        "question_structure",
        "tense_aspect",
        "state_action",
        "ellipsis",
        "register",
        "named_term_interpretation",
    }
)
_ALLOWED_DECLARED_QUESTION_TYPES = frozenset(
    {"who", "whose", "what", "why", "when", "where", "how", "which", "yes_no", "other"}
)
_MODAL_WORDS = frozenset(
    {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
)
_BE_FORMS = frozenset({"am", "are", "is", "was", "were", "be", "been", "being"})
_PAST_PARTICIPLES = frozenset(
    {
        "been",
        "become",
        "begun",
        "broken",
        "come",
        "done",
        "found",
        "given",
        "gone",
        "known",
        "left",
        "made",
        "run",
        "seen",
        "shut",
        "taken",
        "told",
        "worn",
        "written",
    }
)
_OBVIOUS_PAST_FORMS = _PAST_PARTICIPLES | frozenset(
    {
        "brought",
        "built",
        "caught",
        "chose",
        "drew",
        "fell",
        "felt",
        "got",
        "heard",
        "held",
        "kept",
        "knew",
        "lost",
        "met",
        "ran",
        "said",
        "saw",
        "sent",
        "spoke",
        "stood",
        "thought",
        "took",
        "went",
        "wore",
    }
)
_TEMPORAL_IDENTITY_WORDS = frozenset(
    {
        "after",
        "already",
        "before",
        "dawn",
        "day",
        "earlier",
        "evening",
        "later",
        "midnight",
        "moment",
        "morning",
        "night",
        "noon",
        "now",
        "since",
        "still",
        "sunrise",
        "sunset",
        "then",
        "today",
        "tomorrow",
        "tonight",
        "until",
        "while",
        "yesterday",
        "yet",
    }
)
_CONTROL_GARBAGE_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
_ROLE_OR_SENTINEL_GARBAGE_PATTERN = re.compile(
    r"<\|[^|]+\|>|\[(?:system|assistant|user)\]|(?:^|\s)(?:SYSTEM|ASSISTANT|USER):",
    re.IGNORECASE,
)
_RISK_TYPE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


class TranslationExperimentMode(str, Enum):
    """Explicit experiment modes; neither is a production default."""

    BASELINE_C = "baseline_c"
    SEMANTIC_CONTEXT_C = "semantic_context_c"


@dataclass(frozen=True)
class LocalContextRegion:
    region_id: int
    reading_order: int
    source: str
    region_type: str
    scene_id: str | None = None


@dataclass(frozen=True)
class SemanticContextRequest:
    previous_context: tuple[str, ...]
    target_source: str
    next_context: tuple[str, ...]
    named_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticNote:
    span: str
    intended_sense: str
    evidence: str


@dataclass(frozen=True)
class SemanticResolution:
    ambiguous: bool
    confidence: float
    semantic_notes: tuple[SemanticNote, ...]
    question_type: str | None
    tense_aspect: str
    referents: tuple[Any, ...]
    register_hint: str | None
    clarified_target: str


@dataclass(frozen=True)
class ClarificationDecision:
    selected_target: str
    clarification_used: bool
    rejection_reason: str | None
    validation_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolverOutcome:
    request: SemanticContextRequest
    raw_response: str
    resolution: SemanticResolution | None
    decision: ClarificationDecision
    resolver_failed: bool = False
    malformed_json: bool = False


@dataclass(frozen=True)
class TranslationRiskNote:
    span: str
    resolved_meaning: str
    evidence: str


@dataclass(frozen=True)
class TranslationRiskResolution:
    rewrite_needed: bool
    confidence: float
    risk_types: tuple[str, ...]
    semantic_notes: tuple[TranslationRiskNote, ...]
    question_type: str | None
    tense_aspect: str | None
    referents: tuple[Any, ...]
    clarified_target: str


@dataclass(frozen=True)
class RewriteDecision:
    selected_target: str
    rewrite_used: bool
    rejection_reason: str | None
    validation_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class TranslationRiskOutcome:
    request: SemanticContextRequest
    raw_response: str
    resolution: TranslationRiskResolution | None
    decision: RewriteDecision
    resolver_failed: bool = False
    malformed_json: bool = False


@dataclass(frozen=True)
class ControlledEnglishNote:
    span: str
    resolved_meaning: str
    evidence: str


@dataclass(frozen=True)
class ControlledEnglishResolution:
    rewrite_needed: bool
    confidence: float
    risk_types: tuple[str, ...]
    semantic_notes: tuple[ControlledEnglishNote, ...]
    question_word: str | None
    tense_aspect: str | None
    referents: tuple[Any, ...]
    controlled_target: str


@dataclass(frozen=True)
class ControlledBridgeDecision:
    selected_target: str
    rewrite_used: bool
    rejection_reason: str | None
    validation_failures: tuple[str, ...] = ()
    validator_uncertain: bool = False


@dataclass(frozen=True)
class ControlledBridgeOutcome:
    request: SemanticContextRequest
    raw_response: str
    resolution: ControlledEnglishResolution | None
    decision: ControlledBridgeDecision
    resolver_failed: bool = False
    malformed_json: bool = False


def select_local_context(
    regions: Iterable[LocalContextRegion],
    target_region_id: int,
    *,
    max_previous: int = 3,
    max_next: int = 1,
    max_order_distance: int = 6,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select nearby ordinary English regions without crossing known scene bounds."""
    ordered = sorted(regions, key=lambda item: (item.reading_order, item.region_id))
    target_index = next(
        (index for index, item in enumerate(ordered) if item.region_id == target_region_id),
        None,
    )
    if target_index is None:
        raise ValueError(f"Unknown target region ID: {target_region_id}")

    target = ordered[target_index]

    def is_nearby(candidate: LocalContextRegion) -> bool:
        if target.scene_id is not None and candidate.scene_id != target.scene_id:
            return False
        return abs(candidate.reading_order - target.reading_order) <= max_order_distance

    previous: list[str] = []
    for candidate in reversed(ordered[:target_index]):
        if not is_nearby(candidate):
            if target.scene_id is not None or (
                target.reading_order - candidate.reading_order > max_order_distance
            ):
                break
            continue
        if candidate.region_type not in NORMAL_CONTEXT_REGION_TYPES:
            continue
        source = candidate.source.strip()
        if source:
            previous.append(source)
        if len(previous) == max_previous:
            break
    previous.reverse()

    following: list[str] = []
    for candidate in ordered[target_index + 1 :]:
        if not is_nearby(candidate):
            if target.scene_id is not None or (
                candidate.reading_order - target.reading_order > max_order_distance
            ):
                break
            continue
        if candidate.region_type not in NORMAL_CONTEXT_REGION_TYPES:
            continue
        source = candidate.source.strip()
        if source:
            following.append(source)
        if len(following) == max_next:
            break

    return tuple(previous), tuple(following)


def _format_context(lines: tuple[str, ...]) -> str:
    if not lines:
        return "(none)"
    return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, start=1))


def render_semantic_resolver_prompt(request: SemanticContextRequest) -> str:
    """Render the strict text-only Qwen semantic resolver prompt."""
    return (
        "You are a semantic disambiguation assistant for English dialogue and narration.\n\n"
        "Your task is NOT to translate into Turkish.\n"
        "Use the nearby context only to determine the intended meaning of the TARGET sentence.\n"
        "Do not invent story facts.\n"
        "Do not add information not supported by the target or context.\n"
        "Do not paraphrase for style.\n"
        "Do not rewrite named terms.\n"
        "Do not translate proper names or abilities.\n\n"
        "Return JSON only. Do not use Markdown or add prose before or after the JSON.\n\n"
        f"PREVIOUS CONTEXT:\n{_format_context(request.previous_context)}\n\n"
        f"TARGET:\n{request.target_source}\n\n"
        f"NEXT CONTEXT:\n{_format_context(request.next_context)}\n\n"
        "Required JSON schema:\n"
        "{\n"
        '  "ambiguous": true,\n'
        '  "confidence": 0.0,\n'
        '  "semantic_notes": [\n'
        "    {\n"
        '      "span": "...",\n'
        '      "intended_sense": "...",\n'
        '      "evidence": "..."\n'
        "    }\n"
        "  ],\n"
        '  "question_type": null,\n'
        '  "tense_aspect": "...",\n'
        '  "referents": [],\n'
        '  "register_hint": null,\n'
        '  "clarified_target": "..."\n'
        "}\n\n"
        "The clarified_target must remain English and preserve all named terms, numbers, "
        "polarity, question type, and tense/aspect. Make only the minimum change required "
        "to disambiguate meaning. If no clarification is necessary, set clarified_target "
        "exactly equal to TARGET. If context does not support a confident resolution, do "
        "not guess: keep ambiguous true, use confidence below the acceptance threshold, "
        "and set clarified_target exactly equal to TARGET."
    )


def render_translation_risk_resolver_prompt(request: SemanticContextRequest) -> str:
    """Render the strict V2 translation-risk resolver prompt."""
    return (
        "You are an English semantic clarification assistant for a downstream machine "
        "translation system.\n\n"
        "You do NOT translate into Turkish.\n\n"
        "Your job is to determine whether the TARGET contains wording that a translation "
        "model could plausibly interpret with the wrong meaning, even when the intended "
        "meaning is clear from nearby context.\n\n"
        "Use PREVIOUS CONTEXT and NEXT CONTEXT only to resolve the meaning already expressed "
        "by TARGET.\n\n"
        "If a word, phrase, pronoun, question structure, tense/aspect construction, or other "
        "expression could be mistranslated, and the local context clearly determines its "
        "intended meaning, rewrite the TARGET in minimally clarified English.\n\n"
        "The clarified TARGET must preserve the exact original meaning.\n\n"
        "Do not improve style.\n"
        "Do not add story facts.\n"
        "Do not explain implied lore.\n"
        "Do not translate proper names.\n"
        "Do not replace named abilities.\n"
        "Do not change numbers.\n"
        "Do not change polarity.\n"
        "Do not change speaker intent.\n"
        "Do not change question type.\n"
        "Do not change tense/aspect unless required to preserve the same meaning more "
        "explicitly.\n\n"
        "A rewrite is appropriate even if YOU personally understand the original sentence "
        "without ambiguity. The question is whether clearer English would reduce translation "
        "risk for another model.\n\n"
        "Examples of valid semantic clarification:\n\n"
        '"Frost Chain can hold three targets at once."\n'
        "If context clearly shows Frost Chain immobilizes enemies:\n"
        '"Frost Chain can restrain three targets at once."\n\n'
        '"Who charged the meal to my room?"\n'
        "If context clearly establishes a hotel room bill:\n"
        '"Who added the meal charge to my room bill?"\n\n'
        "Examples of invalid clarification:\n\n"
        "Adding weapon types, magic effects, motivations, identities, locations, or events "
        "not stated or supported.\n\n"
        "Return JSON only.\n"
        "No Markdown.\n"
        "No prose outside JSON.\n\n"
        f"PREVIOUS CONTEXT:\n{_format_context(request.previous_context)}\n\n"
        f"TARGET:\n{request.target_source}\n\n"
        f"NEXT CONTEXT:\n{_format_context(request.next_context)}\n\n"
        "Required JSON schema:\n"
        "{\n"
        '  "rewrite_needed": true,\n'
        '  "confidence": 0.0,\n'
        '  "risk_types": [],\n'
        '  "semantic_notes": [\n'
        "    {\n"
        '      "span": "...",\n'
        '      "resolved_meaning": "...",\n'
        '      "evidence": "..."\n'
        "    }\n"
        "  ],\n"
        '  "question_type": null,\n'
        '  "tense_aspect": null,\n'
        '  "referents": [],\n'
        '  "clarified_target": "..."\n'
        "}\n\n"
        "Rules:\n\n"
        "If no rewrite would materially reduce translation risk:\n"
        "rewrite_needed = false\n"
        "clarified_target MUST equal TARGET exactly.\n\n"
        "If context is insufficient to safely determine the intended meaning:\n"
        "rewrite_needed = false\n"
        "confidence must be low\n"
        "clarified_target MUST equal TARGET exactly.\n\n"
        "If context clearly determines a translation-risky sense:\n"
        "rewrite_needed = true\n"
        "confidence should reflect certainty\n"
        "clarified_target should make only the minimum semantic clarification necessary."
    )


def parse_semantic_resolution(raw_response: str) -> SemanticResolution:
    """Parse the resolver's strict JSON-only response and validate its schema."""
    try:
        payload = json.loads(raw_response.strip())
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("malformed_resolver_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("resolver_json_must_be_object")

    required_keys = {
        "ambiguous",
        "confidence",
        "semantic_notes",
        "question_type",
        "tense_aspect",
        "referents",
        "register_hint",
        "clarified_target",
    }
    if set(payload) != required_keys:
        raise ValueError("resolver_json_schema_mismatch")

    ambiguous = payload["ambiguous"]
    confidence = payload["confidence"]
    notes_payload = payload["semantic_notes"]
    question_type = payload["question_type"]
    tense_aspect = payload["tense_aspect"]
    referents = payload["referents"]
    register_hint = payload["register_hint"]
    clarified_target = payload["clarified_target"]

    if not isinstance(ambiguous, bool):
        raise ValueError("resolver_ambiguous_must_be_boolean")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("resolver_confidence_must_be_number")
    confidence_float = float(confidence)
    if not 0.0 <= confidence_float <= 1.0:
        raise ValueError("resolver_confidence_out_of_range")
    if not isinstance(notes_payload, list):
        raise ValueError("resolver_semantic_notes_must_be_list")

    notes: list[SemanticNote] = []
    for note in notes_payload:
        if not isinstance(note, dict) or set(note) != {
            "span",
            "intended_sense",
            "evidence",
        }:
            raise ValueError("resolver_semantic_note_schema_mismatch")
        if not all(isinstance(note[key], str) for key in note):
            raise ValueError("resolver_semantic_note_values_must_be_strings")
        notes.append(
            SemanticNote(
                span=note["span"],
                intended_sense=note["intended_sense"],
                evidence=note["evidence"],
            )
        )

    if question_type is not None and not isinstance(question_type, str):
        raise ValueError("resolver_question_type_must_be_string_or_null")
    if not isinstance(tense_aspect, str):
        raise ValueError("resolver_tense_aspect_must_be_string")
    if not isinstance(referents, list):
        raise ValueError("resolver_referents_must_be_list")
    if register_hint is not None and not isinstance(register_hint, str):
        raise ValueError("resolver_register_hint_must_be_string_or_null")
    if not isinstance(clarified_target, str):
        raise ValueError("resolver_clarified_target_must_be_string")

    return SemanticResolution(
        ambiguous=ambiguous,
        confidence=confidence_float,
        semantic_notes=tuple(notes),
        question_type=question_type.strip().lower() if question_type else None,
        tense_aspect=tense_aspect,
        referents=tuple(referents),
        register_hint=register_hint,
        clarified_target=clarified_target,
    )


def parse_translation_risk_resolution(
    raw_response: str,
) -> TranslationRiskResolution:
    """Parse one strict V2 JSON-only response and validate its exact schema."""
    try:
        payload = json.loads(raw_response.strip())
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("malformed_translation_risk_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("translation_risk_json_must_be_object")

    required_keys = {
        "rewrite_needed",
        "confidence",
        "risk_types",
        "semantic_notes",
        "question_type",
        "tense_aspect",
        "referents",
        "clarified_target",
    }
    if set(payload) != required_keys:
        raise ValueError("translation_risk_json_schema_mismatch")

    rewrite_needed = payload["rewrite_needed"]
    confidence = payload["confidence"]
    risk_types_payload = payload["risk_types"]
    notes_payload = payload["semantic_notes"]
    question_type = payload["question_type"]
    tense_aspect = payload["tense_aspect"]
    referents = payload["referents"]
    clarified_target = payload["clarified_target"]

    if not isinstance(rewrite_needed, bool):
        raise ValueError("translation_risk_rewrite_needed_must_be_boolean")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("translation_risk_confidence_must_be_number")
    confidence_float = float(confidence)
    if not 0.0 <= confidence_float <= 1.0:
        raise ValueError("translation_risk_confidence_out_of_range")
    if not isinstance(risk_types_payload, list) or not all(
        isinstance(value, str) and _RISK_TYPE_PATTERN.fullmatch(value)
        for value in risk_types_payload
    ):
        raise ValueError("translation_risk_types_must_be_controlled_strings")
    if len(set(risk_types_payload)) != len(risk_types_payload):
        raise ValueError("translation_risk_types_must_be_unique")
    if not isinstance(notes_payload, list):
        raise ValueError("translation_risk_semantic_notes_must_be_list")

    notes: list[TranslationRiskNote] = []
    for note in notes_payload:
        if not isinstance(note, dict) or set(note) != {
            "span",
            "resolved_meaning",
            "evidence",
        }:
            raise ValueError("translation_risk_semantic_note_schema_mismatch")
        if not all(isinstance(note[key], str) for key in note):
            raise ValueError("translation_risk_semantic_note_values_must_be_strings")
        notes.append(
            TranslationRiskNote(
                span=note["span"],
                resolved_meaning=note["resolved_meaning"],
                evidence=note["evidence"],
            )
        )

    if question_type is not None:
        if not isinstance(question_type, str):
            raise ValueError("translation_risk_question_type_must_be_string_or_null")
        question_type = question_type.strip().lower()
        if question_type not in _ALLOWED_DECLARED_QUESTION_TYPES:
            raise ValueError("translation_risk_question_type_not_supported")
    if tense_aspect is not None and not isinstance(tense_aspect, str):
        raise ValueError("translation_risk_tense_aspect_must_be_string_or_null")
    if not isinstance(referents, list):
        raise ValueError("translation_risk_referents_must_be_list")
    if not isinstance(clarified_target, str):
        raise ValueError("translation_risk_clarified_target_must_be_string")
    return TranslationRiskResolution(
        rewrite_needed=rewrite_needed,
        confidence=confidence_float,
        risk_types=tuple(risk_types_payload),
        semantic_notes=tuple(notes),
        question_type=question_type,
        tense_aspect=tense_aspect.strip() if tense_aspect else None,
        referents=tuple(referents),
        clarified_target=clarified_target,
    )


def detect_question_type(text: str) -> str | None:
    """Classify English question structure without using benchmark-specific phrases."""
    if not re.search(r"\?\s*[\"'’)]*$", text.strip()):
        return None
    words = [word.lower().replace("’", "'") for word in _WORD_PATTERN.findall(text)]
    for word in words[:4]:
        if word in _WH_QUESTION_WORDS:
            return _WH_QUESTION_WORDS[word]
    if words and words[0] in _YES_NO_AUXILIARIES:
        return "yes_no"
    return "other"


def _number_tokens(text: str) -> Counter[str]:
    tokens = [match.group(0).lower() for match in _DIGIT_NUMBER_PATTERN.finditer(text)]
    tokens.extend(
        word.lower()
        for word in _WORD_PATTERN.findall(text)
        if word.lower() in _NUMBER_WORDS
    )
    return Counter(tokens)


def _contains_phrase(text: str, phrase: str) -> bool:
    parts = [re.escape(part) for part in phrase.split() if part]
    if not parts:
        return True
    pattern = r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)"
    return bool(re.search(pattern, text, re.IGNORECASE))


def _contains_exact_phrase(text: str, phrase: str) -> bool:
    parts = [re.escape(part) for part in phrase.split() if part]
    if not parts:
        return True
    pattern = r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)"
    return bool(re.search(pattern, text))


def _looks_turkish(text: str) -> bool:
    if _TURKISH_CHARACTER_PATTERN.search(text) or _TURKISH_SUFFIX_PATTERN.search(text):
        return True
    words = [word.lower() for word in _WORD_PATTERN.findall(text)]
    return sum(word in _COMMON_TURKISH_WORDS for word in words) >= 2


def _looks_non_english(text: str) -> bool:
    if _looks_turkish(text):
        return True
    letters = [character for character in text if character.isalpha()]
    if not letters or not _WORD_PATTERN.search(text):
        return True
    non_ascii_letters = sum(
        not ("A" <= character <= "Z" or "a" <= character <= "z")
        for character in letters
    )
    return non_ascii_letters / len(letters) > 0.2


def _negation_tokens(text: str) -> Counter[str]:
    normalized: list[str] = []
    for match in _NEGATION_PATTERN.finditer(text):
        token = match.group(0).lower().replace("’", "'")
        if token in {"not", "cannot"} or token.endswith("n't"):
            normalized.append("not")
        else:
            normalized.append(token)
    return Counter(normalized)


def _has_perfect_auxiliary(words: list[str], auxiliary: str) -> bool:
    for index, word in enumerate(words[:-1]):
        if word != auxiliary:
            continue
        following_index = index + 1
        if words[following_index] == "not" and following_index + 1 < len(words):
            following_index += 1
        following = words[following_index]
        if following.endswith(("ed", "en")) or following in _PAST_PARTICIPLES:
            return True
    return False


def _tense_aspect_signature(text: str) -> tuple[Any, ...]:
    words = [word.lower().replace("’", "'") for word in _WORD_PATTERN.findall(text)]
    modals = Counter(word for word in words if word in _MODAL_WORDS)
    present_perfect = any(_has_perfect_auxiliary(words, word) for word in ("have", "has"))
    past_perfect = _has_perfect_auxiliary(words, "had")
    progressive = any(
        word in _BE_FORMS and index + 1 < len(words) and words[index + 1].endswith("ing")
        for index, word in enumerate(words)
    )
    determiners = {"the", "a", "an", "this", "that", "my", "your", "his", "her", "its", "our", "their"}
    obvious_past_words = []
    for idx, word in enumerate(words):
        if word in _OBVIOUS_PAST_FORMS:
            obvious_past_words.append(word)
        elif word.endswith("ed"):
            if idx > 0 and words[idx - 1] in determiners:
                continue
            obvious_past_words.append(word)
    obvious_past = len(obvious_past_words) > 0
    temporal_identities = Counter(word for word in words if word in _TEMPORAL_IDENTITY_WORDS)
    return (
        tuple(sorted(modals.items())),
        present_perfect,
        past_perfect,
        progressive,
        obvious_past,
        tuple(sorted(temporal_identities.items())),
    )


def _normalized_sentence(text: str) -> str:
    return " ".join(word.lower() for word in _WORD_PATTERN.findall(text))


def _copies_context_sentence(
    original: str,
    clarified: str,
    context_sources: tuple[str, ...],
) -> bool:
    original_normalized = _normalized_sentence(original)
    clarified_normalized = _normalized_sentence(clarified)
    for source in context_sources:
        context_normalized = _normalized_sentence(source)
        if len(context_normalized.split()) < 3:
            continue
        if (
            context_normalized in clarified_normalized
            and context_normalized not in original_normalized
        ):
            return True
    return False


def _contains_structured_garbage(text: str) -> bool:
    stripped = text.strip()
    if _CONTROL_GARBAGE_PATTERN.search(text):
        return True
    if _ROLE_OR_SENTINEL_GARBAGE_PATTERN.search(text):
        return True
    if "```" in text or "`" in text or any(character in text for character in "{}[]"):
        return True
    if re.match(r"^(?:#{1,6}\s|>\s|[-*+]\s)", stripped):
        return True
    return bool(re.search(r'"(?:rewrite_needed|clarified_target|semantic_notes)"\s*:', text))


def _terminal_punctuation(text: str) -> str | None:
    match = re.search(r"([?!.])\s*[\"'’)]*$", text.strip())
    return match.group(1) if match else None


def _unsupported_named_entities(
    original: str,
    clarified: str,
    context_sources: tuple[str, ...],
) -> list[str]:
    letters = [character for character in clarified if character.isalpha()]
    if letters and sum(character.isupper() for character in letters) / len(letters) >= 0.8:
        # All-caps comic dialogue does not expose reliable proper-noun boundaries.
        return []
    allowed_text = "\n".join((original, *context_sources))
    unsupported: list[str] = []
    for match in _TITLE_ENTITY_PATTERN.finditer(clarified):
        entity = match.group(0).strip()
        if entity in {"I"}:
            continue
        if match.start() == 0 and " " not in entity and not entity.isupper():
            continue
        if not _contains_phrase(allowed_text, entity):
            unsupported.append(entity)
    return unsupported


def validate_clarified_target(
    original_target: str,
    clarified_target: str,
    *,
    named_terms: Iterable[str] = (),
    context_sources: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return generic deterministic safety failures for a proposed English target."""
    original = original_target.strip()
    clarified = clarified_target.strip()
    failures: list[str] = []

    if not clarified:
        return ("empty_clarified_target",)
    if _looks_non_english(clarified):
        failures.append("clarified_target_not_english")
    if _contains_structured_garbage(clarified):
        failures.append("structured_or_control_garbage")

    protected_identities = {
        term.strip()
        for term in named_terms
        if term.strip() and _contains_phrase(original, term.strip())
    }
    protected_identities.update(_SENTINEL_PATTERN.findall(original))
    if any(not _contains_exact_phrase(clarified, term) for term in protected_identities):
        failures.append("named_term_loss")

    if _number_tokens(original) != _number_tokens(clarified):
        failures.append("number_changed")
    if _negation_tokens(original) != _negation_tokens(clarified):
        failures.append("polarity_changed")

    original_question = detect_question_type(original)
    clarified_question = detect_question_type(clarified)
    if original_question != clarified_question:
        failures.append("question_type_changed")

    original_terminal = _terminal_punctuation(original)
    clarified_terminal = _terminal_punctuation(clarified)
    if (
        original_terminal in {"?", "!"}
        or clarified_terminal in {"?", "!"}
    ) and original_terminal != clarified_terminal:
        failures.append("terminal_punctuation_changed")

    original_word_count = max(1, len(_WORD_PATTERN.findall(original)))
    clarified_word_count = len(_WORD_PATTERN.findall(clarified))
    max_words = max(original_word_count + 6, math.ceil(original_word_count * 1.6))
    if clarified_word_count > max_words:
        failures.append("clarified_target_too_long")

    context_tuple = tuple(source.strip() for source in context_sources if source.strip())
    if _tense_aspect_signature(original) != _tense_aspect_signature(clarified):
        failures.append("tense_aspect_changed")
    if _copies_context_sentence(original, clarified, context_tuple):
        failures.append("context_sentence_copied")
    if _unsupported_named_entities(original, clarified, context_tuple):
        failures.append("unsupported_named_entity")

    return tuple(dict.fromkeys(failures))


def decide_clarification(
    request: SemanticContextRequest,
    resolution: SemanticResolution,
    *,
    min_confidence: float = DEFAULT_SEMANTIC_RESOLVER_MIN_CONFIDENCE,
) -> ClarificationDecision:
    """Apply confidence and deterministic safety gates to a parsed resolution."""
    original = request.target_source
    clarified = resolution.clarified_target.strip()
    if not resolution.ambiguous:
        return ClarificationDecision(original, False, "not_ambiguous")
    if clarified == original:
        return ClarificationDecision(original, False, "unchanged_target")
    if resolution.confidence < min_confidence:
        return ClarificationDecision(original, False, "low_confidence")

    failures = list(
        validate_clarified_target(
            original,
            clarified,
            named_terms=request.named_terms,
            context_sources=(*request.previous_context, *request.next_context),
        )
    )
    original_question = detect_question_type(original)
    if resolution.question_type != original_question:
        failures.append("declared_question_type_mismatch")
    failures = list(dict.fromkeys(failures))
    if failures:
        return ClarificationDecision(
            original,
            False,
            "clarification_validation_failed",
            tuple(failures),
        )
    return ClarificationDecision(clarified, True, None)


def resolve_with_fallback(
    request: SemanticContextRequest,
    resolver_call: Callable[[SemanticContextRequest], str],
    *,
    min_confidence: float = DEFAULT_SEMANTIC_RESOLVER_MIN_CONFIDENCE,
) -> ResolverOutcome:
    """Resolve one target; any resolver/parse failure safely selects the original."""
    try:
        raw_response = resolver_call(request)
    except Exception:
        return ResolverOutcome(
            request=request,
            raw_response="",
            resolution=None,
            decision=ClarificationDecision(
                request.target_source,
                False,
                "resolver_failure",
            ),
            resolver_failed=True,
        )

    try:
        resolution = parse_semantic_resolution(raw_response)
    except ValueError:
        return ResolverOutcome(
            request=request,
            raw_response=raw_response,
            resolution=None,
            decision=ClarificationDecision(
                request.target_source,
                False,
                "malformed_json",
            ),
            malformed_json=True,
        )

    return ResolverOutcome(
        request=request,
        raw_response=raw_response,
        resolution=resolution,
        decision=decide_clarification(
            request,
            resolution,
            min_confidence=min_confidence,
        ),
    )


def decide_translation_risk_rewrite(
    request: SemanticContextRequest,
    resolution: TranslationRiskResolution,
    *,
    min_confidence: float = DEFAULT_SEMANTIC_RESOLVER_MIN_CONFIDENCE,
) -> RewriteDecision:
    """Apply the V2 rewrite-needed, confidence, and deterministic safety gates."""
    original = request.target_source
    clarified = resolution.clarified_target.strip()
    if not resolution.rewrite_needed:
        return RewriteDecision(original, False, "rewrite_not_needed")
    if clarified == original:
        return RewriteDecision(original, False, "unchanged_target")
    if resolution.confidence < min_confidence:
        return RewriteDecision(original, False, "low_confidence")

    failures = list(
        validate_clarified_target(
            original,
            clarified,
            named_terms=request.named_terms,
            context_sources=(*request.previous_context, *request.next_context),
        )
    )
    if resolution.question_type != detect_question_type(original):
        failures.append("declared_question_type_mismatch")
    failures = list(dict.fromkeys(failures))
    if failures:
        return RewriteDecision(
            original,
            False,
            "rewrite_validation_failed",
            tuple(failures),
        )
    return RewriteDecision(clarified, True, None)


def resolve_translation_risk_with_fallback(
    request: SemanticContextRequest,
    resolver_call: Callable[[SemanticContextRequest], str],
    *,
    min_confidence: float = DEFAULT_SEMANTIC_RESOLVER_MIN_CONFIDENCE,
) -> TranslationRiskOutcome:
    """Resolve one V2 target; failures always fall back to the original target."""
    try:
        raw_response = resolver_call(request)
    except Exception:
        return TranslationRiskOutcome(
            request=request,
            raw_response="",
            resolution=None,
            decision=RewriteDecision(
                request.target_source,
                False,
                "resolver_failure",
            ),
            resolver_failed=True,
        )

    try:
        resolution = parse_translation_risk_resolution(raw_response)
    except ValueError:
        return TranslationRiskOutcome(
            request=request,
            raw_response=raw_response,
            resolution=None,
            decision=RewriteDecision(
                request.target_source,
                False,
                "malformed_json",
            ),
            malformed_json=True,
        )

    return TranslationRiskOutcome(
        request=request,
        raw_response=raw_response,
        resolution=resolution,
        decision=decide_translation_risk_rewrite(
            request,
            resolution,
            min_confidence=min_confidence,
        ),
    )


def render_controlled_english_bridge_prompt(request: SemanticContextRequest) -> str:
    """Render the exact V3 controlled-English translation bridge prompt."""
    return (
        "You are an English controlled-language bridge for a downstream machine translation system.\n\n"
        "You do NOT translate into Turkish.\n\n"
        "The downstream translator sometimes chooses the wrong meaning for English words, "
        "phrasal verbs, pronouns, question constructions, tense/aspect, or compact narrative expressions.\n\n"
        "Your job is to decide whether the TARGET should be rewritten into simpler, "
        "more explicit English before translation.\n\n"
        "Use nearby context only to resolve meaning already contained in the TARGET.\n\n"
        "Do not add new story information.\n"
        "Do not add motivations.\n"
        "Do not add locations.\n"
        "Do not add identities.\n"
        "Do not add lore.\n"
        "Do not explain the story.\n"
        "Do not improve literary style.\n"
        "Do not summarize.\n"
        "Do not translate proper names.\n"
        "Do not rename abilities or terms.\n\n"
        "IMPORTANT:\n\n"
        "Do not merely replace a risky word with another similarly ambiguous synonym.\n\n"
        "When a rewrite is necessary, prefer a simple explicit English construction "
        "that expresses the intended semantic relation directly.\n\n"
        "Example:\n\n"
        "TARGET:\n"
        "Frost Chain can hold three targets at once.\n\n"
        "If context clearly shows the targets cannot move:\n\n"
        "GOOD:\n"
        "Frost Chain can keep three targets from moving at the same time.\n\n"
        "LESS PREFERRED:\n"
        "Frost Chain can restrain three targets at once.\n\n"
        "The first version is preferred because it explicitly states the semantic relation.\n\n"
        "Another example:\n\n"
        "TARGET:\n"
        "Who charged the meal to my room?\n\n"
        "If context clearly refers to a hotel bill:\n\n"
        "GOOD:\n"
        "Who added the cost of the meal to my room bill?\n\n"
        "Do not change WHO into HOW.\n\n"
        "Do not rewrite when the original sentence is already sufficiently clear "
        "for machine translation.\n\n"
        "Return JSON only.\n"
        "No Markdown.\n"
        "No commentary outside JSON.\n\n"
        f"PREVIOUS CONTEXT:\n{_format_context(request.previous_context)}\n\n"
        f"TARGET:\n{request.target_source}\n\n"
        f"NEXT CONTEXT:\n{_format_context(request.next_context)}\n\n"
        "Required JSON schema:\n\n"
        "{\n"
        '  "rewrite_needed": true,\n'
        '  "confidence": 0.0,\n'
        '  "risk_types": [],\n'
        '  "semantic_notes": [\n'
        "    {\n"
        '      "span": "...",\n'
        '      "resolved_meaning": "...",\n'
        '      "evidence": "..."\n'
        "    }\n"
        "  ],\n"
        '  "question_word": null,\n'
        '  "referents": [],\n'
        '  "tense_aspect": null,\n'
        '  "controlled_target": "..."\n'
        "}\n\n"
        "Rules:\n\n"
        "If rewriting is not materially useful:\n"
        "rewrite_needed = false\n"
        "controlled_target MUST equal TARGET exactly.\n\n"
        "If context is insufficient:\n"
        "rewrite_needed = false\n"
        "confidence must be low\n"
        "controlled_target MUST equal TARGET exactly.\n\n"
        "If rewriting is useful and context clearly establishes the intended meaning:\n"
        "rewrite_needed = true\n"
        "confidence should reflect certainty\n"
        "controlled_target must use simple explicit English\n"
        "and preserve the exact proposition of TARGET."
    )


def parse_controlled_english_resolution(
    raw_response: str,
) -> ControlledEnglishResolution:
    """Parse strict V3 JSON response with resilient schema matching and question_word tolerance."""
    clean_text = raw_response.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    try:
        payload = json.loads(clean_text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("malformed_controlled_english_json") from exc

    if not isinstance(payload, dict):
        raise ValueError("controlled_english_json_must_be_object")

    if "controlled_target" not in payload:
        raise ValueError("controlled_english_missing_target")

    rewrite_needed = payload.get("rewrite_needed", False)
    confidence = payload.get("confidence", 0.0)
    risk_types_payload = payload.get("risk_types", [])
    notes_payload = payload.get("semantic_notes", [])
    q_word = payload.get("question_word") if "question_word" in payload else payload.get("question_type")
    tense_aspect = payload.get("tense_aspect")
    referents = payload.get("referents", [])
    controlled_target = payload.get("controlled_target", "")

    if not isinstance(rewrite_needed, bool):
        raise ValueError("controlled_english_rewrite_needed_must_be_boolean")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("controlled_english_confidence_must_be_number")
    confidence_float = float(confidence)
    if not 0.0 <= confidence_float <= 1.0:
        raise ValueError("controlled_english_confidence_out_of_range")

    if not isinstance(risk_types_payload, list):
        risk_types_payload = []
    risk_types = [str(r) for r in risk_types_payload if isinstance(r, str)]

    notes: list[ControlledEnglishNote] = []
    if isinstance(notes_payload, list):
        for note in notes_payload:
            if isinstance(note, dict):
                span = str(note.get("span", ""))
                res = str(note.get("resolved_meaning", note.get("intended_sense", "")))
                ev = str(note.get("evidence", ""))
                notes.append(ControlledEnglishNote(span=span, resolved_meaning=res, evidence=ev))

    norm_q_word: str | None = None
    if q_word is not None and isinstance(q_word, str):
        qw_clean = q_word.strip().lower()
        if qw_clean in {"who", "whom", "whose"}:
            norm_q_word = "who"
        elif qw_clean in {"what", "why", "when", "where", "how", "which"}:
            norm_q_word = qw_clean

    if tense_aspect is not None and not isinstance(tense_aspect, str):
        tense_aspect = None
    if not isinstance(referents, list):
        referents = []
    if not isinstance(controlled_target, str):
        raise ValueError("controlled_english_target_must_be_string")

    return ControlledEnglishResolution(
        rewrite_needed=rewrite_needed,
        confidence=confidence_float,
        risk_types=tuple(dict.fromkeys(risk_types)),
        semantic_notes=tuple(notes),
        question_word=norm_q_word,
        tense_aspect=tense_aspect.strip() if tense_aspect else None,
        referents=tuple(referents),
        controlled_target=controlled_target,
    )


def validate_controlled_target(
    original_target: str,
    controlled_target: str,
    *,
    named_terms: Iterable[str] = (),
    context_sources: Iterable[str] = (),
) -> tuple[tuple[str, ...], bool]:
    """Return generic deterministic safety failures for a V3 proposed English target."""
    original = original_target.strip()
    controlled = controlled_target.strip()
    failures: list[str] = []
    validator_uncertain = False

    if not controlled:
        return ("empty_controlled_target",), False
    if _looks_non_english(controlled):
        failures.append("controlled_target_not_english")
    if _contains_structured_garbage(controlled):
        failures.append("structured_or_control_garbage")

    protected_identities = {
        term.strip()
        for term in named_terms
        if term.strip() and _contains_phrase(original, term.strip())
    }
    protected_identities.update(_SENTINEL_PATTERN.findall(original))
    if any(not _contains_exact_phrase(controlled, term) for term in protected_identities):
        failures.append("named_term_loss")

    if _number_tokens(original) != _number_tokens(controlled):
        failures.append("number_changed")

    orig_neg = _negation_tokens(original)
    ctrl_neg = _negation_tokens(controlled)

    has_orig_neg = len(orig_neg) > 0
    has_ctrl_neg = len(ctrl_neg) > 0

    if has_orig_neg != has_ctrl_neg:
        validator_uncertain = True
        failures.append("polarity_changed")

    original_question = detect_question_type(original)
    controlled_question = detect_question_type(controlled)
    if original_question != controlled_question:
        failures.append("question_type_changed")

    original_terminal = _terminal_punctuation(original)
    controlled_terminal = _terminal_punctuation(controlled)
    if (
        original_terminal in {"?", "!"}
        or controlled_terminal in {"?", "!"}
    ) and original_terminal != controlled_terminal:
        failures.append("terminal_punctuation_changed")

    original_word_count = max(1, len(_WORD_PATTERN.findall(original)))
    controlled_word_count = len(_WORD_PATTERN.findall(controlled))
    max_words = max(original_word_count + 12, math.ceil(original_word_count * 2.2))
    if controlled_word_count > max_words:
        failures.append("controlled_target_too_long")

    context_tuple = tuple(source.strip() for source in context_sources if source.strip())
    if _tense_aspect_signature(original) != _tense_aspect_signature(controlled):
        failures.append("tense_aspect_changed")
    if _copies_context_sentence(original, controlled, context_tuple):
        failures.append("context_sentence_copied")
    if _unsupported_named_entities(original, controlled, context_tuple):
        failures.append("unsupported_named_entity")

    return tuple(dict.fromkeys(failures)), validator_uncertain


def decide_controlled_bridge(
    request: SemanticContextRequest,
    resolution: ControlledEnglishResolution,
    *,
    min_confidence: float = DEFAULT_SEMANTIC_RESOLVER_MIN_CONFIDENCE,
) -> ControlledBridgeDecision:
    """Apply V3 confidence, controlled-English validation, and safety gates."""
    original = request.target_source
    controlled = resolution.controlled_target.strip()

    if not resolution.rewrite_needed:
        return ControlledBridgeDecision(original, False, "rewrite_not_needed")
    if controlled == original:
        return ControlledBridgeDecision(original, False, "unchanged_target")
    if resolution.confidence < min_confidence:
        return ControlledBridgeDecision(original, False, "low_confidence")

    failures, validator_uncertain = validate_controlled_target(
        original,
        controlled,
        named_terms=request.named_terms,
        context_sources=(*request.previous_context, *request.next_context),
    )

    if failures:
        return ControlledBridgeDecision(
            selected_target=original,
            rewrite_used=False,
            rejection_reason="controlled_validation_failed",
            validation_failures=failures,
            validator_uncertain=validator_uncertain,
        )

    return ControlledBridgeDecision(
        selected_target=controlled,
        rewrite_used=True,
        rejection_reason=None,
        validation_failures=(),
        validator_uncertain=validator_uncertain,
    )


def resolve_controlled_bridge_with_fallback(
    request: SemanticContextRequest,
    resolver_call: Callable[[SemanticContextRequest], str],
    *,
    min_confidence: float = DEFAULT_SEMANTIC_RESOLVER_MIN_CONFIDENCE,
) -> ControlledBridgeOutcome:
    """Resolve one V3 target; any failure safely selects original target."""
    try:
        raw_response = resolver_call(request)
    except Exception:
        return ControlledBridgeOutcome(
            request=request,
            raw_response="",
            resolution=None,
            decision=ControlledBridgeDecision(
                request.target_source,
                False,
                "resolver_failure",
            ),
            resolver_failed=True,
        )

    try:
        resolution = parse_controlled_english_resolution(raw_response)
    except ValueError:
        return ControlledBridgeOutcome(
            request=request,
            raw_response=raw_response,
            resolution=None,
            decision=ControlledBridgeDecision(
                request.target_source,
                False,
                "malformed_json",
            ),
            malformed_json=True,
        )

    return ControlledBridgeOutcome(
        request=request,
        raw_response=raw_response,
        resolution=resolution,
        decision=decide_controlled_bridge(
            request,
            resolution,
            min_confidence=min_confidence,
        ),
    )
