"""Generic glossary proposal domain logic, models, and quality validation.

Decoupled from LLM provider. Formulates multi-option Turkish target proposals for unconfirmed
terminology and titles, keeping candidates provisional and keeping SeriesProfile untouched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from core.translation.profile_discovery import CandidateStore, ProfileCandidate
from core.translation.series_profile import SeriesProfile

UNSUPPORTED_CLAIM_PATTERNS = [
    r"yaygın olarak",
    r"genellikle çevrilir",
    r"türk webtoon",
    r"standart çeviri",
    r"çevirilerinde",
]


@dataclass
class GlossaryProposal:
    """A proposed multi-option Turkish target translation for a candidate entity or term."""

    source: str
    kind: str
    options: list[str] = field(default_factory=list)
    preferred_target: str = ""
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    requires_review: bool = False
    is_valid: bool = True

    @property
    def suggested_target(self) -> str:
        """Backward compatibility property returning preferred_target."""
        return self.preferred_target

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "kind": self.kind,
            "options": list(self.options),
            "preferred_target": self.preferred_target,
            "suggested_target": self.preferred_target,
            "reason": self.reason,
            "warnings": list(self.warnings),
            "requires_review": self.requires_review,
            "is_valid": self.is_valid,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GlossaryProposal:
        raw_options = data.get("options", [])
        options = (
            [str(opt).strip() for opt in raw_options if str(opt).strip()]
            if isinstance(raw_options, list)
            else []
        )
        pref = str(data.get("preferred_target", data.get("suggested_target", ""))).strip()
        raw_warnings = data.get("warnings", [])
        warnings = (
            [str(w) for w in raw_warnings]
            if isinstance(raw_warnings, list)
            else []
        )

        return cls(
            source=str(data.get("source", "")).strip(),
            kind=str(data.get("kind", "term")).strip(),
            options=options,
            preferred_target=pref,
            reason=str(data.get("reason", "")).strip(),
            warnings=warnings,
            requires_review=bool(data.get("requires_review", False)),
            is_valid=bool(data.get("is_valid", True)),
        )


def validate_glossary_proposal(prop: GlossaryProposal) -> tuple[bool, list[str]]:
    """Deterministically validate a glossary proposal and attach quality warnings.

    Returns (is_valid, warnings).
    """
    warnings: list[str] = list(prop.warnings)
    is_valid = True

    cleaned_source = prop.source.strip()
    pref_target = prop.preferred_target.strip()

    # 1. Hard Structural Failure: Empty preferred target
    if not pref_target:
        warnings.append("empty_preferred_target")
        prop.is_valid = False
        prop.requires_review = True
        return False, warnings

    # 2. Options Deduplication (case-insensitive) without silent mutation of missing preferred_target
    deduped_options: list[str] = []
    seen_lower = set()
    for opt in prop.options:
        opt_str = opt.strip()
        if opt_str and opt_str.lower() not in seen_lower:
            deduped_options.append(opt_str)
            seen_lower.add(opt_str.lower())
    prop.options = deduped_options

    if not prop.options:
        warnings.append("no_options")
        prop.is_valid = False
        prop.requires_review = True
        return False, warnings

    # 3. Hard Structural Failure: Preferred target NOT in options (No silent self-correction!)
    if pref_target.lower() not in seen_lower:
        warnings.append("preferred_not_in_options")
        prop.is_valid = False
        prop.requires_review = True
        is_valid = False

    # 4. Hard Structural Failure: CJK hallucination
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", pref_target):
        warnings.append("cjk_hallucination")
        prop.is_valid = False
        prop.requires_review = True
        return False, warnings

    # 5. Hard Structural Failure: Punctuation only
    if not re.search(r"[A-Za-z0-9\u00c0-\u024f]", pref_target):
        warnings.append("punctuation_only")
        prop.is_valid = False
        prop.requires_review = True
        return False, warnings

    # 6. Hard Structural Failure: Excessively long target
    if len(pref_target) > 100:
        warnings.append("excessively_long_target")
        prop.is_valid = False
        prop.requires_review = True
        return False, warnings

    # 7. Soft Warning: Source language leakage check (untranslated English content tokens)
    # Exclude character_name and place_name (proper-name preservation is expected)
    if prop.kind not in ("character_name", "place_name"):
        src_tokens = re.findall(r"[A-Za-z]{4,}", cleaned_source)
        for tok in src_tokens:
            pattern = r"(?<![A-Za-z0-9])" + re.escape(tok) + r"(?![A-Za-z0-9])"
            if re.search(pattern, pref_target, re.IGNORECASE):
                warnings.append(f"possible_source_language_leak_{tok.upper()}")
                prop.requires_review = True
                break

    # 8. Soft Warning: Unsupported external claim phrases in reason
    if prop.reason:
        reason_lower = prop.reason.lower()
        for pat in UNSUPPORTED_CLAIM_PATTERNS:
            if re.search(pat, reason_lower):
                warnings.append("unsupported_external_claim")
                prop.requires_review = True
                break

    prop.warnings = list(dict.fromkeys(warnings))
    prop.is_valid = is_valid
    return is_valid, prop.warnings


def select_candidates_for_proposal(
    candidate_store: CandidateStore,
    existing_profile: SeriesProfile | None = None,
) -> list[ProfileCandidate]:
    """Select candidates eligible for Turkish target proposal."""
    confirmed_keys = set()
    if existing_profile:
        confirmed_keys.update(k.upper() for k in existing_profile.known_names)
        confirmed_keys.update(k.upper() for k in existing_profile.glossary)

    eligible: list[ProfileCandidate] = []
    for key, cand in candidate_store.candidates.items():
        if cand.kind == "character_name":
            continue
        if cand.status == "rejected":
            continue
        if key in confirmed_keys:
            continue
        eligible.append(cand)

    return eligible


def apply_glossary_proposals(
    candidate_store: CandidateStore,
    proposals: list[GlossaryProposal],
    profile: SeriesProfile | None = None,
) -> list[ProfileCandidate]:
    """Attach proposals to candidates in CandidateStore.

    CRITICAL SAFETY RULES:
    1. Candidate status MUST remain 'provisional'.
    2. SeriesProfile glossary/known_names are NEVER modified by proposals.
    """
    updated_candidates: list[ProfileCandidate] = []

    for prop in proposals:
        key = prop.source.strip().upper()
        cand = candidate_store.candidates.get(key)
        if cand and cand.status != "rejected":
            cand.suggested_target = prop.preferred_target
            if cand.status != "confirmed":
                cand.status = "provisional"
            updated_candidates.append(cand)

    logger.info(f"Applied {len(proposals)} proposals to candidate store '{candidate_store.series_id}'. Candidates remain provisional.")
    return updated_candidates
