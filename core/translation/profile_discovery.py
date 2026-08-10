"""Generic SeriesProfile candidate discovery domain models and logic.

Decoupled from any LLM provider. Handles candidate storage, deterministic
evidence validation, merging, evidence accumulation, JSON persistence, and confirmation.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from core.translation.series_profile import SeriesProfile, DEFAULT_PROFILES_DIR
from providers.translation.base import TranslationItem

VALID_KINDS = {"character_name", "place_name", "title_or_rank", "term"}
VALID_STATUSES = {"discovered", "provisional", "ready_for_review", "approved", "confirmed", "rejected"}


@dataclass
class TermObservation:
    """An observed translation occurrence of a source term to a target surface span."""

    chapter_id: str
    region_id: int
    source_text: str
    translated_text: str
    observed_target_form: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "region_id": self.region_id,
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "observed_target_form": self.observed_target_form,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TermObservation:
        return cls(
            chapter_id=str(data.get("chapter_id", "")),
            region_id=int(data.get("region_id", 0)),
            source_text=str(data.get("source_text", "")),
            translated_text=str(data.get("translated_text", "")),
            observed_target_form=str(data.get("observed_target_form", "")),
        )


def candidate_phrase_pattern(source_phrase: str) -> re.Pattern[str] | None:
    """Build the shared safe matcher for a canonical English source term.

    The returned pattern owns both boundary handling and the matched English
    inflectional suffix.  Relevance checks and source protection therefore see
    exactly the same span for ``Spirit Stone``, ``Spirit Stones`` and
    ``Spirit Stone's`` without allowing ``YU`` to match inside ``YOU``.
    """
    if not source_phrase:
        return None
    cleaned_phrase = source_phrase.strip()
    if not cleaned_phrase:
        return None
    pattern = (
        r"(?<![A-Za-z0-9_])"
        + re.escape(cleaned_phrase)
        + r"(?P<english_suffix>'s|es|s)?(?![A-Za-z0-9_])"
    )
    return re.compile(pattern, re.IGNORECASE)


def find_candidate_phrase_matches(source_phrase: str, text: str) -> list[re.Match[str]]:
    """Return all non-overlapping spans produced by the shared term matcher."""
    if not text:
        return []
    pattern = candidate_phrase_pattern(source_phrase)
    return list(pattern.finditer(text)) if pattern else []


def contains_candidate_phrase(source_phrase: str, text: str) -> bool:
    """Check source-term relevance using the shared safe span matcher."""
    return bool(find_candidate_phrase_matches(source_phrase, text))


@dataclass
class CandidateEvidence:
    """Evidence occurrence of a discovered candidate."""

    chapter_id: str
    region_id: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "region_id": self.region_id,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateEvidence:
        return cls(
            chapter_id=str(data.get("chapter_id", "")),
            region_id=int(data.get("region_id", 0)),
            text=str(data.get("text", "")),
        )


@dataclass
class ProfileCandidate:
    """A discovered name, place, title, or terminology candidate."""

    source: str
    kind: str
    suggested_target: str | None = None
    status: str = "discovered"
    evidence_count: int = 0
    evidence: list[CandidateEvidence] = field(default_factory=list)
    observations: list[TermObservation] = field(default_factory=list)

    @property
    def observed_target_counts(self) -> dict[str, int]:
        """Return counts of observed target surface forms (normalized lowercase)."""
        counts: dict[str, int] = {}
        for obs in self.observations:
            norm = obs.observed_target_form.strip().lower()
            if norm:
                counts[norm] = counts.get(norm, 0) + 1
        return counts

    def add_observation(self, obs: TermObservation) -> bool:
        """Add a translation observation with strict deduplication identity.

        Deduplication key: (chapter_id, region_id, canonical_source_term).
        Same region with two different terms produce two distinct observations.
        Re-running exact same region for same term does NOT add duplicate evidence.
        Returns True if a new observation was added.
        """
        canonical_key = self.source.strip().upper()
        existing_keys = {(o.chapter_id, o.region_id, canonical_key) for o in self.observations}
        if (obs.chapter_id, obs.region_id, canonical_key) in existing_keys:
            return False

        self.observations.append(obs)
        self.evidence_count = len(self.observations)
        self._update_lifecycle_status()
        return True

    def _update_lifecycle_status(self) -> None:
        """Update candidate lifecycle status based on observation evidence."""
        if self.status in ("approved", "confirmed", "rejected"):
            return

        if not self.observations:
            self.status = "discovered"
            return

        if len(self.observations) == 1:
            self.status = "provisional"
            return

        # 2+ observations: check if independent regions exist and forms are consistent
        unique_regions = {(o.chapter_id, o.region_id) for o in self.observations}
        if len(unique_regions) >= 2:
            counts = self.observed_target_counts
            if counts:
                max_count = max(counts.values())
                # If dominant target form represents majority of observations
                if max_count >= 2 and (max_count / len(self.observations)) >= 0.5:
                    self.status = "ready_for_review"
                    return

        self.status = "provisional"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "kind": self.kind,
            "suggested_target": self.suggested_target,
            "status": self.status,
            "evidence_count": self.evidence_count,
            "evidence": [e.to_dict() for e in self.evidence],
            "observations": [o.to_dict() for o in self.observations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileCandidate:
        raw_evidence = data.get("evidence", [])
        evidence_list = (
            [CandidateEvidence.from_dict(e) for e in raw_evidence if isinstance(e, dict)]
            if isinstance(raw_evidence, list)
            else []
        )
        raw_obs = data.get("observations", [])
        obs_list = (
            [TermObservation.from_dict(o) for o in raw_obs if isinstance(o, dict)]
            if isinstance(raw_obs, list)
            else []
        )
        status = str(data.get("status", "discovered"))
        # Map legacy 'confirmed' to 'approved' for consistency
        if status == "confirmed":
            status = "approved"

        return cls(
            source=str(data.get("source", "")),
            kind=str(data.get("kind", "term")),
            suggested_target=data.get("suggested_target"),
            status=status,
            evidence_count=int(data.get("evidence_count", len(obs_list) or len(evidence_list))),
            evidence=evidence_list,
            observations=obs_list,
        )


@dataclass
class CandidateStore:
    """Container and manager for series profile candidates."""

    series_id: str
    candidates: dict[str, ProfileCandidate] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "candidates": {k: v.to_dict() for k, v in self.candidates.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateStore:
        if not isinstance(data, dict):
            return cls(series_id="unknown")

        series_id = str(data.get("series_id", "unknown"))
        raw_candidates = data.get("candidates", {})
        candidates = {}

        if isinstance(raw_candidates, dict):
            for k, v in raw_candidates.items():
                if isinstance(v, dict):
                    cand = ProfileCandidate.from_dict(v)
                    candidates[k.strip().upper()] = cand

        return cls(series_id=series_id, candidates=candidates)

    @classmethod
    def load_from_json(cls, file_path_or_series_id: str | Path) -> CandidateStore:
        """Load CandidateStore from JSON file. Returns empty store if missing or malformed."""
        path = Path(file_path_or_series_id)
        if not path.is_file():
            # If passed a series_id or path in DEFAULT_PROFILES_DIR
            if not str(file_path_or_series_id).endswith(".json"):
                path = DEFAULT_PROFILES_DIR / f"{file_path_or_series_id}.candidates.json"

        if not path.exists():
            logger.info(f"Candidate store file not found at {path}; using empty store")
            return cls(series_id=path.stem.replace(".candidates", ""))

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            store = cls.from_dict(data)
            logger.info(f"Loaded candidate store '{store.series_id}' ({len(store.candidates)} candidates)")
            return store
        except Exception as e:
            logger.warning(f"Failed to load candidate store from {path}: {e}")
            return cls(series_id=path.stem.replace(".candidates", ""))

    def save_to_json(
        self, file_path: str | Path | None = None, base_dir: str | Path = DEFAULT_PROFILES_DIR
    ) -> Path:
        """Save CandidateStore using atomic safe write."""
        if file_path is None:
            target_path = Path(base_dir) / f"{self.series_id}.candidates.json"
        else:
            target_path = Path(file_path)

        target_path.parent.mkdir(parents=True, exist_ok=True)

        temp_fd, temp_file_path = tempfile.mkstemp(
            dir=target_path.parent, prefix=f"{target_path.stem}_", suffix=".tmp"
        )
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(temp_file_path, target_path)
            logger.info(f"Saved candidate store '{self.series_id}' to {target_path}")
        except Exception as e:
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass
            logger.error(f"Failed to save candidate store to {target_path}: {e}")
            raise

        return target_path


@dataclass
class DiscoveryResult:
    """Result summary of a candidate discovery run."""

    candidates: list[ProfileCandidate]
    filtered_count: int = 0
    warnings: list[str] = field(default_factory=list)


def validate_candidate_suggestion(
    source: str,
    kind: str,
    items: list[TranslationItem],
) -> tuple[bool, str | None]:
    """Deterministically validate a proposed candidate suggestion.

    Returns (is_valid, rejection_reason).
    """
    cleaned_source = source.strip()
    if not cleaned_source:
        return False, "empty_source"

    if len(cleaned_source) > 50:
        return False, "source_too_long"

    if kind not in VALID_KINDS:
        return False, f"invalid_kind_{kind}"

    # CJK check (if source contains East Asian characters when English input expected)
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", cleaned_source):
        return False, "cjk_hallucination"

    # Punctuation-only
    if not re.search(r"[A-Za-z0-9]", cleaned_source):
        return False, "punctuation_only"

    # Deterministic source verification: candidate source MUST exist in clean input text
    found_in_text = False
    for item in items:
        if item.source and contains_candidate_phrase(cleaned_source, item.source):
            found_in_text = True
            break

    if not found_in_text:
        return False, "not_found_in_source_text"

    return True, None


def process_discovered_suggestions(
    raw_suggestions: list[dict[str, Any]],
    items: list[TranslationItem],
    chapter_id: str,
    candidate_store: CandidateStore,
    existing_profile: SeriesProfile | None = None,
) -> DiscoveryResult:
    """Process raw candidate suggestions through deterministic verification and merge into store.

    Guarantees that discovered candidates are saved ONLY as provisional (or merged with existing),
    and NEVER automatically update confirmed SeriesProfile files.
    """
    valid_candidates: list[ProfileCandidate] = []
    filtered_count = 0
    warnings: list[str] = []

    # Map items by region_id for evidence lookups
    items_by_id = {item.region_id: item for item in items}
    all_source_items = [item for item in items if item.source]

    # Pre-fetch confirmed keys from profile to avoid duplicate provisional entries
    confirmed_names = (
        {k.upper(): v for k, v in existing_profile.known_names.items()}
        if existing_profile
        else {}
    )
    confirmed_glossary = (
        {k.upper(): v for k, v in existing_profile.glossary.items()}
        if existing_profile
        else {}
    )

    for suggestion in raw_suggestions:
        if not isinstance(suggestion, dict):
            filtered_count += 1
            continue

        raw_src = str(suggestion.get("source", "")).strip()
        kind = str(suggestion.get("kind", "term")).strip().lower()
        suggested_target = suggestion.get("suggested_target")
        if suggested_target is not None:
            suggested_target = str(suggested_target).strip()

        # Discovery narrow scope: terms and titles/ranks must NOT have Turkish translations attached during discovery
        if kind in ("term", "title_or_rank"):
            suggested_target = None

        evidence_ids = suggestion.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            evidence_ids = []

        is_valid, reason = validate_candidate_suggestion(raw_src, kind, all_source_items)
        if not is_valid:
            filtered_count += 1
            warnings.append(f"Filtered candidate '{raw_src}': {reason}")
            continue

        key = raw_src.upper()

        # Collect evidence instances
        evidences: list[CandidateEvidence] = []
        for eid in evidence_ids:
            item = items_by_id.get(eid)
            if item and item.source:
                evidences.append(
                    CandidateEvidence(
                        chapter_id=chapter_id,
                        region_id=item.region_id,
                        text=item.source,
                    )
                )

        # Fallback evidence if model didn't supply valid evidence_ids
        if not evidences:
            for item in all_source_items:
                if contains_candidate_phrase(raw_src, item.source):
                    evidences.append(
                        CandidateEvidence(
                            chapter_id=chapter_id,
                            region_id=item.region_id,
                            text=item.source,
                        )
                    )

        # 1. If already confirmed in SeriesProfile, update usage evidence in store if present, but do not create duplicate provisional
        if key in confirmed_names or key in confirmed_glossary:
            if key in candidate_store.candidates:
                existing_cand = candidate_store.candidates[key]
                _append_evidences(existing_cand, evidences)
            continue

        # 2. Check candidate_store
        if key in candidate_store.candidates:
            existing_cand = candidate_store.candidates[key]
            if existing_cand.status == "rejected":
                # Rejected candidate remains rejected
                _append_evidences(existing_cand, evidences)
            else:
                _append_evidences(existing_cand, evidences)
                if suggested_target and not existing_cand.suggested_target:
                    existing_cand.suggested_target = suggested_target
                valid_candidates.append(existing_cand)
        else:
            new_cand = ProfileCandidate(
                source=raw_src,
                kind=kind,
                suggested_target=suggested_target,
                status="discovered",
                evidence_count=len(evidences),
                evidence=evidences,
            )
            candidate_store.candidates[key] = new_cand
            valid_candidates.append(new_cand)

    logger.info(
        f"Discovery processed {len(raw_suggestions)} suggestions -> {len(valid_candidates)} candidates added/updated, {filtered_count} filtered"
    )


    return DiscoveryResult(
        candidates=valid_candidates,
        filtered_count=filtered_count,
        warnings=warnings,
    )


def _append_evidences(candidate: ProfileCandidate, new_evidences: list[CandidateEvidence]) -> None:
    existing_keys = {(e.chapter_id, e.region_id) for e in candidate.evidence}
    for ev in new_evidences:
        if (ev.chapter_id, ev.region_id) not in existing_keys:
            candidate.evidence.append(ev)
            existing_keys.add((ev.chapter_id, ev.region_id))
    candidate.evidence_count = len(candidate.evidence)


def confirm_candidate(
    candidate_store: CandidateStore,
    profile: SeriesProfile,
    source: str,
    target_override: str | None = None,
) -> bool:
    """Manually confirm a provisional candidate and transfer it into confirmed SeriesProfile data.

    Returns True if confirmed successfully.
    """
    key = source.strip().upper()
    candidate = candidate_store.candidates.get(key)
    if not candidate:
        logger.warning(f"Cannot confirm '{source}': not found in candidate store")
        return False

    target = target_override or candidate.suggested_target or candidate.source

    if candidate.kind in ("character_name", "place_name"):
        profile.known_names[key] = target
    else:  # title_or_rank or term
        profile.glossary[key] = target

    candidate.status = "approved"
    logger.info(f"Confirmed candidate '{key}' as '{target}' ({candidate.kind}) into profile '{profile.series_id}'")
    return True


def reject_candidate(
    candidate_store: CandidateStore,
    source: str,
) -> bool:
    """Mark a candidate as rejected in the candidate store."""
    key = source.strip().upper()
    candidate = candidate_store.candidates.get(key)
    if not candidate:
        logger.warning(f"Cannot reject '{source}': not found in candidate store")
        return False

    candidate.status = "rejected"
    logger.info(f"Rejected candidate '{key}' in store '{candidate_store.series_id}'")
    return True


def get_relevant_terms_for_item(
    source_text: str,
    profile: SeriesProfile | None = None,
    candidate_store: CandidateStore | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Retrieve approved terms (guidance constraints) and provisional terms (observation targets) relevant to a single source dialogue item.

    Returns (approved_terms: dict[src_key, target_str], provisional_terms: list[src_key]).
    Per-item retrieval ensures that terms NOT present in source_text are never injected into the prompt.
    """
    if not source_text or not source_text.strip():
        return {}, []

    approved_terms: dict[str, str] = {}
    provisional_terms: list[str] = []

    # 1. Approved terms from SeriesProfile
    if profile:
        for k, v in profile.known_names.items():
            if contains_candidate_phrase(k, source_text):
                approved_terms[k.strip().upper()] = v.strip()
        for k, v in profile.glossary.items():
            if contains_candidate_phrase(k, source_text):
                approved_terms[k.strip().upper()] = v.strip()

    # 2. Provisional/Discovered/Ready-for-review terms from CandidateStore
    if candidate_store:
        for k, cand in candidate_store.candidates.items():
            key = k.strip().upper()
            if key in approved_terms:
                continue
            if cand.status in ("discovered", "provisional", "ready_for_review"):
                if contains_candidate_phrase(cand.source, source_text):
                    provisional_terms.append(cand.source.strip())

    return approved_terms, provisional_terms


@dataclass
class TermUsageValidation:
    """Result of validating a term alignment."""

    valid: bool
    source_term: str
    target_form: str | None
    warning: str | None = None


def _norm_turk_text(text: str) -> str:
    """Normalize text for Turkish case-insensitive comparison (handles dotted I correctly)."""
    if not text:
        return ""
    return text.replace("İ", "i").replace("I", "ı").lower()


def validate_term_alignment(
    source_term: str,
    source_text: str,
    target_form: str | None,
    translated_text: str,
) -> TermUsageValidation:
    """Deterministically validate a reported or extracted term alignment.

    Safety Rules:
    1. source_term must be present in source_text.
    2. target_form must be non-empty and present as exact text span in translated_text.
    3. Partial multiword alignment guard:
       If source_term contains 2+ English word tokens, and target_form (case-insensitive) is EQUAL to
       an isolated token of source_term (e.g. MANA CORE -> Mana), return valid=False, warning="partial_term_alignment".
    """
    if not source_term or not source_term.strip():
        return TermUsageValidation(valid=False, source_term="", target_form=target_form, warning="missing_source_term")

    src_term = source_term.strip()

    # 1. Source term MUST appear in source text
    if not contains_candidate_phrase(src_term, source_text):
        return TermUsageValidation(valid=False, source_term=src_term, target_form=target_form, warning="source_term_not_in_source")

    if not target_form or not target_form.strip() or not translated_text or not translated_text.strip():
        return TermUsageValidation(valid=False, source_term=src_term, target_form=target_form, warning="missing_target_span")

    tgt_form = target_form.strip()

    # 2. Target form MUST exist in translated_text (using Turkish case normalization)
    if _norm_turk_text(tgt_form) not in _norm_turk_text(translated_text):
        return TermUsageValidation(valid=False, source_term=src_term, target_form=tgt_form, warning="ungrounded_target_span")

    # 3. Partial multiword term alignment guard
    src_tokens = [w.strip().lower() for w in re.findall(r"[A-Za-z0-9]+", src_term) if len(w) >= 2]
    if len(src_tokens) >= 2:
        norm_tgt = _norm_turk_text(tgt_form)
        if norm_tgt in src_tokens:
            return TermUsageValidation(
                valid=False,
                source_term=src_term,
                target_form=tgt_form,
                warning="partial_term_alignment",
            )

    return TermUsageValidation(valid=True, source_term=src_term, target_form=tgt_form, warning=None)


def extract_target_form_from_translation(
    source_term: str,
    source_text: str,
    translated_text: str,
    suggested_target: str | None = None,
    known_obs_forms: list[str] | None = None,
) -> str | None:
    """Deterministically find the surface span in translated_text corresponding to source_term."""
    if not source_term or not translated_text:
        return None

    norm_trans = translated_text.strip()
    lower_trans = norm_trans.lower()

    # 1. Check suggested_target if present
    if suggested_target and suggested_target.strip():
        s_target = suggested_target.strip()
        idx = lower_trans.find(s_target.lower())
        if idx >= 0:
            return norm_trans[idx : idx + len(s_target)]

    # 2. Check known observation forms
    if known_obs_forms:
        for form in known_obs_forms:
            if form and form.strip():
                f_str = form.strip()
                idx = lower_trans.find(f_str.lower())
                if idx >= 0:
                    return norm_trans[idx : idx + len(f_str)]

    # 3. Direct token match with Turkish suffixes (e.g., DANTIAN -> Dantian'ını, MANA -> Mana'sını)
    words = [w.strip() for w in re.findall(r"[A-Za-z0-9]+", source_term) if len(w) >= 3]
    for w in words:
        pattern = r"\b" + re.escape(w) + r"(?:'[a-zA-ZıiğüşöÇĞİÜŞÖ]+|[a-zA-ZıiğüşöÇĞİÜŞÖ]*)\b"
        match = re.search(pattern, norm_trans, re.IGNORECASE)
        if match:
            return match.group(0)

    # Return None if no safe alignment found (do not fabricate evidence)
    return None



def record_validated_term_observations(
    candidate_store: CandidateStore,
    chapter_id: str,
    region_id: int,
    source_text: str,
    translated_text: str,
    raw_term_usages: list[dict[str, Any]],
    term_id_map: dict[str, str] | None = None,
    fidelity_flags: list[str] | None = None,
    requires_review: bool = False,
) -> list[TermObservation]:
    """Validate term_usages against translation output and record observations into CandidateStore.

    CRITICAL SAFETY RULES:
    1. Only called AFTER translation generation is fully accepted.
    2. SKIPS observation recording if requires_review=True or fidelity_flags is non-empty.
    3. Resolves term_id using item-scoped term_id_map (e.g. "T1" -> "MANA CORE").
    4. Enforces validate_term_alignment (exact span grounding, partial multiword guard).
    5. Invalid term_usages generate warnings and are skipped WITHOUT failing translation.
    6. Deduplication key: (chapter_id, region_id, canonical_source_term).
    """
    if not candidate_store or not translated_text or not translated_text.strip():
        return []

    # Fidelity Gate: If item has critical issues or fidelity flags, SKIP terminology learning!
    if requires_review:
        logger.debug(f"Skipping terminology observation for region {region_id}: requires_review is True")
        return []

    if fidelity_flags:
        logger.warning(f"Skipping terminology observation for region {region_id}: fidelity_flags={fidelity_flags}")
        return []

    provided_raw_usages = bool(raw_term_usages)
    term_map = term_id_map or {}
    usable_usages: list[dict[str, Any]] = []

    if raw_term_usages:
        for usage in raw_term_usages:
            if not isinstance(usage, dict):
                continue
            term_id = str(usage.get("term_id", "")).strip()
            src_term = str(usage.get("source_term", "")).strip()
            tgt_form = str(usage.get("target_form", "")).strip()

            # Resolve term_id to source_term using item-scoped map
            if term_id:
                resolved_src = term_map.get(term_id)
                if not resolved_src:
                    logger.warning(f"Region {region_id}: term_id '{term_id}' not found in item term_id_map {term_map}. Skipping.")
                    continue
                src_term = resolved_src

            if src_term and tgt_form:
                usable_usages.append({"source_term": src_term, "target_form": tgt_form})

    # If NO raw_term_usages were provided at all by LLM, attempt deterministic fallback extraction
    if not provided_raw_usages and candidate_store.candidates:
        for k, cand in candidate_store.candidates.items():
            if cand.status in ("discovered", "provisional", "ready_for_review"):
                if contains_candidate_phrase(cand.source, source_text):
                    tgt_span = extract_target_form_from_translation(
                        source_term=cand.source,
                        source_text=source_text,
                        translated_text=translated_text,
                        suggested_target=cand.suggested_target,
                        known_obs_forms=list(cand.observed_target_counts.keys()),
                    )
                    if tgt_span:
                        usable_usages.append({
                            "source_term": cand.source,
                            "target_form": tgt_span,
                        })

    recorded: list[TermObservation] = []

    for usage in usable_usages:
        src_term = str(usage.get("source_term", "")).strip()
        tgt_form = str(usage.get("target_form", "")).strip()

        val = validate_term_alignment(
            source_term=src_term,
            source_text=source_text,
            target_form=tgt_form,
            translated_text=translated_text,
        )

        if not val.valid:
            logger.warning(f"Region {region_id}: term alignment invalid for '{src_term}' -> '{tgt_form}': {val.warning}. Skipping observation.")
            continue

        key = src_term.upper()

        candidate = candidate_store.candidates.get(key)
        if not candidate:
            candidate = ProfileCandidate(source=src_term, kind="term", status="discovered")
            candidate_store.candidates[key] = candidate

        obs = TermObservation(
            chapter_id=chapter_id,
            region_id=region_id,
            source_text=source_text,
            translated_text=translated_text,
            observed_target_form=tgt_form,
        )

        if candidate.add_observation(obs):
            recorded.append(obs)
            logger.info(f"Recorded term observation: {key} -> '{tgt_form}' (status={candidate.status}, obs_count={len(candidate.observations)})")

    return recorded
