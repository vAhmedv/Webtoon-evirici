"""Translation core data models, discovery, glossary proposal, and utilities."""
from core.translation.series_profile import SeriesProfile
from core.translation.profile_discovery import (
    CandidateEvidence,
    CandidateStore,
    DiscoveryResult,
    ProfileCandidate,
    confirm_candidate,
    contains_candidate_phrase,
    process_discovered_suggestions,
    reject_candidate,
    validate_candidate_suggestion,
)
from core.translation.glossary_proposal import (
    GlossaryProposal,
    apply_glossary_proposals,
    select_candidates_for_proposal,
)

__all__ = [
    "SeriesProfile",
    "CandidateEvidence",
    "CandidateStore",
    "DiscoveryResult",
    "ProfileCandidate",
    "confirm_candidate",
    "contains_candidate_phrase",
    "process_discovered_suggestions",
    "reject_candidate",
    "validate_candidate_suggestion",
    "GlossaryProposal",
    "apply_glossary_proposals",
    "select_candidates_for_proposal",
]
