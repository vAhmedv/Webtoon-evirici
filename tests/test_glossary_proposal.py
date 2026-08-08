"""Unit tests for GlossaryProposal multi-option schema, deterministic validator, and safety constraints."""
import pytest
from core.translation.glossary_proposal import (
    GlossaryProposal,
    apply_glossary_proposals,
    select_candidates_for_proposal,
    validate_glossary_proposal,
)
from core.translation.profile_discovery import CandidateStore, ProfileCandidate
from core.translation.series_profile import SeriesProfile


def test_proposal_multi_option_schema_and_dict_conversion():
    prop = GlossaryProposal(
        source="INNER DISCIPLE",
        kind="title_or_rank",
        options=["İç Mürit", "İç Öğrenci", "İç Tarikat Öğrencisi"],
        preferred_target="İç Mürit",
        reason="Murim klan yapısında iç mekanlarda eğitilen öğrenci unvanıdır.",
    )
    d = prop.to_dict()
    assert d["source"] == "INNER DISCIPLE"
    assert d["kind"] == "title_or_rank"
    assert d["options"] == ["İç Mürit", "İç Öğrenci", "İç Tarikat Öğrencisi"]
    assert d["preferred_target"] == "İç Mürit"
    assert d["suggested_target"] == "İç Mürit"

    restored = GlossaryProposal.from_dict(d)
    assert restored.source == prop.source
    assert restored.kind == prop.kind
    assert restored.options == prop.options
    assert restored.preferred_target == prop.preferred_target
    assert restored.suggested_target == prop.suggested_target


def test_validate_glossary_proposal_valid_case():
    prop = GlossaryProposal(
        source="SECRET REALM",
        kind="term",
        options=["gizli âlem", "gizli dünya"],
        preferred_target="gizli âlem",
        reason="Gizemli kurgusal boyutu ifade eder.",
    )
    is_valid, warnings = validate_glossary_proposal(prop)
    assert is_valid
    assert prop.is_valid
    assert not prop.requires_review
    assert len(warnings) == 0


def test_validate_glossary_proposal_preferred_not_in_options():
    # Model specified preferred_target not in options list: must NOT silently self-correct!
    prop = GlossaryProposal(
        source="GUILD MASTER",
        kind="title_or_rank",
        options=["Lonca Lideri", "Lonca Başkanı"],
        preferred_target="Lonca Reisi",  # Not in options!
        reason="Lonca yöneticisini ifade eder.",
    )
    is_valid, warnings = validate_glossary_proposal(prop)
    assert not is_valid
    assert not prop.is_valid
    assert prop.requires_review
    assert "preferred_not_in_options" in warnings


def test_validate_glossary_proposal_source_language_leak_warning():
    # Source content token 'GUILD' left untranslated in preferred_target
    prop = GlossaryProposal(
        source="GUILD MASTER",
        kind="title_or_rank",
        options=["GUILD BAŞKANI", "Lonca Lideri"],
        preferred_target="GUILD BAŞKANI",
        reason="Lonca yöneticisini ifade eder.",
    )
    is_valid, warnings = validate_glossary_proposal(prop)
    # Leak is a soft warning, so structure remains valid but flags review
    assert is_valid
    assert prop.requires_review
    assert any("possible_source_language_leak" in w for w in warnings)


def test_validate_glossary_proposal_unsupported_external_claim_warning():
    prop = GlossaryProposal(
        source="MANA CORE",
        kind="term",
        options=["mana çekirdeği", "enerji odağı"],
        preferred_target="mana çekirdeği",
        reason="Türk webtoon çevirilerinde yaygın olarak çekirdek diye geçer.",
    )
    is_valid, warnings = validate_glossary_proposal(prop)
    assert is_valid
    assert prop.requires_review
    assert "unsupported_external_claim" in warnings


def test_select_candidates_for_proposal_filtering():
    store = CandidateStore(series_id="test_series")
    store.candidates["KANG MINHO"] = ProfileCandidate(
        source="KANG MINHO", kind="character_name", status="provisional"
    )
    store.candidates["GUILD MASTER"] = ProfileCandidate(
        source="GUILD MASTER", kind="title_or_rank", status="provisional"
    )
    store.candidates["REJECTED TERM"] = ProfileCandidate(
        source="REJECTED TERM", kind="term", status="rejected"
    )
    store.candidates["CONFIRMED TERM"] = ProfileCandidate(
        source="CONFIRMED TERM", kind="term", status="provisional"
    )

    profile = SeriesProfile(
        series_id="test_series",
        glossary={"CONFIRMED TERM": "onaylı terim"},
    )

    eligible = select_candidates_for_proposal(store, profile)
    eligible_sources = [c.source for c in eligible]

    assert "KANG MINHO" not in eligible_sources
    assert "REJECTED TERM" not in eligible_sources
    assert "CONFIRMED TERM" not in eligible_sources
    assert "GUILD MASTER" in eligible_sources


def test_apply_glossary_proposals_immutability_and_provisional_status():
    store = CandidateStore(series_id="test_series")
    profile = SeriesProfile(series_id="test_series")

    cand = ProfileCandidate(
        source="MANA CORE",
        kind="term",
        suggested_target=None,
        status="provisional",
    )
    store.candidates["MANA CORE"] = cand

    proposals = [
        GlossaryProposal(
            source="MANA CORE",
            kind="term",
            options=["mana çekirdeği", "enerji odağı"],
            preferred_target="mana çekirdeği",
            reason="Kurgusal enerji depolama organı.",
        )
    ]

    updated = apply_glossary_proposals(store, proposals, profile)

    assert len(updated) == 1
    assert store.candidates["MANA CORE"].suggested_target == "mana çekirdeği"
    # Safety Principle: Candidate status MUST remain provisional
    assert store.candidates["MANA CORE"].status == "provisional"
    # Safety Principle: SeriesProfile MUST remain untouched
    assert "MANA CORE" not in profile.glossary
    assert len(profile.glossary) == 0
