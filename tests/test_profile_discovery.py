"""Unit tests for profile discovery domain logic, evidence validation, confirmation API, and persistence."""
import pytest
from core.translation.profile_discovery import (
    CandidateEvidence,
    CandidateStore,
    ProfileCandidate,
    confirm_candidate,
    contains_candidate_phrase,
    process_discovered_suggestions,
    reject_candidate,
    validate_candidate_suggestion,
)
from core.translation.series_profile import SeriesProfile
from providers.translation.base import TranslationItem


def test_contains_candidate_phrase_boundary_matching():
    # YU in YOU SAW IT YOURSELF must NOT match YU
    assert not contains_candidate_phrase("YU", "YOU SAW IT YOURSELF JUST NOW.")
    # YU in YOUNG MASTER YU must match
    assert contains_candidate_phrase("YU", "YOUNG MASTER YU, CAPTAIN GAO")

    # Hyphenated names/ranks
    assert contains_candidate_phrase("JIN-WOO", "THE SECT LEADER HAS SUMMONED JIN-WOO.")
    assert contains_candidate_phrase("S-RANK", "HE IS AN S-RANK HUNTER.")
    assert contains_candidate_phrase("LEVEL-1", "A LEVEL-1 ABILITY USER")

    # Exact name distinction
    assert contains_candidate_phrase("LUO TIAN", "MY NAME IS LUO TIAN.")
    assert not contains_candidate_phrase("LUO YAN", "MY NAME IS LUO TIAN.")

    # Canonical approved terms own the full safe plural/possessive span.
    assert contains_candidate_phrase("SPIRIT STONE", "Spirit Stone")
    assert contains_candidate_phrase("SPIRIT STONE", "Spirit Stones")
    assert contains_candidate_phrase("SPIRIT STONE", "Spirit Stone's value")

    # Short glossary keys must not bleed into unrelated words.
    assert not contains_candidate_phrase("YU", "YOU")
    assert not contains_candidate_phrase("YU", "YOUR")
    assert not contains_candidate_phrase("YU", "YOURSELF")


def test_validate_candidate_suggestion():
    items = [TranslationItem(region_id=1, source="KANG MINHO IS A GUILD MASTER.")]

    # Valid candidate present in source text
    valid, reason = validate_candidate_suggestion("KANG MINHO", "character_name", items)
    assert valid
    assert reason is None

    # Hallucinated candidate not present in source text
    valid, reason = validate_candidate_suggestion("HALLUCINATED NAME", "character_name", items)
    assert not valid
    assert reason == "not_found_in_source_text"

    # CJK hallucination
    valid, reason = validate_candidate_suggestion("你好", "term", items)
    assert not valid
    assert reason == "cjk_hallucination"

    # Invalid kind
    valid, reason = validate_candidate_suggestion("KANG MINHO", "invalid_kind", items)
    assert not valid
    assert reason == "invalid_kind_invalid_kind"


def test_process_discovered_suggestions_merge_and_safety():
    store = CandidateStore(series_id="test_series")
    items = [
        TranslationItem(region_id=10, source="MY NAME IS LUO TIAN."),
        TranslationItem(region_id=11, source="LUO TIAN IS A SECRET REALM GUIDE."),
    ]
    raw_suggestions = [
        {
            "source": "LUO TIAN",
            "kind": "character_name",
            "suggested_target": "Luo Tian",
            "evidence_ids": [10],
        },
        {
            "source": "SECRET REALM",
            "kind": "term",
            "suggested_target": "gizli âlem",
            "evidence_ids": [11],
        },
        {
            "source": "HALLUCINATED PERSON",
            "kind": "character_name",
            "suggested_target": "Fake",
            "evidence_ids": [10],
        },
    ]

    res = process_discovered_suggestions(
        raw_suggestions=raw_suggestions,
        items=items,
        chapter_id="ch1",
        candidate_store=store,
    )

    # Hallucinated item filtered out
    assert res.filtered_count == 1
    assert len(res.candidates) == 2

    # Safety principle: Newly discovered candidates start as 'discovered'
    for cand in res.candidates:
        assert cand.status == "discovered"


    luo_cand = store.candidates["LUO TIAN"]
    assert luo_cand.kind == "character_name"
    assert luo_cand.suggested_target == "Luo Tian"
    assert luo_cand.evidence_count >= 1

    secret_cand = store.candidates["SECRET REALM"]
    assert secret_cand.kind == "term"
    # Discovery narrow scope: terms have suggested_target = None
    assert secret_cand.suggested_target is None


def test_confirm_and_reject_candidate():
    store = CandidateStore(series_id="test_series")
    profile = SeriesProfile(series_id="test_series")

    store.candidates["KANG MINHO"] = ProfileCandidate(
        source="KANG MINHO",
        kind="character_name",
        suggested_target="Kang Minho",
        status="provisional",
    )
    store.candidates["MANA CORE"] = ProfileCandidate(
        source="MANA CORE",
        kind="term",
        suggested_target="mana çekirdeği",
        status="provisional",
    )

    # Confirm character name -> goes to profile.known_names
    ok = confirm_candidate(store, profile, "KANG MINHO")
    assert ok
    assert store.candidates["KANG MINHO"].status in ("approved", "confirmed")
    assert profile.known_names["KANG MINHO"] == "Kang Minho"

    # Confirm term -> goes to profile.glossary
    ok = confirm_candidate(store, profile, "MANA CORE", target_override="Mana Çekirdeği")
    assert ok
    assert store.candidates["MANA CORE"].status in ("approved", "confirmed")
    assert profile.glossary["MANA CORE"] == "Mana Çekirdeği"

    # Reject candidate
    store.candidates["WEAK ITEM"] = ProfileCandidate(source="WEAK ITEM", kind="term", status="provisional")
    ok = reject_candidate(store, "WEAK ITEM")
    assert ok
    assert store.candidates["WEAK ITEM"].status == "rejected"
    assert "WEAK ITEM" not in profile.glossary


def test_candidate_store_persistence(tmp_path):
    store_file = tmp_path / "custom_series.candidates.json"
    store = CandidateStore(series_id="custom_series")
    store.candidates["RED GATE"] = ProfileCandidate(
        source="RED GATE",
        kind="place_name",
        suggested_target="Kırmızı Geçit",
        status="provisional",
        evidence_count=1,
    )

    saved_path = store.save_to_json(store_file)
    assert saved_path.exists()

    loaded = CandidateStore.load_from_json(saved_path)
    assert loaded.series_id == "custom_series"
    assert "RED GATE" in loaded.candidates
    assert loaded.candidates["RED GATE"].suggested_target == "Kırmızı Geçit"


def test_synthetic_fixture_a_dungeon_discovery():
    items = [
        TranslationItem(region_id=1, source="KANG MINHO, REPORT TO THE GUILD MASTER."),
        TranslationItem(region_id=2, source="THE RED GATE HAS OPENED AGAIN."),
        TranslationItem(region_id=3, source="ONLY AWAKENERS WITH A STABLE MANA CORE MAY ENTER."),
    ]
    store = CandidateStore(series_id="dungeon_series")

    raw_suggestions = [
        {"source": "KANG MINHO", "kind": "character_name", "suggested_target": "Kang Minho", "evidence_ids": [1]},
        {"source": "GUILD MASTER", "kind": "title_or_rank", "suggested_target": "Lonca Lideri", "evidence_ids": [1]},
        {"source": "RED GATE", "kind": "place_name", "suggested_target": "Kırmızı Geçit", "evidence_ids": [2]},
        {"source": "AWAKENER", "kind": "term", "suggested_target": "uyanmış kişi", "evidence_ids": [3]},
        {"source": "MANA CORE", "kind": "term", "suggested_target": "mana çekirdeği", "evidence_ids": [3]},
    ]

    res = process_discovered_suggestions(
        raw_suggestions=raw_suggestions,
        items=items,
        chapter_id="ch1",
        candidate_store=store,
    )

    assert len(res.candidates) == 5
    assert set(store.candidates.keys()) == {"KANG MINHO", "GUILD MASTER", "RED GATE", "AWAKENER", "MANA CORE"}
    for cand in res.candidates:
        assert cand.status == "discovered"



def test_synthetic_fixture_b_murim_isolation():
    store_a = CandidateStore(series_id="dungeon_series")
    store_b = CandidateStore(series_id="murim_series")

    items_b = [
        TranslationItem(region_id=10, source="THE SECT LEADER HAS SUMMONED JIN-WOO."),
        TranslationItem(region_id=11, source="AN INNER DISCIPLE MUST PROTECT HIS DANTIAN."),
        TranslationItem(region_id=12, source="THE HEAVENLY DEMON CULT IS MOVING AGAIN."),
    ]
    raw_suggestions_b = [
        {"source": "JIN-WOO", "kind": "character_name", "suggested_target": "Jin-Woo", "evidence_ids": [10]},
        {"source": "SECT LEADER", "kind": "title_or_rank", "suggested_target": "Mezhep Lideri", "evidence_ids": [10]},
        {"source": "INNER DISCIPLE", "kind": "title_or_rank", "suggested_target": "İç Öğrenci", "evidence_ids": [11]},
        {"source": "DANTIAN", "kind": "term", "suggested_target": "dantian", "evidence_ids": [11]},
        {"source": "HEAVENLY DEMON CULT", "kind": "term", "suggested_target": "İlahi İblis Tarikatı", "evidence_ids": [12]},
    ]

    res_b = process_discovered_suggestions(
        raw_suggestions=raw_suggestions_b,
        items=items_b,
        chapter_id="ch1",
        candidate_store=store_b,
    )

    # Series B stores only Murim candidates
    assert len(res_b.candidates) == 5
    assert "JIN-WOO" in store_b.candidates
    assert "JIN-WOO" not in store_a.candidates
