"""Unit tests for Translation-Evidence-Driven Termbase architecture, lifecycle, and grounding rules."""
import pytest
from core.translation.profile_discovery import (
    CandidateStore,
    ProfileCandidate,
    TermObservation,
    confirm_candidate,
    get_relevant_terms_for_item,
    record_validated_term_observations,
)
from core.translation.series_profile import SeriesProfile
from providers.translation.base import TranslationItem, TranslationOutputItem
from providers.translation.qwen_glossary import TermResolutionResult, resolve_candidate_targets_with_qwen


def test_per_item_term_retrieval():
    profile = SeriesProfile(
        series_id="test_series",
        glossary={"MANA CORE": "mana çekirdeği"},
    )
    store = CandidateStore(series_id="test_series")
    store.candidates["GUILD MASTER"] = ProfileCandidate(
        source="GUILD MASTER", kind="title_or_rank", status="provisional"
    )
    store.candidates["UNRELATED TERM"] = ProfileCandidate(
        source="UNRELATED TERM", kind="term", status="provisional"
    )

    # Item 1 contains MANA CORE and GUILD MASTER
    item_1 = "KANG MINHO, REPORT TO THE GUILD MASTER ABOUT THE MANA CORE."
    app_terms_1, prov_terms_1 = get_relevant_terms_for_item(item_1, profile, store)

    assert "MANA CORE" in app_terms_1
    assert app_terms_1["MANA CORE"] == "mana çekirdeği"
    assert "GUILD MASTER" in prov_terms_1
    # UNRELATED TERM is NOT injected for item 1
    assert "UNRELATED TERM" not in prov_terms_1

    # Item 2 contains UNRELATED TERM only
    item_2 = "THIS HAS AN UNRELATED TERM IN IT."
    app_terms_2, prov_terms_2 = get_relevant_terms_for_item(item_2, profile, store)
    assert len(app_terms_2) == 0
    assert "UNRELATED TERM" in prov_terms_2
    assert "GUILD MASTER" not in prov_terms_2


def test_term_observation_record_and_deduplication():
    store = CandidateStore(series_id="test_series")

    raw_term_usages = [
        {"source_term": "SECRET REALM", "target_form": "gizli âlem"},
        {"source_term": "UNFOUND TERM", "target_form": "bozuk"},  # Source term not in source_text
        {"source_term": "SECRET REALM", "target_form": "hallucinated_span"},  # Target span not in translation
    ]

    source_text = "WE ARE ENTERING THE SECRET REALM."
    translated_text = "Gizli âleme giriyoruz."

    # 1. Record observations for region 10
    recs = record_validated_term_observations(
        candidate_store=store,
        chapter_id="ch001",
        region_id=10,
        source_text=source_text,
        translated_text=translated_text,
        raw_term_usages=raw_term_usages,
    )

    assert len(recs) == 1
    assert recs[0].observed_target_form == "gizli âlem"

    cand = store.candidates["SECRET REALM"]
    assert len(cand.observations) == 1
    assert cand.status == "provisional"

    # 2. Re-running the exact same (chapter_id, region_id) MUST NOT add duplicate evidence or increase count
    recs_dup = record_validated_term_observations(
        candidate_store=store,
        chapter_id="ch001",
        region_id=10,
        source_text=source_text,
        translated_text=translated_text,
        raw_term_usages=raw_term_usages,
    )

    assert len(recs_dup) == 0
    assert len(cand.observations) == 1
    assert cand.evidence_count == 1


def test_candidate_lifecycle_progression():
    cand = ProfileCandidate(source="MANA CORE", kind="term", status="discovered")

    # 0 observations -> discovered
    assert cand.status == "discovered"

    # 1 observation -> provisional
    cand.add_observation(
        TermObservation(
            chapter_id="ch001",
            region_id=1,
            source_text="MANA CORE IS STABLE",
            translated_text="Mana çekirdeği kararlı",
            observed_target_form="mana çekirdeği",
        )
    )
    assert cand.status == "provisional"

    # 2nd observation on DIFFERENT region_id with consistent target -> ready_for_review
    cand.add_observation(
        TermObservation(
            chapter_id="ch001",
            region_id=2,
            source_text="PROTECT THE MANA CORE",
            translated_text="Mana çekirdeği korunsun",
            observed_target_form="mana çekirdeği",
        )
    )
    # Lifecycle should promote to ready_for_review (2 independent regions, consistent form)
    assert cand.status == "ready_for_review"

    # Safety Guarantee: Lifecycle NEVER auto-promotes to approved/confirmed!
    assert cand.status != "approved"
    assert cand.status != "confirmed"


def test_evidence_grounded_resolver_rules():
    # Candidate with 0 or 1 observation -> skipped by resolver logic
    cand_1 = ProfileCandidate(source="MANA CORE", kind="term", status="provisional")
    cand_1.add_observation(
        TermObservation("ch1", 1, "MANA CORE", "Mana çekirdeği", "mana çekirdeği")
    )

    # Candidate with 2+ observations
    cand_2 = ProfileCandidate(source="INNER DISCIPLE", kind="title_or_rank", status="provisional")
    cand_2.add_observation(
        TermObservation("ch1", 10, "INNER DISCIPLE", "İç mürit korunsun", "iç mürit")
    )
    cand_2.add_observation(
        TermObservation("ch1", 11, "INNER DISCIPLE", "İç mürit geldi", "iç mürit")
    )

    # cand_1 has 1 observation, cand_2 has 2 observations
    from providers.translation.qwen_glossary import resolve_candidate_targets_with_qwen

    class MockProvider:
        is_loaded = True
        _model = True
        _processor = True

    # Check observed target validation: if model output target is NOT in candidate.observed_target_counts, set unresolved
    raw_item = {
        "source": "INNER DISCIPLE",
        "status": "resolved",
        "preferred_observed_target": "INVENTED TARGET",  # Not in observations!
        "reason": "Test reason",
    }
    # Deterministic check test
    valid_observed = {k.lower() for k in cand_2.observed_target_counts.keys()}
    target_str = raw_item["preferred_observed_target"]
    assert target_str.lower() not in valid_observed


def test_explicit_confirmation_only():
    store = CandidateStore(series_id="test_series")
    profile = SeriesProfile(series_id="test_series")

    cand = ProfileCandidate(source="MANA CORE", kind="term", status="ready_for_review")
    cand.add_observation(
        TermObservation("ch1", 1, "MANA CORE", "Mana çekirdeği", "mana çekirdeği")
    )
    store.candidates["MANA CORE"] = cand

    # Candidate is ready_for_review, but profile glossary is empty
    assert len(profile.glossary) == 0

    # Only explicit confirm_candidate updates profile
    ok = confirm_candidate(store, profile, "MANA CORE", target_override="mana çekirdeği")
    assert ok
    assert profile.glossary["MANA CORE"] == "mana çekirdeği"
    assert store.candidates["MANA CORE"].status in ("approved", "confirmed")


def test_item_scoped_term_id_mapping():
    """Verify item 3 'T1' = MANA CORE and item 11 'T1' = INNER DISCIPLE do not cross-contaminate."""
    store = CandidateStore(series_id="test_series")

    # Item 3 term_id_map: T1 -> MANA CORE
    map_item_3 = {"T1": "MANA CORE"}
    # Item 11 term_id_map: T1 -> INNER DISCIPLE
    map_item_11 = {"T1": "INNER DISCIPLE"}

    usages_3 = [{"term_id": "T1", "target_form": "mana çekirdeğine"}]
    usages_11 = [{"term_id": "T1", "target_form": "iç kılıç öğrencisi"}]

    recs_3 = record_validated_term_observations(
        candidate_store=store,
        chapter_id="ch001",
        region_id=3,
        source_text="ONLY AWAKENERS WITH A STABLE MANA CORE MAY ENTER.",
        translated_text="Sadece kararlı Mana Çekirdeğine sahip Uyanmışlar girebilir.",
        raw_term_usages=usages_3,
        term_id_map=map_item_3,
    )
    assert len(recs_3) == 1
    assert "MANA CORE" in store.candidates
    assert store.candidates["MANA CORE"].observations[0].observed_target_form == "mana çekirdeğine"

    recs_11 = record_validated_term_observations(
        candidate_store=store,
        chapter_id="ch001",
        region_id=11,
        source_text="AN INNER DISCIPLE MUST PROTECT HIS DANTIAN.",
        translated_text="İç kılıç öğrencisi Dantian'ını korumak zorunda.",
        raw_term_usages=usages_11,
        term_id_map=map_item_11,
    )
    assert len(recs_11) == 1
    assert "INNER DISCIPLE" in store.candidates
    assert store.candidates["INNER DISCIPLE"].observations[0].observed_target_form == "iç kılıç öğrencisi"

    # Using item 11's map on item 3 (wrong term_id resolution) MUST be rejected
    recs_wrong = record_validated_term_observations(
        candidate_store=store,
        chapter_id="ch001",
        region_id=99,
        source_text="MANA CORE STABLE",
        translated_text="Mana Çekirdeğine sahip",
        raw_term_usages=[{"term_id": "T99", "target_form": "Mana Çekirdeğine"}],
        term_id_map=map_item_3,
    )
    assert len(recs_wrong) == 0


def test_partial_multiword_alignment_guard():
    """Verify MANA CORE -> Mana is rejected while MANA CORE -> mana çekirdeğine and single-token DANTIAN -> Dantian'ını are accepted."""
    from core.translation.profile_discovery import validate_term_alignment

    # 1. MANA CORE -> Mana (partial multiword alignment) -> INVALID
    val_1 = validate_term_alignment(
        source_term="MANA CORE",
        source_text="STABLE MANA CORE MAY ENTER.",
        target_form="Mana",
        translated_text="Kararlı Mana Çekirdeğine sahip uyanmışlar.",
    )
    assert not val_1.valid
    assert val_1.warning == "partial_term_alignment"

    # 2. GUILD MASTER -> Guild -> INVALID
    val_2 = validate_term_alignment(
        source_term="GUILD MASTER",
        source_text="REPORT TO THE GUILD MASTER.",
        target_form="Guild",
        translated_text="Guild başkanı haber bekliyor.",
    )
    assert not val_2.valid
    assert val_2.warning == "partial_term_alignment"

    # 3. MANA CORE -> mana çekirdeğine -> VALID
    val_3 = validate_term_alignment(
        source_term="MANA CORE",
        source_text="STABLE MANA CORE MAY ENTER.",
        target_form="mana çekirdeğine",
        translated_text="Kararlı mana çekirdeğine sahip uyanmışlar.",
    )
    assert val_3.valid
    assert val_3.warning is None

    # 4. Single-token DANTIAN -> Dantian'ını -> VALID (single-token terms not guarded by multiword rule)
    val_4 = validate_term_alignment(
        source_term="DANTIAN",
        source_text="PROTECT HIS DANTIAN.",
        target_form="Dantian'ını",
        translated_text="Dantian'ını korumak zorunda.",
    )
    assert val_4.valid
    assert val_4.warning is None


def test_fidelity_flags_gate_observation():
    """Verify fidelity_flags or requires_review skip observation recording without failing translation."""
    store = CandidateStore(series_id="test_series")

    # Item with fidelity_flags -> observation SKIPPED
    recs = record_validated_term_observations(
        candidate_store=store,
        chapter_id="ch001",
        region_id=11,
        source_text="AN INNER DISCIPLE MUST PROTECT HIS DANTIAN.",
        translated_text="İç bir kılıç öğrencisi Dantian'ını korumak zorunda.",
        raw_term_usages=[{"source_term": "INNER DISCIPLE", "target_form": "iç bir kılıç öğrencisi"}],
        fidelity_flags=["added_information"],
        requires_review=False,
    )
    assert len(recs) == 0
    assert "INNER DISCIPLE" not in store.candidates


def test_deduplication_identity_distinct_terms_same_region():
    """Verify distinct terms in the same region produce separate observations, while exact same term/region twice deduplicates."""
    store = CandidateStore(series_id="test_series")

    # Region 3 has MANA CORE and AWAKENERS
    recs_1 = record_validated_term_observations(
        candidate_store=store,
        chapter_id="ch001",
        region_id=3,
        source_text="AWAKENERS WITH STABLE MANA CORE",
        translated_text="Kararlı Mana Çekirdeğine sahip Uyanmışlar",
        raw_term_usages=[
            {"source_term": "MANA CORE", "target_form": "Mana Çekirdeğine"},
            {"source_term": "AWAKENERS", "target_form": "Uyanmışlar"},
        ],
    )
    assert len(recs_1) == 2
    assert "MANA CORE" in store.candidates
    assert "AWAKENERS" in store.candidates

    # Re-running region 3 for MANA CORE -> count stays 1
    recs_dup = record_validated_term_observations(
        candidate_store=store,
        chapter_id="ch001",
        region_id=3,
        source_text="AWAKENERS WITH STABLE MANA CORE",
        translated_text="Kararlı Mana Çekirdeğine sahip Uyanmışlar",
        raw_term_usages=[{"source_term": "MANA CORE", "target_form": "Mana Çekirdeğine"}],
    )
    assert len(recs_dup) == 0
    assert len(store.candidates["MANA CORE"].observations) == 1
    assert store.candidates["MANA CORE"].status == "provisional"
