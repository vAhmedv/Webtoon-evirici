from __future__ import annotations

from unittest.mock import patch

from core.translation.protection import (
    ProtectedTermMeta,
    _suffix_category,
    detect_named_terms_in_items,
    restore_protected_translation,
)
from core.translation.series_profile import SeriesProfile
from core.translation.source_normalization import normalize_translation_source_case
from providers.translation.base import TranslationInput, TranslationItem
from providers.translation.qwen_gguf_translation_v2 import QwenGGUFTranslationProviderV2


def _axe_profile() -> SeriesProfile:
    return SeriesProfile(
        series_id="axe_god",
        known_names={"GAO YUAN": "Gao Yuan", "LUO TIAN": "Luo Tian", "YU": "Yu"},
        glossary={
            "ABILITY USER": "yetenek kullanıcısı",
            "SECRET REALM GUIDE": "gizli âlem rehberi",
        },
    )


def test_all_caps_prose_normalization_preserves_names_terms_and_structure() -> None:
    profile = _axe_profile()
    cases = {
        "I'M USED TO IT.": "I'm used to it.",
        "LOOKS LIKE MY MONEY WASN'T WASTED. YOU'RE WORTH EVERY PENNY, KID!":
            "Looks like my money wasn't wasted. You're worth every penny, kid!",
        "WITHIN THESE SECRET REALMS, DANGER LURKS EVERYWHERE.":
            "Within these secret realms, danger lurks everywhere.",
        "CAPTAIN GAO YUAN IS AN ABILITY USER WITH ZFO, HP, MP AND LV.13.":
            "Captain Gao Yuan is an ABILITY USER with ZFO, HP, MP and LV.13.",
    }
    for source, expected in cases.items():
        normalized = normalize_translation_source_case(source, profile=profile)
        assert normalized == expected
        assert normalize_translation_source_case(normalized, profile=profile) == normalized

    structured = "[CRAFT] 〈PIERCE〉 《SLOTTED》"
    assert normalize_translation_source_case(structured, profile=profile) == structured
    assert normalize_translation_source_case("SKILL: PHANTOM THREAD", profile=profile) == "SKILL: PHANTOM THREAD"


def test_named_term_detection_rejects_used_to_prose_but_keeps_explicit_cues() -> None:
    ordinary = [
        TranslationItem(region_id=1, source="I'M USED TO IT."),
        TranslationItem(region_id=2, source="This tool is used to open it."),
    ]
    assert detect_named_terms_in_items(ordinary) == set()

    explicit = [
        TranslationItem(region_id=3, source="Activate Phantom Thread."),
        TranslationItem(region_id=4, source="Skill: PHANTOM THREAD"),
        TranslationItem(region_id=5, source="It is called Frost Chain."),
    ]
    detected = detect_named_terms_in_items(explicit)
    assert {term.casefold() for term in detected} == {"phantom thread", "frost chain"}


def test_sentinel_nominal_morphology_and_boundaries() -> None:
    common = ProtectedTermMeta(
        sentinel="__WTTERM0001__",
        source_original="ABILITY USER",
        target_base="yetenek kullanıcısı",
        is_approved=True,
        proper_name=False,
    )
    proper = ProtectedTermMeta(
        sentinel="__WTTERM0002__",
        source_original="GAO YUAN",
        target_base="Gao Yuan",
        is_approved=True,
        proper_name=True,
    )
    mapping = {common.sentinel: common, proper.sentinel: proper}

    assert _suffix_category("", "DIR") == "copular"
    assert _suffix_category("", "İM") == "person_1sg"
    assert restore_protected_translation("__WTTERM0001__DIR bugün", mapping) == "yetenek kullanıcısıdır bugün"
    assert restore_protected_translation("__WTTERM0001__'DIR bugün", mapping) == "yetenek kullanıcısıdır bugün"
    assert restore_protected_translation("__WTTERM0001__İM artık", mapping) == "yetenek kullanıcısıyım artık"
    assert restore_protected_translation("__WTTERM0001__'İM artık", mapping) == "yetenek kullanıcısıyım artık"
    assert restore_protected_translation("__WTTERM0002__DIR", mapping) == "Gao Yuan'dır"


def test_qwen_v2_normalizes_for_model_but_preserves_original_source() -> None:
    provider = QwenGGUFTranslationProviderV2()
    provider._loaded = True
    inp = TranslationInput(
        items=[TranslationItem(region_id=1, source="I'M USED TO IT.")],
        profile=_axe_profile(),
    )
    with patch.object(provider, "_check_health", return_value=True), patch.object(
        provider,
        "_request_translation",
        return_value=("Buna alışkınım.", "Buna alışkınım.", False),
    ) as request:
        output = provider.translate(inp)

    assert request.call_args.args[0] == "I'm used to it."
    assert "__WTTERM" not in request.call_args.args[0]
    assert output.results[0].source == "I'M USED TO IT."
    assert output.results[0].translation == "Buna alışkınım."


def test_qwen_v2_long_all_caps_protects_approved_term_and_restores_copula() -> None:
    provider = QwenGGUFTranslationProviderV2()
    provider._loaded = True
    source = (
        "CAPTAIN GAO YUAN IS A PEAK LEVEL 1 ABILITY USER, "
        "AND THE REST OF THE TEAM ARE NO PUSHOVERS EITHER."
    )
    inp = TranslationInput(items=[TranslationItem(region_id=19, source=source)], profile=_axe_profile())

    def fake_request(prepared_text: str, label: str) -> tuple[str, str, bool]:
        assert prepared_text.startswith("__WTTERM")
        sentinels = [
            token.rstrip(",")
            for token in prepared_text.split()
            if token.startswith("__WTTERM")
        ]
        assert len(sentinels) == 2
        name_sentinel, ability_sentinel = sentinels
        raw = (
            f"{name_sentinel} zirvede bir {ability_sentinel}DIR, "
            "takımın kalanı da hafife alınmaz."
        )
        return raw, raw, False

    with patch.object(provider, "_check_health", return_value=True), patch.object(
        provider, "_request_translation", side_effect=fake_request
    ):
        result = provider.translate(inp).results[0]

    assert result.source == source
    assert "yetenek kullanıcısıdır" in (result.translation or "")
    assert "__WTTERM" not in (result.translation or "")
    assert result.requires_review is False
