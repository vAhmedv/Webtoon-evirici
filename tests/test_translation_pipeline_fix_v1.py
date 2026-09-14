from __future__ import annotations

from unittest.mock import patch

from core.translation.protection import (
    ProtectedTermMeta,
    _suffix_category,
    collapse_stem_doubles,
    decapitalize_common_lock_targets,
    detect_named_terms_in_items,
    find_name_glue,
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


def test_sentinel_unknown_suffix_falls_back_to_bare_base() -> None:
    # S5 çekim kapısı: "Dünya"+"ine" uydurması yerine yalın taban.
    world = ProtectedTermMeta(
        sentinel="__WTTERM0009__",
        source_original="WORLD",
        target_base="Dünya",
        is_approved=True,
        proper_name=False,
    )
    mapping = {world.sentinel: world}
    assert restore_protected_translation("__WTTERM0009__ine", mapping) == "Dünya"
    assert restore_protected_translation("__WTTERM0009__xyz", mapping) == "Dünya"
    # Bilinen ekler etkilenmez.
    assert restore_protected_translation("__WTTERM0009__da", mapping) == "Dünyada"
    proper = ProtectedTermMeta(
        sentinel="__WTTERM0010__",
        source_original="GAO YUAN",
        target_base="Gao Yuan",
        is_approved=True,
        proper_name=True,
    )
    pmap = {proper.sentinel: proper}
    assert restore_protected_translation("__WTTERM0010__xyz", pmap) == "Gao Yuan"
    assert restore_protected_translation("__WTTERM0010__DIR", pmap) == "Gao Yuan'dır"


def test_collapse_stem_doubles_keeps_inflected() -> None:
    assert collapse_stem_doubles("Son Seviye seviyesine kaldi.") == "Son seviyesine kaldi."
    # Eş-form tekrarlar (vurgu/ikileme) korunur.
    assert collapse_stem_doubles("yavas yavas ilerle.") == "yavas yavas ilerle."
    assert collapse_stem_doubles("cok cok guzel.") == "cok cok guzel."
    assert collapse_stem_doubles("cok heyecanli bir gundu.") == "cok heyecanli bir gundu."
    assert collapse_stem_doubles("") == ""


def _money_meta() -> ProtectedTermMeta:
    return ProtectedTermMeta(
        sentinel="__WTTERM0001__",
        source_original="MONEY",
        source_term="MONEY",
        target_base="Para",
        is_approved=True,
        proper_name=False,
    )


def _echo_meta() -> ProtectedTermMeta:
    return ProtectedTermMeta(
        sentinel="__WTTERM0002__",
        source_original="HYUNJI",
        source_term="HYUNJI",
        target_base="HYUNJI",
        is_approved=True,
        proper_name=False,
    )


def test_decapitalize_common_lock_mid_sentence() -> None:
    mapping = {"__WTTERM0001__": _money_meta()}
    assert (
        restore_protected_translation("Onun için bu çok fazla __WTTERM0001__.", mapping)
        == "Onun için bu çok fazla para."
    )
    assert (
        restore_protected_translation("__WTTERM0001__ cebimde.", mapping)
        == "Para cebimde."
    )


def test_decapitalize_skips_echo_and_shout() -> None:
    echo = {"__WTTERM0002__": _echo_meta()}
    # Yankı-hedef (ad) aynen durur.
    assert (
        restore_protected_translation("Bu __WTTERM0002__ odası.", echo)
        == "Bu HYUNJI odası."
    )
    # Bağırma bloğuna dokunulmaz.
    money = {"__WTTERM0001__": _money_meta()}
    assert (
        restore_protected_translation("VER __WTTERM0001__ HEMEN!", money)
        == "VER Para HEMEN!"
    )


def test_decapitalize_keeps_multiword_surfaces() -> None:
    # Çok-kelimeli yüzeyler ad/tamlama olabilir ("Gizli Diyar") — dokunulmaz.
    meta = ProtectedTermMeta(
        sentinel="__WTTERM0003__",
        source_original="SECRET REALM",
        source_term="SECRET REALM",
        target_base="Gizli Diyar",
        is_approved=True,
        proper_name=False,
    )
    mapping = {"__WTTERM0003__": meta}
    # Çekim biçimi restore'un işi; bu test yalnız decap'in dokunmadığını kilitler.
    assert (
        restore_protected_translation("Takım __WTTERM0003__'e girdi.", mapping)
        == "Takım Gizli Diyara girdi."
    )


def test_join_hyphen_splits() -> None:
    from core.translation.source_normalization import _join_hyphen_splits

    assert _join_hyphen_splits("HERE, IT'S DIFFER- ENT.") == "HERE, IT'S DIFFERENT."
    assert _join_hyphen_splits("REINCARNA- TED OLDUM") == "REINCARNATED OLDUM"
    # Dokunulmazlar: diyalog tiresi, aralıklı tire, bitişik bileşik.
    assert _join_hyphen_splits("- Hey, sen!") == "- Hey, sen!"
    assert _join_hyphen_splits("well - known") == "well - known"
    assert _join_hyphen_splits("well-known adam") == "well-known adam"


def test_find_name_glue_flags_mutation_and_missing_apostrophe() -> None:
    src = "I LEFT THE FIRST PART OF THE PAYMENT IN HYUNJI'S ROOM."
    assert find_name_glue(src, "Ödemenin ilk kısmını HYUNJInin odasında bıraktım.") == ["name_glue"]
    # Doğru imla geçer: kesmeli + birebir yankı.
    assert find_name_glue(src, "Ödemenin ilk kısmını HYUNJI'nin odasında bıraktım.") == []
    assert find_name_glue("LOOK, ALLEN!", "Bak, ALLEN!") == []
    # Tam yankı muaf (S0 Katman-1).
    assert find_name_glue("ZFO GOBLINS?!", "ZFO GOBLINS?!") == []
    # Küçük-harfler etkilenmez.
    assert find_name_glue("SHE SAW THE MONEY.", "Parayı gördü.") == []
    assert find_name_glue("ALLEN WENT HOME.", "Allen eve gitti.") == []


def test_echo_lock_is_proper_name() -> None:
    """Yankı-kilit (hedef==kaynak) özel-addır: ekler kesmeyle gelir."""
    from core.translation.protection import protect_source_text

    prepared, pmap = protect_source_text("HYUNJI'S ROOM", {"HYUNJI": "HYUNJI"}, set())
    assert "__WTTERM" in prepared
    meta = next(iter(pmap.values()))
    assert meta.proper_name is True
    assert restore_protected_translation("__WTTERM0001__nin odası.", pmap) == "HYUNJI'nin odası."
    prepared2, pmap2 = protect_source_text("HYUNJI CAME", {"HYUNJI": "HYUNJI"}, set())
    assert restore_protected_translation("__WTTERM0001__ geldi.", pmap2) == "HYUNJI geldi."
