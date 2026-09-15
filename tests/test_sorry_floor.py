"""SORRY tabanı: model sürçmesi deterministik zemine oturur (sentetik)."""

from application.chapter_analyzer import _FATAL_TRANSLATION_WARNINGS
from providers.translation.hy_mt2_gguf_translation import post_process_turkish_translation as pp


def test_sorry_misinflection_fixed() -> None:
    assert pp("Ağabey, ben özürüm.", "OPPA, I'M SORRY.") == "Ağabey, ben özür dilerim."
    assert pp("Ben gerçekten özürüm.", "I'M REALLY SORRY.") == "Ben gerçekten özür dilerim."


def test_sorry_capital_preserved() -> None:
    assert pp("Özürüm.", "I'M SORRY.") == "Özür dilerim."


def test_sorry_rule_needs_sorry_source() -> None:
    # Meşru özne-kullanım ezilmez (kaynakta SORRY yok).
    assert pp("özürüm burada duruyor", "MY APOLOGY STAYS HERE.") == "özürüm burada duruyor"


def test_sorry_accusative_untouched() -> None:
    # "özrümü" ayrı biçim — sınır eşleşmez, SORRY olsa bile dokunulmaz.
    assert pp("özrümü kabul et", "SORRY, TAKE MY APOLOGY.") == "özrümü kabul et"


def test_kinship_is_fatal() -> None:
    assert "kinship_ambiguous" in _FATAL_TRANSLATION_WARNINGS
