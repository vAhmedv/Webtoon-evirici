"""Unit tests for generic translation provider behavior and prompt generation."""
import pytest
from core.translation.series_profile import SeriesProfile
from providers.translation.base import TranslationInput, TranslationItem
from providers.translation.qwen_translation import QwenTranslationProvider, _SYSTEM_PROMPT


def test_no_hardcoded_koharu_terms_in_system_prompt():
    # Production system prompt must contain NO Koharu or specific series hardcoded strings
    hardcoded_terms = ["luo tian", "hu san", "gao yuan", "ability user", "secret realm", "blackwind ravine"]
    for term in hardcoded_terms:
        assert term not in _SYSTEM_PROMPT.lower()


def test_build_prompt_empty_profile():
    provider = QwenTranslationProvider()
    inp = TranslationInput(
        items=[TranslationItem(region_id=1, source="Hello world", reading_order=0)]
    )
    prompt = provider._build_prompt(inp)
    assert "Dialogue bubbles to translate:" in prompt
    assert "[0] id=1 | Hello world" in prompt
    assert "Glossary / Terminology guidance:" not in prompt


def test_build_prompt_with_profile_and_context():
    provider = QwenTranslationProvider()
    profile = SeriesProfile(
        series_id="generic_manhwa",
        known_names={"HERO": "Kahraman"},
        glossary={"DUNGEON": "zindan"},
        notes=["Keep tone dark"],
    )
    context_item = TranslationItem(region_id=10, source="Background context text", reading_order=0)
    item = TranslationItem(region_id=11, source="Look at the HERO", reading_order=1)

    inp = TranslationInput(
        items=[item],
        profile=profile,
        context_items=[context_item],
    )
    prompt = provider._build_prompt(inp)

    assert "CONTEXT ONLY (Do NOT translate these; for background understanding only):" in prompt
    assert "[0] id=10 | Background context text" in prompt
    assert "Dialogue bubbles to translate:" in prompt
    assert "[1] id=11 | Look at the HERO" in prompt
    assert "HERO -> Kahraman" in prompt
    assert "DUNGEON -> zindan" in prompt
    assert "Keep tone dark" in prompt
