"""Tests for Batched / Multi-Line Prompting and Fallback in Hy-MT2 Translation."""

from unittest.mock import MagicMock, patch
import pytest

from providers.translation.base import (
    TranslationInput,
    TranslationItem,
    TranslationProvider,
)
from providers.translation.hy_mt2_gguf_translation import (
    HyMT2GGUFTranslationProvider,
    clean_hy_mt2_output,
    render_hy_mt2_prompt,
)


def test_translation_provider_translate_batch_default():
    class DummyTranslator(TranslationProvider):
        def load(self): pass
        def unload(self): pass
        def translate(self, inp):
            from providers.translation.base import TranslationOutput, TranslationOutputItem
            results = [
                TranslationOutputItem(
                    region_id=item.region_id,
                    source=item.source,
                    translation=f"TR_{item.source}",
                    raw_model_response=f"TR_{item.source}",
                )
                for item in inp.items
            ]
            return TranslationOutput(inputs=inp, results=results, raw_response="", repair_model="dummy")

    provider = DummyTranslator()
    res = provider.translate_batch(["Hello", "World"])
    assert res == ["TR_Hello", "TR_World"]
    assert provider.translate_batch([]) == []


def test_hy_mt2_batch_parsing_success():
    provider = HyMT2GGUFTranslationProvider(managed=False)
    provider._loaded = True

    items = [
        TranslationItem(region_id=1, source="I will forge the ultimate blade."),
        TranslationItem(region_id=2, source="Zero Fantasy Online is starting now."),
    ]
    inp = TranslationInput(items=items)

    mock_batch_reply = "[1] Nihai kılıcı döveceğim.\n[2] Zero Fantasy Online şimdi başlıyor."

    with patch.object(provider, "_check_health", return_value=True), \
         patch.object(provider, "_server_identity_compatible", return_value=True), \
         patch.object(provider, "_request_translation", return_value=(mock_batch_reply, mock_batch_reply, False)):
        out = provider.translate(inp, chunk_size=4)

    assert len(out.results) == 2
    assert out.results[0].translation == "Nihai kılıcı döveceğim."
    assert out.results[1].translation == "Zero Fantasy Online şimdi başlıyor."


def test_hy_mt2_batch_fallback_on_malformed_response():
    provider = HyMT2GGUFTranslationProvider(managed=False)
    provider._loaded = True

    items = [
        TranslationItem(region_id=1, source="Line one."),
        TranslationItem(region_id=2, source="Line two."),
    ]
    inp = TranslationInput(items=items)

    # Malformed output missing [2]
    mock_bad_batch = "Line one and two combined translation."

    def fake_request(prompt, label, max_tokens=None):
        if "batch" in label:
            return mock_bad_batch, mock_bad_batch, False
        if "Line one" in prompt:
            return "Satır bir.", "Satır bir.", False
        return "Satır iki.", "Satır iki.", False

    with patch.object(provider, "_check_health", return_value=True), \
         patch.object(provider, "_server_identity_compatible", return_value=True), \
         patch.object(provider, "_request_translation", side_effect=fake_request):
        out = provider.translate(inp, chunk_size=4)

    # Fallback should cleanly succeed for all items
    assert len(out.results) == 2
    assert out.results[0].translation == "Satır bir."
    assert out.results[1].translation == "Satır iki."


def test_hy_mt2_32_item_batch_translation():
    """Verifies that Hy-MT2 can translate a full 32-dialogue batch in one single call."""
    provider = HyMT2GGUFTranslationProvider(managed=False)
    provider._loaded = True

    items = [
        TranslationItem(region_id=i + 1, source=f"Dialogue {i + 1}")
        for i in range(32)
    ]
    inp = TranslationInput(items=items)

    mock_batch_reply = "\n".join(f"[{i + 1}] Diyalog {i + 1}" for i in range(32))

    call_count = 0
    def fake_request(prompt, label, max_tokens=None):
        nonlocal call_count
        call_count += 1
        return mock_batch_reply, mock_batch_reply, False

    with patch.object(provider, "_check_health", return_value=True), \
         patch.object(provider, "_server_identity_compatible", return_value=True), \
         patch.object(provider, "_request_translation", side_effect=fake_request):
        out = provider.translate(inp, chunk_size=32)

    assert call_count == 1
    assert len(out.results) == 32
    for i in range(32):
        assert out.results[i].translation == f"Diyalog {i + 1}"


def test_hy_mt2_secondary_pattern_recovery():
    """Verifies recovery when the model formats responses as '1. text' instead of '[1] text'."""
    provider = HyMT2GGUFTranslationProvider(managed=False)
    provider._loaded = True

    items = [
        TranslationItem(region_id=1, source="First line"),
        TranslationItem(region_id=2, source="Second line"),
        TranslationItem(region_id=3, source="Third line"),
    ]
    inp = TranslationInput(items=items)

    mock_dot_reply = "1. İlk satır\n2. İkinci satır\n3. Üçüncü satır"

    call_count = 0
    def fake_request(prompt, label, max_tokens=None):
        nonlocal call_count
        call_count += 1
        return mock_dot_reply, mock_dot_reply, False

    with patch.object(provider, "_check_health", return_value=True), \
         patch.object(provider, "_server_identity_compatible", return_value=True), \
         patch.object(provider, "_request_translation", side_effect=fake_request):
        out = provider.translate(inp, chunk_size=32)

    assert call_count == 1
    assert len(out.results) == 3
    assert out.results[0].translation == "İlk satır"
    assert out.results[1].translation == "İkinci satır"
    assert out.results[2].translation == "Üçüncü satır"

