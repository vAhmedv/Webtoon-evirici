"""Unit tests for GeminiTranslationProvider."""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from providers.translation.base import TranslationInput, TranslationItem
from providers.translation.gemini_translation import (
    GeminiTranslationProvider,
    DEFAULT_GEMINI_MODEL,
)


def test_gemini_provider_init():
    provider = GeminiTranslationProvider(api_key="test-key-123", model_name="gemini-2.5-flash")
    assert provider.is_loaded is True
    assert provider.name == "Gemini (gemini-2.5-flash)"
    assert provider.version == "2.5"


def test_gemini_provider_missing_key_raises():
    with patch.dict("os.environ", {}, clear=True):
        provider = GeminiTranslationProvider(api_key="")
        assert provider.is_loaded is False
        with pytest.raises(ValueError, match="Gemini API key is required"):
            provider.load()


def test_gemini_parse_json_translations():
    # Direct dictionary with translations array
    text1 = json.dumps({"translations": [{"id": 1, "turkish": "Naber?"}, {"id": 2, "turkish": "İyiyim."}]})
    res1 = GeminiTranslationProvider._parse_json_translations(text1)
    assert res1 == {1: "Naber?", 2: "İyiyim."}

    # Markdown code fence JSON
    text2 = f"```json\n{text1}\n```"
    res2 = GeminiTranslationProvider._parse_json_translations(text2)
    assert res2 == {1: "Naber?", 2: "İyiyim."}

    # List of items directly
    text3 = json.dumps([{"id": 3, "turkish": "Kaç!"}])
    res3 = GeminiTranslationProvider._parse_json_translations(text3)
    assert res3 == {3: "Kaç!"}

    # Empty text
    assert GeminiTranslationProvider._parse_json_translations("") == {}


def test_gemini_translate_mocked():
    provider = GeminiTranslationProvider(api_key="fake-key-abc")

    mock_response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps({
                                "translations": [
                                    {"id": 101, "turkish": "Kahretsin, nereden çıktı bu adam?!"},
                                    {"id": 102, "turkish": "Hemen geri çekilin!"}
                                ]
                            })
                        }
                    ]
                }
            }
        ]
    }

    mock_resp = io.BytesIO(json.dumps(mock_response_data).encode("utf-8"))

    with patch("urllib.request.urlopen", return_value=mock_resp):
        inp = TranslationInput(
            items=[
                TranslationItem(region_id=101, source="Damn it, where did this guy come from?!"),
                TranslationItem(region_id=102, source="Fall back immediately!"),
            ]
        )
        out = provider.translate(inp)

    assert len(out.results) == 2
    assert out.results[0].region_id == 101
    assert out.results[0].translation == "Kahretsin, nereden çıktı bu adam?!"
    assert out.results[0].requires_review is False
    assert out.results[1].region_id == 102
    assert out.results[1].translation == "Hemen geri çekilin!"


def test_gemini_translate_batch_mocked():
    provider = GeminiTranslationProvider(api_key="fake-key-abc")

    mock_response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps({
                                "translations": [
                                    {"id": 1, "turkish": "Selam!"},
                                    {"id": 2, "turkish": "Görüşürüz!"}
                                ]
                            })
                        }
                    ]
                }
            }
        ]
    }

    mock_resp = io.BytesIO(json.dumps(mock_response_data).encode("utf-8"))

    with patch("urllib.request.urlopen", return_value=mock_resp):
        texts = ["Hello!", "Bye!"]
        results = provider.translate_batch(texts)

    assert results == ["Selam!", "Görüşürüz!"]


def test_gemini_verify_connection_success():
    mock_data = {
        "candidates": [
            {"content": {"parts": [{"text": "Merhaba"}]}}
        ]
    }
    mock_resp = io.BytesIO(json.dumps(mock_data).encode("utf-8"))

    with patch("urllib.request.urlopen", return_value=mock_resp):
        ok, msg = GeminiTranslationProvider.verify_connection(api_key="valid-key", model_name="gemini-2.0-flash")
        assert ok is True
        assert "Bağlantı Başarılı" in msg
        assert "Merhaba" in msg


def test_gemini_verify_connection_empty_key():
    ok, msg = GeminiTranslationProvider.verify_connection(api_key="")
    assert ok is False
    assert "boş olamaz" in msg
