"""Unit tests for QwenGGUFTranslationProviderV2 (Shootout V1)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from providers.translation.base import TranslationInput, TranslationItem
from providers.translation.qwen_gguf_translation_v2 import (
    DEFAULT_QWEN_MODEL_PATH,
    DEFAULT_QWEN_SERVER_URL,
    QwenGGUFTranslationProviderV2,
)


def _input(items: list[str]) -> TranslationInput:
    return TranslationInput(
        items=[
            TranslationItem(region_id=idx + 1, source=text, reading_order=idx + 1)
            for idx, text in enumerate(items)
        ]
    )


def test_qwen_translator_v2_init_defaults() -> None:
    provider = QwenGGUFTranslationProviderV2()
    assert provider.model_path == DEFAULT_QWEN_MODEL_PATH
    assert provider.server_url == DEFAULT_QWEN_SERVER_URL
    assert provider.gpu_layers == 99
    assert provider.name == "Qwen3.5-9B-GGUF-Translator-V2"


@patch("urllib.request.urlopen")
def test_qwen_translator_v2_chat_completion_payload(mock_urlopen) -> None:
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": "Frost Chain aynı anda üç hedefi sabitleyebilir."
                    }
                }
            ],
            "usage": {"prompt_tokens": 45, "completion_tokens": 12},
        }
    ).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    provider = QwenGGUFTranslationProviderV2()
    provider._loaded = True

    with patch.object(provider, "_check_health", return_value=True):
        output = provider.translate(_input(["Frost Chain can hold three targets at once."]))

    assert len(output.results) == 1
    item = output.results[0]
    assert item.translation == "Frost Chain aynı anda üç hedefi sabitleyebilir."
    assert item.requires_review is False

    # Inspect requested HTTP call
    req = mock_urlopen.call_args[0][0]
    assert req.full_url == "http://127.0.0.1:8083/v1/chat/completions"
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["temperature"] == 0.0
    assert payload["model"] == "qwen3.5-9b-translator"
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert "precise English to Turkish translator" in payload["messages"][0]["content"]


@patch("urllib.request.urlopen")
def test_qwen_translator_v2_reasoning_contamination_raises(mock_urlopen) -> None:
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "Thinking step 1...",
                        "content": "Çeviri",
                    }
                }
            ],
        }
    ).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    provider = QwenGGUFTranslationProviderV2()
    provider._loaded = True

    with patch.object(provider, "_check_health", return_value=True):
        output = provider.translate(_input(["Hello"]))

    assert output.results[0].translation is None
    assert "translation_server_error" in output.results[0].validation_warnings
    assert provider.metrics.reasoning_contamination_count >= 1


@patch("urllib.request.urlopen")
def test_qwen_translator_v2_chatbot_explanation_flagged(mock_urlopen) -> None:
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": "As a large language model, I can help you translate this sentence."
                    }
                }
            ],
        }
    ).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    provider = QwenGGUFTranslationProviderV2()
    provider._loaded = True

    with patch.object(provider, "_check_health", return_value=True):
        output = provider.translate(_input(["How are you?"]))

    assert output.results[0].translation is None
    assert "chatbot_or_explanation_output" in output.results[0].validation_warnings
    assert output.results[0].requires_review is True


@patch("urllib.request.urlopen")
def test_qwen_translator_v2_empty_translation_flagged(mock_urlopen) -> None:
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({"choices": [{"message": {"content": ""}}]}).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    provider = QwenGGUFTranslationProviderV2()
    provider._loaded = True

    with patch.object(provider, "_check_health", return_value=True):
        output = provider.translate(_input(["Test sentence"]))

    assert output.results[0].translation is None
    assert "empty_translation" in output.results[0].validation_warnings


def test_qwen_translator_v2_process_cleanup() -> None:
    provider = QwenGGUFTranslationProviderV2()
    mock_proc = MagicMock()
    provider._process = mock_proc
    provider._owned_process = True
    provider._loaded = True

    provider.unload()

    mock_proc.terminate.assert_called_once()
    assert provider._process is None
    assert provider._loaded is False
