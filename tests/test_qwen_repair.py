"""Focused unit tests for Qwen OCR repair provider.

Tests:
- Parser: JSON resolved output
- Parser: JSON unresolved output
- Parser: verbose/malformed output rejected
- Parser: invalid/empty/CJK/repetition -> unresolved
- Agreement skip: Qwen not called when needs_repair=False
- VRAM metrics initialization
"""
from __future__ import annotations

import pytest
from PIL import Image
from unittest.mock import MagicMock
import json

from providers.ocr.qwen_repair import (
    OCRAdjudicatedResult,
    QwenRepairConfig,
    QwenRepairMetrics,
    QwenRepairProvider,
    adjudicate_ocr,
)
from providers.ocr.repair import OCRRepairInput, OCRRepairResult


# --- Parser tests (no model needed) ---

def _make_provider():
    return QwenRepairProvider()


def _unresolved_json():
    return _make_provider()._parse_output(
        '{"status": "unresolved", "text": null}'
    )


def _resolved_json(text="LUO TIAN"):
    return _make_provider()._parse_output(
        f'{{"status": "resolved", "text": "{text}"}}'
    )


class TestParseOutput:
    def test_resolved_json(self):
        r = _resolved_json("PUSHOVERS")
        assert not r.unresolved
        assert r.repaired_text == "PUSHOVERS"
        assert r.changed

    def test_unresolved_json(self):
        r = _unresolved_json()
        assert r.unresolved
        assert r.repaired_text is None
        assert not r.changed

    def test_empty_output(self):
        r = _make_provider()._parse_output("")
        assert r.unresolved
        assert r.repaired_text is None

    def test_placeholder_output(self):
        r = _make_provider()._parse_output(
            '{"status": "resolved", "text": "<exact text>"}'
        )
        assert r.unresolved

    def test_cjk_output(self):
        r = _make_provider()._parse_output("这是一个中文测试")
        assert r.unresolved

    def test_obvious_repetition(self):
        r = _make_provider()._parse_output("LUO LUO LUO TIAN TIAN TIAN")
        assert r.unresolved

    def test_natural_language_fallback(self):
        raw = 'The model reads: "HU SAN" based on the visual evidence.'
        r = _make_provider()._parse_output(raw)
        assert r.unresolved
        assert r.metadata["rejection_reason"] == "malformed_or_verbose_output"

    def test_natural_language_line_pattern(self):
        raw = 'Line 1: "HELLO WORLD"\nLine 2: "FOO BAR"'
        r = _make_provider()._parse_output(raw)
        assert r.unresolved

    def test_non_json_falls_back_to_natural(self):
        raw = "I see the text says PUSHOVERS clearly."
        r = _make_provider()._parse_output(raw)
        assert r.unresolved

    def test_raw_output_truncated(self):
        big = "x" * 1000
        r = _make_provider()._parse_output(big)
        assert r.unresolved
        assert len(r.metadata.get("raw_output", "")) <= 500

    def test_verbose_json_with_reason_is_rejected(self):
        r = _make_provider()._parse_output(
            '{"status":"resolved","text":"HELLO","reason":"because"}'
        )
        assert r.unresolved
        assert r.metadata["rejection_reason"] == "invalid_output_contract"

    def test_resolved_word_is_not_accepted_as_ocr_text(self):
        r = _make_provider()._parse_output('{"status":"resolved","text":"resolved"}')
        assert r.unresolved


class TestAdjudicate:
    def _verdict(self, needs_repair, accepted_text="accepted", reason="word_difference"):
        class _V:
            pass
        v = _V()
        v.needs_repair = needs_repair
        v.accepted_text = accepted_text
        v.source = "primary"
        v.reason = reason
        v.primary_raw = "LHO TIAN"
        v.primary_normalized = "LHO TIAN"
        v.verifier_raw = "LUO TIAN"
        v.verifier_normalized = "LUO TIAN"
        return v

    def test_agreement_skip_qwen_not_called(self):
        provider = QwenRepairProvider()
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        verdict = self._verdict(needs_repair=False, accepted_text="RELAX KID")
        outcome = adjudicate_ocr(verdict, img, provider)
        assert outcome.repair_result is None
        assert outcome.clean_source_text == "RELAX KID"
        assert not outcome.requires_review
        assert not provider.is_loaded

    def test_unloaded_provider_skips_repair(self):
        provider = QwenRepairProvider()
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        verdict = self._verdict(needs_repair=True)
        outcome = adjudicate_ocr(verdict, img, provider)
        assert outcome.repair_result is None
        assert outcome.clean_source_text is None
        assert outcome.requires_review


class TestMetrics:
    def test_metrics_defaults(self):
        m = QwenRepairMetrics()
        assert m.model_load_vram_gb == 0.0
        assert m.peak_vram_gb == 0.0
        assert m.repair_model == ""
        assert m.repair_calls == 0
        assert m.accepted_repairs == 0
        assert m.rejected_repairs == 0

    def test_config_defaults(self):
        cfg = QwenRepairConfig()
        assert cfg.max_new_tokens == 96
        assert cfg.max_memory_gb == 12


class TestProcessOwnership:
    def test_unload_does_not_terminate_external_server_process(self):
        external = MagicMock()
        external.pid = 4321
        provider = QwenRepairProvider()
        provider._server_process = external
        provider._owns_server = False
        provider._loaded = True

        provider.unload()

        external.terminate.assert_not_called()
        external.kill.assert_not_called()
        assert provider._server_process is None

    def test_external_server_identity_must_match_configured_model(self, monkeypatch):
        provider = QwenRepairProvider()
        monkeypatch.setattr(
            provider,
            "_fetch_json",
            lambda path: {"model_path": "C:/models/other-model.gguf"},
        )
        compatible, identity = provider._check_server_identity(
            __import__("pathlib").Path(provider._config.model_path)
        )
        assert not compatible
        assert identity["props"]["model_path"].endswith("other-model.gguf")

    def test_external_server_identity_accepts_configured_model(self, monkeypatch):
        provider = QwenRepairProvider()
        monkeypatch.setattr(
            provider,
            "_fetch_json",
            lambda path: {"model_path": provider._config.model_path},
        )
        compatible, _ = provider._check_server_identity(
            __import__("pathlib").Path(provider._config.model_path)
        )
        assert compatible


def test_repair_request_disables_thinking(monkeypatch):
    provider = QwenRepairProvider()
    provider._loaded = True
    captured = {}

    class _Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"status\\":\\"unresolved\\",\\"text\\":null}"}}]}'

    def _urlopen(request, timeout):
        captured.update(json.loads(request.data.decode("utf-8")))
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    provider.repair(
        OCRRepairInput("", "", "X", "X", "primary_empty_verifier_filled"),
        Image.new("RGB", (20, 20), "white"),
    )
    assert captured["reasoning_effort"] == "none"
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["max_tokens"] == 96
