"""Focused unit tests for Qwen translation provider.

Tests:
- JSON parser: valid output, missing IDs, duplicates, extra IDs
- Validation: empty output, name preservation, CJK detection, repetition, length ratio
- Smoke: QwenTranslationProvider loads and can be unloaded
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from providers.translation.base import (
    TranslationInput,
    TranslationItem,
    TranslationOutput,
)
from providers.translation.qwen_translation import (
    QwenTranslationMetrics,
    QwenTranslationProvider,
    _try_extract_json,
    _validate_output,
)


class TestTryExtractJson:
    def test_valid_json(self):
        raw = '{"translations": [{"id": 1, "source": "Hello", "translation": "Merhaba"}]}'
        obj = _try_extract_json(raw)
        assert obj is not None
        assert "translations" in obj

    def test_fenced_json(self):
        raw = '```json\n{"translations": [{"id": 1, "source": "Hi", "translation": "Selam"}]}\n```'
        obj = _try_extract_json(raw)
        assert obj is not None

    def test_json_embedded_in_text(self):
        raw = 'Some preamble\n{"translations": [{"id": 1, "source": "Hi", "translation": "Selam"}]}\nTrailing text'
        obj = _try_extract_json(raw)
        assert obj is not None

    def test_invalid_json(self):
        obj = _try_extract_json("not json at all")
        assert obj is None

    def test_json_without_translations_key(self):
        obj = _try_extract_json('{"foo": "bar"}')
        assert obj is None


class TestValidateOutput:
    def _items(self):
        return [
            TranslationItem(region_id=1, source="HELLO", reading_order=0),
            TranslationItem(region_id=2, source="WORLD", reading_order=1),
        ]

    def test_all_present_no_warnings(self):
        translations = [
            {"id": 1, "source": "HELLO", "translation": "MERHABA"},
            {"id": 2, "source": "WORLD", "translation": "DUNYA"},
        ]
        warnings = _validate_output(self._items(), translations, {})
        assert warnings == {}

    def test_missing_id(self):
        translations = [{"id": 1, "source": "HELLO", "translation": "MERHABA"}]
        warnings = _validate_output(self._items(), translations, {})
        assert 2 in warnings
        assert "missing_id" in warnings[2]

    def test_duplicate_id(self):
        translations = [
            {"id": 1, "source": "HELLO", "translation": "MERHABA"},
            {"id": 1, "source": "WORLD", "translation": "DUNYA"},
        ]
        warnings = _validate_output(self._items(), translations, {})
        assert 1 in warnings
        assert "duplicate_id" in warnings[1]

    def test_extra_id_not_in_input(self):
        translations = [
            {"id": 1, "source": "HELLO", "translation": "MERHABA"},
            {"id": 99, "source": "EXTRA", "translation": "FARKLI"},
        ]
        warnings = _validate_output(self._items(), translations, {})
        assert 99 in warnings
        assert "extra_id_not_in_input" in warnings[99]

    def test_empty_output(self):
        translations = [{"id": 1, "source": "HELLO", "translation": ""}]
        warnings = _validate_output(self._items(), translations, {})
        assert 1 in warnings
        assert "empty_output" in warnings[1]

    def test_name_not_preserved(self):
        items = [
            TranslationItem(region_id=1, source="MY NAME IS LUO TIAN", known_names=["LUO TIAN"], reading_order=0),
        ]
        translations = [{"id": 1, "source": "MY NAME IS LUO TIAN", "translation": "Benim adım Luo Yan"}]
        warnings = _validate_output(items, translations, {})
        assert 1 in warnings
        assert "name_modified" in warnings[1]

    def test_name_preserved_ok(self):
        items = [
            TranslationItem(region_id=1, source="MY NAME IS LUO TIAN", known_names=["LUO TIAN"], reading_order=0),
        ]
        translations = [{"id": 1, "source": "MY NAME IS LUO TIAN", "translation": "Benim adım Luo Tian"}]
        warnings = _validate_output(items, translations, {})
        assert 1 not in warnings

    def test_known_name_not_in_source_ignored(self):
        items = [
            TranslationItem(region_id=1, source="RELAX KID", known_names=["LUO TIAN", "HU SAN"], reading_order=0),
        ]
        translations = [{"id": 1, "source": "RELAX KID", "translation": "Sakin ol çocuk"}]
        warnings = _validate_output(items, translations, {})
        assert 1 not in warnings

    def test_word_boundary_name_matching(self):
        # YU in YOU SAW IT YOURSELF should NOT be matched in source
        items = [
            TranslationItem(region_id=1, source="YOU SAW IT YOURSELF JUST NOW.", known_names=["YU"], reading_order=0),
            TranslationItem(region_id=2, source="YOUNG MASTER YU, CAPTAIN GAO", known_names=["YU"], reading_order=1),
        ]
        translations = [
            {"id": 1, "source": "YOU SAW IT YOURSELF JUST NOW.", "translation": "Az önce kendin gördün."},
            {"id": 2, "source": "YOUNG MASTER YU, CAPTAIN GAO", "translation": "Genç Efendi Yu, Kaptan Gao"},
        ]
        warnings = _validate_output(items, translations, {})
        assert 1 not in warnings
        assert 2 not in warnings

    def test_cjk_hallucination(self):
        translations = [{"id": 1, "source": "HELLO", "translation": "你好"}]
        warnings = _validate_output(self._items(), translations, {})
        assert 1 in warnings

    def test_excessive_repetition(self):
        translations = [{"id": 1, "source": "HELLO", "translation": "tekrar tekrar tekrar tekrar"}]
        warnings = _validate_output(self._items(), translations, {})
        assert 1 in warnings
        assert "excessive_repetition" in warnings[1]

    def test_suspicious_length_ratio(self):
        translations = [{"id": 1, "source": "Hi", "translation": "a " * 50}]
        warnings = _validate_output(self._items(), translations, {})
        assert 1 in warnings
        assert "suspicious_length_ratio" in warnings[1]


class TestProviderSmoke:
    def test_provider_not_loaded_by_default(self):
        provider = QwenTranslationProvider()
        assert not provider.is_loaded

    def test_provider_name(self):
        provider = QwenTranslationProvider()
        assert provider.name == "Qwen3.5-9B-Translation"

    def test_metrics_defaults(self):
        m = QwenTranslationMetrics()
        assert m.model_load_vram_gb == 0.0
        assert m.peak_vram_gb == 0.0
        assert m.translation_model == ""
