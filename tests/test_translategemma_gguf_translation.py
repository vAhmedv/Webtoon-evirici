"""Unit tests for TranslateGemma GGUF translation provider and factory.
"""
import json
import unittest
from unittest.mock import MagicMock, patch

from core.translation.profile_discovery import CandidateStore
from core.translation.series_profile import SeriesProfile
from providers.translation import (
    QwenGGUFTranslationProvider,
    TranslateGemmaGGUFTranslationProvider,
    TranslationInput,
    TranslationItem,
    get_translation_provider,
)
from providers.translation.translategemma_gguf_translation import (
    _clean_translategemma_output,
    build_translategemma_user_prompt,
)


class TestTranslateGemmaGGUFTranslation(unittest.TestCase):

    def test_factory_default_and_explicit_backends(self):
        default_p = get_translation_provider()
        self.assertIsInstance(default_p, TranslateGemmaGGUFTranslationProvider)

        gemma_p = get_translation_provider(backend="translategemma_gguf")
        self.assertIsInstance(gemma_p, TranslateGemmaGGUFTranslationProvider)

        qwen_p = get_translation_provider(backend="qwen_gguf")
        self.assertIsInstance(qwen_p, QwenGGUFTranslationProvider)

    def test_prompt_construction_no_qwen_system_prompt(self):
        profile = SeriesProfile(
            series_id="test_series",
            known_names={"ARIN SOL": "Arin Sol", "HU SAN": "Hu San"},
            glossary={"SPIRIT CORE": "Ruh Çekirdeği", "MANA CORE": "Mana Çekirdeği"},
        )
        item = TranslationItem(
            region_id=1, source="ARIN SOL activated the SPIRIT CORE.", reading_order=1
        )
        ctx_item = TranslationItem(
            region_id=0, source="The underground chamber was quiet.", reading_order=0
        )

        prompt = build_translategemma_user_prompt(
            item=item, context_items=[ctx_item], profile=profile
        )

        # Assert Qwen system prompt is NOT present
        self.assertNotIn("You are a professional English", prompt)
        self.assertNotIn("OUTPUT SCHEMA", prompt)
        self.assertNotIn("fidelity_flags", prompt)

        # Assert context is reference-only
        self.assertIn("Context (for background understanding only", prompt)
        self.assertIn("The underground chamber was quiet.", prompt)

        # Assert relevant approved terms included
        self.assertIn("ARIN SOL = Arin Sol", prompt)
        self.assertIn("SPIRIT CORE = Ruh Çekirdeği", prompt)

        # Assert unrelated terms are filtered out
        self.assertNotIn("MANA CORE", prompt)
        self.assertNotIn("HU SAN", prompt)

    def test_output_cleanup_strips_control_tokens(self):
        raw_1 = "Durun. O öyle olmadı.<|file_separator|>\nBekleyin."
        clean_1 = _clean_translategemma_output(raw_1)
        self.assertEqual(clean_1, "Durun. O öyle olmadı.")

        raw_2 = '"Kaptan, başka bir giriş bulduk."'
        clean_2 = _clean_translategemma_output(raw_2)
        self.assertEqual(clean_2, "Kaptan, başka bir giriş bulduk.")

        raw_3 = "Normal Türkçe cümle bozulmasın."
        clean_3 = _clean_translategemma_output(raw_3)
        self.assertEqual(clean_3, "Normal Türkçe cümle bozulmasın.")

    def test_no_fabricated_metadata(self):
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        inp = TranslationInput(
            items=[TranslationItem(region_id=1, source="Hello world.", reading_order=1)]
        )

        with patch.object(
            provider,
            "_query_chat_completion",
            return_value=("Merhaba dünya.", 10, 5, 0.1),
        ), patch.object(provider, "_check_health", return_value=True):

            out = provider.translate(inp)

        self.assertEqual(len(out.results), 1)
        res = out.results[0]
        self.assertEqual(res.translation, "Merhaba dünya.")
        self.assertEqual(res.fidelity_flags, [])
        self.assertEqual(res.term_usages, [])
        self.assertFalse(res.requires_review)

    def test_candidate_store_remains_unmutated_by_translation(self):
        store = CandidateStore(series_id="test_series")
        store_before = json.dumps(store.to_dict())

        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        inp = TranslationInput(
            items=[TranslationItem(region_id=1, source="Test item.", reading_order=1)],
            candidate_store=store,
        )

        with patch.object(
            provider,
            "_query_chat_completion",
            return_value=("Test ögesi.", 10, 5, 0.1),
        ), patch.object(provider, "_check_health", return_value=True):

            provider.translate(inp)

        store_after = json.dumps(store.to_dict())
        self.assertEqual(store_before, store_after, "CandidateStore was directly mutated by TranslateGemma provider!")

    def test_error_handling_empty_response(self):
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        inp = TranslationInput(
            items=[TranslationItem(region_id=1, source="Broken source line.", reading_order=1)]
        )

        with patch.object(
            provider,
            "_query_chat_completion",
            return_value=("", 10, 0, 0.1),
        ), patch.object(provider, "_check_health", return_value=True):

            out = provider.translate(inp)

        self.assertEqual(len(out.results), 1)
        res = out.results[0]
        self.assertIsNone(res.translation)
        self.assertTrue(res.requires_review)
        self.assertIn("empty_translation", res.validation_warnings)


if __name__ == "__main__":
    unittest.main()
