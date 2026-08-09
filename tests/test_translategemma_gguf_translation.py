"""Unit tests for TranslateGemma GGUF translation provider, protection layer, and factory.
"""
import json
import unittest
from unittest.mock import MagicMock, patch

from core.translation.batcher import TranslationBatcher
from core.translation.profile_discovery import CandidateStore
from core.translation.protection import (
    detect_named_terms_in_items,
    protect_source_text,
    restore_protected_translation,
    validate_protected_terms,
)
from core.translation.series_profile import SeriesProfile
from core.translation.system_text import is_system_ui_line, translate_system_ui_line
from providers.translation import (
    QwenGGUFTranslationProvider,
    TranslateGemmaGGUFTranslationProvider,
    TranslationInput,
    TranslationItem,
    get_translation_provider,
)
from providers.translation.translategemma_gguf_translation import (
    _clean_translategemma_output,
    is_explanation_like_output,
)


class TestTranslateGemmaMigration(unittest.TestCase):

    def test_factory_default_and_explicit_backends(self):
        default_p = get_translation_provider()
        self.assertIsInstance(default_p, TranslateGemmaGGUFTranslationProvider)

        gemma_p = get_translation_provider(backend="translategemma_gguf")
        self.assertIsInstance(gemma_p, TranslateGemmaGGUFTranslationProvider)

        qwen_p = get_translation_provider(backend="qwen_gguf")
        self.assertIsInstance(qwen_p, QwenGGUFTranslationProvider)

    def test_official_request_payload_format(self):
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        captured_payload = {}

        def mock_query(prepared_text):
            return "Merhaba dünya.", 10, 5, 0.1

        inp = TranslationInput(
            items=[TranslationItem(region_id=1, source="Hello world.", reading_order=1)]
        )

        with patch.object(provider, "_query_official_translation", side_effect=mock_query), \
             patch.object(provider, "_check_health", return_value=True):
            out = provider.translate(inp)

        self.assertEqual(len(out.results), 1)
        self.assertEqual(out.results[0].translation, "Merhaba dünya.")

    def test_no_prompt_instructions_inside_prepared_source(self):
        profile = SeriesProfile(
            series_id="test_series",
            known_names={"ARIN SOL": "Arin Sol"},
            glossary={"SECRET REALM": "Gizli Diyar"},
        )
        item = TranslationItem(
            region_id=1, source="ARIN SOL entered the SECRET REALM.", reading_order=1
        )

        app_t = {"SECRET REALM": "Gizli Diyar", "ARIN SOL": "Arin Sol"}
        prepared_text, p_map = protect_source_text(item.source, app_t, set())

        # Assert no Qwen or instruction-style strings inside prepared text
        self.assertNotIn("Translate the following", prepared_text)
        self.assertNotIn("Context (", prepared_text)
        self.assertNotIn("Approved Terminology", prepared_text)

        # Assert tokens are protected
        self.assertIn("Gizli_Diyar", prepared_text)
        self.assertIn("Arin_Sol", prepared_text)

        # Assert restoration works naturally
        raw_tr = "Arin_Sol, Gizli_Diyar'a girdi."
        restored = restore_protected_translation(raw_tr, p_map)
        self.assertEqual(restored, "Arin Sol, Gizli Diyar'a girdi.")

    def test_named_ability_detection_and_consistency(self):
        items = [
            TranslationItem(region_id=1, source="It's called Phantom Thread.", reading_order=1),
            TranslationItem(region_id=2, source="Phantom Thread?", reading_order=2),
            TranslationItem(region_id=3, source="Activate Phantom Thread.", reading_order=3),
        ]

        detected = detect_named_terms_in_items(items)
        self.assertIn("Phantom Thread", detected)

        # Test protection preserves source form "Phantom Thread"
        prep_1, p_map_1 = protect_source_text(items[0].source, {}, detected)
        self.assertIn("Phantom_Thread", prep_1)

        raw_tr_1 = "Bunun adı Phantom_Thread."
        restored_1 = restore_protected_translation(raw_tr_1, p_map_1)
        self.assertEqual(restored_1, "Bunun adı Phantom Thread.")

    def test_approved_override_takes_precedence_over_named_detection(self):
        items = [TranslationItem(region_id=1, source="Activate Phantom Thread.", reading_order=1)]
        detected = detect_named_terms_in_items(items)

        # Explicit approved override: PHANTOM THREAD -> Hayalet İplik
        app_t = {"PHANTOM THREAD": "Hayalet İplik"}
        prep, p_map = protect_source_text(items[0].source, app_t, detected)

        self.assertIn("Hayalet_İplik", prep)
        restored = restore_protected_translation("Hayalet_İplik'i etkinleştir.", p_map)
        self.assertEqual(restored, "Hayalet İplik'i etkinleştir.")

    def test_system_ui_text_handling(self):
        s1 = "TITLE ACQUIRED: GRAVE WALKER"
        self.assertTrue(is_system_ui_line(s1))
        tr1 = translate_system_ui_line(s1)
        self.assertEqual(tr1, "Kazanılan Unvan: GRAVE WALKER")

        s2 = "CLASS ADVANCEMENT AVAILABLE"
        self.assertTrue(is_system_ui_line(s2))
        tr2 = translate_system_ui_line(s2)
        self.assertEqual(tr2, "Sınıf Yükseltmesi Mevcut")

        s3 = "ABILITY COOLDOWN: 18 SECONDS."
        self.assertTrue(is_system_ui_line(s3))
        tr3 = translate_system_ui_line(s3)
        self.assertEqual(tr3, "Yetenek Bekleme Süresi: 18 saniye")

    def test_explanation_output_guard(self):
        explanation_raw = """"Void Step?" ifadesinin anlamı bağlama göre değişebilir:
* Boş Adım
* Anlamsız Adım"""
        source = "Void Step?"

        self.assertTrue(is_explanation_like_output(explanation_raw, source))

        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        inp = TranslationInput(
            items=[TranslationItem(region_id=1, source=source, reading_order=1)]
        )

        with patch.object(provider, "_query_official_translation", return_value=(explanation_raw, 10, 50, 0.2)), \
             patch.object(provider, "_check_health", return_value=True):
            out = provider.translate(inp)

        self.assertEqual(len(out.results), 1)
        res = out.results[0]
        self.assertIsNone(res.translation)
        self.assertTrue(res.requires_review)
        self.assertIn("explanation_like_output", res.validation_warnings)

    def test_per_item_retry_and_http_error_isolation(self):
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        inp = TranslationInput(
            items=[
                TranslationItem(region_id=1, source="Item 1 failing.", reading_order=1),
                TranslationItem(region_id=2, source="Item 2 working.", reading_order=2),
            ]
        )

        def mock_query(prep_text):
            if "Item 1" in prep_text:
                raise urllib.error.URLError("Server timeout")
            return "Öge 2 çalışıyor.", 10, 5, 0.1

        with patch.object(provider, "_query_official_translation", side_effect=mock_query), \
             patch.object(provider, "_check_health", return_value=True):
            out = provider.translate(inp)

        self.assertEqual(len(out.results), 2)
        res1, res2 = out.results[0], out.results[1]

        # Item 1 failed with error isolation
        self.assertIsNone(res1.translation)
        self.assertTrue(res1.requires_review)
        self.assertIn("translation_server_error", res1.validation_warnings)

        # Item 2 succeeded
        self.assertEqual(res2.translation, "Öge 2 çalışıyor.")
        self.assertFalse(res2.requires_review)

    def test_server_identity_verification(self):
        provider = TranslateGemmaGGUFTranslationProvider(
            model_path=r"C:\AI\Models\translategemma-12b-it-q5_k_m.gguf",
            server_url="http://127.0.0.1:8081",
        )

        props_response = json.dumps({
            "model_path": "C:\\AI\\Models\\Qwen3.5-9B-Q5_K_M.gguf"  # WRONG MODEL!
        }).encode("utf-8")

        mock_health_resp = MagicMock()
        mock_health_resp.status = 200

        mock_props_resp = MagicMock()
        mock_props_resp.read.return_value = props_response

        def mock_urlopen(req, timeout=3):
            if "/props" in req.full_url:
                return mock_props_resp
            return mock_health_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            self.assertFalse(provider._check_health(), "Identity check should fail when model_path does not match!")

    def test_batcher_overhead_ignores_unrelated_profile_terms(self):
        # Create a large profile with 500 unrelated terms
        huge_names = {f"NAME_{i}": f"İsim_{i}" for i in range(250)}
        huge_glossary = {f"TERM_{i}": f"Terim_{i}" for i in range(250)}
        profile = SeriesProfile(series_id="huge_profile", known_names=huge_names, glossary=huge_glossary)

        inp = TranslationInput(
            items=[TranslationItem(region_id=1, source="Clean short dialogue.", reading_order=1)],
            profile=profile,
        )

        batcher = TranslationBatcher()
        sub_batches = batcher.create_batches(inp)

        # Batch overhead must not collapse the batch
        self.assertEqual(len(sub_batches), 1)
        self.assertEqual(len(sub_batches[0].items), 1)


if __name__ == "__main__":
    unittest.main()
