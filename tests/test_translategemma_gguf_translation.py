"""Unit tests for TranslateGemma GGUF translation provider, template renderer, protection layer, sentinels, and factory.
"""
import json
import unittest
from unittest.mock import MagicMock, patch

from core.config import Config, load_config
from core.translation.batcher import TranslationBatcher
from core.translation.profile_discovery import (
    CandidateStore,
    find_candidate_phrase_matches,
)
from core.translation.protection import (
    ProtectedTermMeta,
    contains_unrestored_protected_term,
    detect_named_terms_in_items,
    has_untranslated_source_prose,
    is_term_only_source,
    protect_source_text,
    restore_protected_translation,
    validate_protected_terms,
)
from core.translation.series_profile import SeriesProfile
from core.translation.system_text import is_system_ui_line, translate_system_ui_line
from core.translation.translategemma_template import render_translategemma_prompt
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


class TestTranslateGemmaHardening(unittest.TestCase):

    def test_factory_default_and_explicit_backends(self):
        default_p = get_translation_provider()
        self.assertIsInstance(default_p, TranslateGemmaGGUFTranslationProvider)

        gemma_p = get_translation_provider(backend="translategemma_gguf")
        self.assertIsInstance(gemma_p, TranslateGemmaGGUFTranslationProvider)

        qwen_p = get_translation_provider(backend="qwen_gguf")
        self.assertIsInstance(qwen_p, QwenGGUFTranslationProvider)

    def test_render_translategemma_prompt_structure(self):
        """Test render_translategemma_prompt template output against specification."""
        source = "Because swords are useful."
        prompt = render_translategemma_prompt(source, source_lang_code="en", target_lang_code="tr")

        # Source appears exactly once
        self.assertEqual(prompt.count(source), 1)

        # Language direction
        self.assertIn("English", prompt)
        self.assertIn("Turkish", prompt)

        # Special Gemma turn tokens
        self.assertIn("<bos>", prompt)
        self.assertIn("<start_of_turn>user", prompt)
        self.assertIn("<end_of_turn>", prompt)
        self.assertIn("<start_of_turn>model", prompt)

        # FORBIDDEN strings check
        self.assertNotIn("Qwen", prompt)
        self.assertNotIn("Glossary", prompt)
        self.assertNotIn("Context", prompt)

    def test_raw_completion_http_payload(self):
        """Intercept urllib.request.Request and verify exact /completion JSON payload structure."""
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        captured_payloads = []
        captured_urls = []

        def mock_urlopen(req, timeout=60.0):
            url_str = req.full_url if hasattr(req, "full_url") else str(req)
            captured_urls.append(url_str)
            resp = MagicMock()
            resp.__enter__.return_value = resp
            resp.status = 200
            if "/props" in url_str:
                resp.read.return_value = json.dumps({"model_path": r"C:\AI\Models\translategemma-12b-it-q5_k_m.gguf"}).encode("utf-8")
                return resp
            if "/health" in url_str:
                resp.read.return_value = b"OK"
                return resp

            if hasattr(req, "data") and req.data:
                data = json.loads(req.data.decode("utf-8"))
                captured_payloads.append(data)

            resp.read.return_value = json.dumps({
                "content": "Çünkü kılıçlar kullanışlıdır.",
                "tokens_evaluated": 15,
                "tokens_predicted": 8,
            }).encode("utf-8")
            return resp

        inp = TranslationInput(
            items=[TranslationItem(region_id=1, source="Because swords are useful.", reading_order=1)]
        )

        with patch("urllib.request.urlopen", side_effect=mock_urlopen), \
             patch.object(provider, "_check_health", return_value=True):
            out = provider.translate(inp)

        # Verify endpoint ends with /completion
        completion_urls = [u for u in captured_urls if u.endswith("/completion")]
        self.assertEqual(len(completion_urls), 1)

        self.assertEqual(len(captured_payloads), 1)
        payload = captured_payloads[0]

        # Verify payload contains 'prompt'
        self.assertIn("prompt", payload)

        # Verify payload DOES NOT contain 'messages' or 'content' JSON objects
        self.assertNotIn("messages", payload)
        self.assertNotIn("content", payload)

        # Result translation verified
        self.assertEqual(out.results[0].translation, "Çünkü kılıçlar kullanışlıdır.")

    def test_opaque_sentinels_and_boundary_safety(self):
        app_t = {"SECRET REALM": "Gizli Diyar", "YU": "Yu"}
        named_terms = {"Phantom Thread"}

        source = "YOUR team entered the SECRET REALM to find Phantom Thread."
        prep_text, p_map = protect_source_text(source, app_t, named_terms)

        self.assertIn("__WTTERM", prep_text)
        self.assertNotIn("Secret_Realm", prep_text)
        self.assertNotIn("Phantom_Thread", prep_text)
        self.assertIn("YOUR", prep_text)

        raw_tr = "Takımınız __WTTERM0001__'e girerek __WTTERM0002__'i buldu."
        restored = restore_protected_translation(raw_tr, p_map)
        self.assertIn("Gizli Diyar", restored)
        self.assertIn("Phantom Thread", restored)

    def test_proper_name_vs_common_term_morphology_restoration(self):
        meta_common = ProtectedTermMeta(
            sentinel="__WTTERM0001__",
            source_original="Inner Disciple",
            target_base="İç Mürit",
            is_approved=True,
            proper_name=False,
        )
        meta_proper = ProtectedTermMeta(
            sentinel="__WTTERM0002__",
            source_original="Luo Tian",
            target_base="Luo Tian",
            is_approved=True,
            proper_name=True,
        )

        p_map = {"__WTTERM0001__": meta_common, "__WTTERM0002__": meta_proper}

        tr_text = "Sadece __WTTERM0001__'ler ve __WTTERM0002__'in müritleri."
        restored = restore_protected_translation(tr_text, p_map)

        self.assertIn("İç Müritler", restored)
        self.assertIn("Luo Tian'ın", restored)

    def test_term_only_bypass(self):
        app_t = {"SECRET REALM": "Gizli Diyar"}
        named_t = {"Phantom Thread"}

        is_b1, tr1 = is_term_only_source("Phantom Thread?", app_t, named_t)
        self.assertTrue(is_b1)
        self.assertEqual(tr1, "Phantom Thread?")

        is_b2, tr2 = is_term_only_source("Secret Realm!", app_t, named_t)
        self.assertTrue(is_b2)
        self.assertEqual(tr2, "Gizli Diyar!")

        is_b3, tr3 = is_term_only_source("Activate Phantom Thread.", app_t, named_t)
        self.assertFalse(is_b3)

        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        inp = TranslationInput(
            items=[TranslationItem(region_id=1, source="Phantom Thread?", reading_order=1)]
        )

        with patch.object(provider, "_check_health", return_value=True):
            out = provider.translate(inp)

        self.assertEqual(out.results[0].translation, "Phantom Thread?")
        self.assertEqual(provider.metrics.generation_call_count, 0)
        self.assertEqual(provider.metrics.term_only_bypass_count, 1)

    def test_normal_sentence_with_named_ability_calls_generation(self):
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        query = MagicMock(return_value=("__WTTERM0001__'i etkinleştir.", 8, 5, 0.1))
        inp = TranslationInput(
            items=[TranslationItem(region_id=1, source="Activate Frost Chain.", reading_order=1)]
        )

        with patch.object(provider, "_query_official_translation", query), \
             patch.object(provider, "_check_health", return_value=True):
            out = provider.translate(inp)

        result = out.results[0]
        self.assertEqual(query.call_count, 1)
        self.assertEqual(provider.metrics.generation_call_count, 1)
        self.assertEqual(result.translation, "Frost Chain'i etkinleştir.")
        self.assertNotIn("Activate", result.translation)
        self.assertFalse(result.requires_review)

    def test_frost_chain_term_only_provider_bypass_has_zero_generation_calls(self):
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        query = MagicMock()
        inp = TranslationInput(
            items=[TranslationItem(region_id=1, source="Frost Chain?", reading_order=1)]
        )

        with patch.object(provider, "_query_official_translation", query), \
             patch.object(provider, "_check_health", return_value=True):
            out = provider.translate(inp)

        self.assertEqual(out.results[0].translation, "Frost Chain?")
        query.assert_not_called()
        self.assertEqual(provider.metrics.generation_call_count, 0)
        self.assertEqual(provider.metrics.term_only_bypass_count, 1)

    def test_approved_inflected_span_matching_and_protection(self):
        forms = ["Spirit Stone", "Spirit Stones", "Spirit Stone's"]
        suffixes = ["", "s", "'s"]
        for source_form, expected_suffix in zip(forms, suffixes):
            matches = find_candidate_phrase_matches("SPIRIT STONE", source_form)
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].group(0), source_form)
            self.assertEqual(matches[0].group("english_suffix") or "", expected_suffix)

        prepared, placeholder_map = protect_source_text(
            "They charged us forty Spirit Stones.",
            {"SPIRIT STONE": "Ruh Taşı"},
            set(),
        )
        self.assertNotIn("Spirit Stones", prepared)
        self.assertIn("__WTTERM0001__", prepared)
        meta = placeholder_map["__WTTERM0001__"]
        self.assertEqual(meta.source_original, "Spirit Stones")
        self.assertEqual(meta.source_suffix, "s")

    def test_approved_target_missing_and_english_leakage_require_review(self):
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        profile = SeriesProfile(
            series_id="approved_validation",
            glossary={"SPIRIT STONE": "Ruh Taşı"},
        )
        inp = TranslationInput(
            items=[TranslationItem(region_id=1, source="Use a Spirit Stone.", reading_order=1)],
            profile=profile,
        )

        with patch.object(
            provider,
            "_query_official_translation",
            return_value=("Use a Spirit Stone.", 8, 5, 0.1),
        ), patch.object(provider, "_check_health", return_value=True):
            out = provider.translate(inp)

        result = out.results[0]
        self.assertTrue(result.requires_review)
        self.assertIn("approved_term_missing", result.validation_warnings)
        self.assertIn("approved_source_term_leakage", result.validation_warnings)
        self.assertIn("untranslated_source_prose", result.validation_warnings)

    def test_untranslated_source_prose_guard_distinguishes_term_only(self):
        meta = ProtectedTermMeta(
            sentinel="__WTTERM0001__",
            source_original="Frost Chain",
            target_base="Frost Chain",
            is_approved=False,
            proper_name=True,
            source_term="Frost Chain",
        )
        p_map = {meta.sentinel: meta}
        self.assertTrue(
            has_untranslated_source_prose(
                "Activate __WTTERM0001__.",
                "Activate Frost Chain.",
                p_map,
            )
        )
        self.assertFalse(
            has_untranslated_source_prose(
                "__WTTERM0001__?",
                "Frost Chain?",
                p_map,
            )
        )

    def test_any_surviving_opaque_sentinel_is_rejected(self):
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        inp = TranslationInput(
            items=[TranslationItem(region_id=1, source="Activate Frost Chain.", reading_order=1)]
        )

        with patch.object(
            provider,
            "_query_official_translation",
            return_value=("__WTTERM9999__ etkinleştirildi.", 8, 5, 0.1),
        ), patch.object(provider, "_check_health", return_value=True):
            out = provider.translate(inp)

        result = out.results[0]
        self.assertTrue(contains_unrestored_protected_term("__WTTERM9999__"))
        self.assertIsNone(result.translation)
        self.assertTrue(result.requires_review)
        self.assertIn("unrestored_protected_term", result.validation_warnings)

    def test_protected_term_restoration_handles_turkish_inflection(self):
        plural_common = ProtectedTermMeta(
            sentinel="__WTTERM0001__",
            source_original="Spirit Stones",
            target_base="Ruh Taşı",
            is_approved=True,
            proper_name=False,
            source_term="SPIRIT STONE",
            source_suffix="s",
        )
        possessive_common = ProtectedTermMeta(
            sentinel="__WTTERM0002__",
            source_original="Mana Core's",
            target_base="Mana Çekirdeği",
            is_approved=True,
            proper_name=False,
            source_term="MANA CORE",
            source_suffix="'s",
        )
        proper_name = ProtectedTermMeta(
            sentinel="__WTTERM0003__",
            source_original="Luo Tian",
            target_base="Luo Tian",
            is_approved=True,
            proper_name=True,
            source_term="LUO TIAN",
        )
        p_map = {
            plural_common.sentinel: plural_common,
            possessive_common.sentinel: possessive_common,
            proper_name.sentinel: proper_name,
        }
        restored = restore_protected_translation(
            "__WTTERM0001__leri, __WTTERM0002__ değeri ve __WTTERM0003__'a.",
            p_map,
        )
        self.assertEqual(restored, "Ruh Taşlarını, Mana Çekirdeğinin değeri ve Luo Tian'a.")
        self.assertEqual(validate_protected_terms(restored, p_map), [])

    def test_system_ui_respects_approved_glossary(self):
        s1 = "PASSIVE SKILL ACQUIRED: ECHO SENSE"
        self.assertTrue(is_system_ui_line(s1))

        tr_unapproved = translate_system_ui_line(s1, {})
        self.assertEqual(tr_unapproved, "Kazanılan Pasif Yetenek: ECHO SENSE")

        tr_approved = translate_system_ui_line(s1, {"ECHO SENSE": "Yankı Sezgisi"})
        self.assertEqual(tr_approved, "Kazanılan Pasif Yetenek: Yankı Sezgisi")

    def test_chatbot_explanation_guard(self):
        short_src = "Because swords are useful."
        chatbot_resp = "As a large language model, I don't have a physical body..."

        self.assertTrue(is_explanation_like_output(chatbot_resp, short_src))

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

        self.assertIsNone(res1.translation)
        self.assertTrue(res1.requires_review)
        self.assertIn("translation_server_error", res1.validation_warnings)

        self.assertEqual(res2.translation, "Öge 2 çalışıyor.")
        self.assertFalse(res2.requires_review)

    def test_server_identity_verification(self):
        provider = TranslateGemmaGGUFTranslationProvider(
            model_path=r"C:\AI\Models\translategemma-12b-it-q5_k_m.gguf",
            server_url="http://127.0.0.1:8081",
        )

        props_response = json.dumps({
            "model_path": "C:\\AI\\Models\\Qwen3.5-9B-Q5_K_M.gguf"
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
            self.assertFalse(provider._check_health())


if __name__ == "__main__":
    unittest.main()
