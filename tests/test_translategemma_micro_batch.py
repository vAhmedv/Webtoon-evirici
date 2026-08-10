"""Focused regression tests for TranslateGemma contextual micro-batching."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from core.translation.protection import (
    protect_source_text,
    restore_protected_translation,
)
from core.translation.series_profile import SeriesProfile
from providers.translation.base import TranslationInput, TranslationItem
from providers.translation.translategemma_gguf_translation import (
    TranslateGemmaGGUFTranslationProvider,
    is_source_translation_wrapper_output,
)


def _items(sources: list[str], region_ids: list[int] | None = None) -> list[TranslationItem]:
    ids = region_ids or list(range(1, len(sources) + 1))
    return [
        TranslationItem(region_id=region_id, source=source, reading_order=index)
        for index, (region_id, source) in enumerate(zip(ids, sources), start=1)
    ]


def _response(text: str) -> tuple[str, int, int, float]:
    return text, 10, 5, 0.1


class TestTranslateGemmaMicroBatch(unittest.TestCase):
    def test_explicit_single_item_mode_disables_micro_batching(self):
        provider = TranslateGemmaGGUFTranslationProvider(
            managed=False,
            micro_batch_enabled=False,
        )
        query = MagicMock(
            side_effect=[
                _response("Birinci."),
                _response("İkinci."),
                _response("Üçüncü."),
                _response("Dördüncü."),
            ]
        )
        inp = TranslationInput(
            items=_items(
                [
                    "First ordinary line.",
                    "Second ordinary line.",
                    "Third ordinary line.",
                    "Fourth ordinary line.",
                ]
            )
        )

        with patch.object(provider, "_query_official_translation", query), patch.object(
            provider, "_check_health", return_value=True
        ):
            out = provider.translate(inp)

        self.assertEqual(query.call_count, 4)
        self.assertEqual(provider.metrics.generation_call_count, 4)
        self.assertEqual(provider.metrics.micro_batch_requests, 0)
        self.assertEqual(provider.micro_batch_history, [])
        self.assertEqual(
            [result.translation for result in out.results],
            ["Birinci.", "İkinci.", "Üçüncü.", "Dördüncü."],
        )
        for result in out.results:
            self.assertIsNone(result.micro_batch_id)
            self.assertEqual(result.micro_batch_region_ids, [])

    def test_four_ordinary_items_use_one_request_and_map_in_reading_order(self):
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        query = MagicMock(
            return_value=_response(
                "__WTSEG0001__ Birinci.\n"
                "__WTSEG0002__ İkinci.\n"
                "__WTSEG0003__ Üçüncü.\n"
                "__WTSEG0004__ Dördüncü."
            )
        )
        region_ids = [41, 7, 99, 3]
        inp = TranslationInput(
            items=_items(
                [
                    "First ordinary line.",
                    "Second ordinary line.",
                    "Third ordinary line.",
                    "Fourth ordinary line.",
                ],
                region_ids,
            )
        )

        with patch.object(provider, "_query_official_translation", query), patch.object(
            provider, "_check_health", return_value=True
        ):
            out = provider.translate(inp)

        self.assertEqual(query.call_count, 1)
        request_text = query.call_args.args[0]
        self.assertEqual(request_text.count("__WTSEG"), 4)
        self.assertNotIn("Context:", request_text)
        self.assertNotIn("Translate these:", request_text)
        self.assertEqual([result.region_id for result in out.results], region_ids)
        self.assertEqual(
            [result.translation for result in out.results],
            ["Birinci.", "İkinci.", "Üçüncü.", "Dördüncü."],
        )
        self.assertEqual(provider.metrics.generation_call_count, 1)
        self.assertEqual(provider.metrics.micro_batch_requests, 1)
        self.assertEqual(provider.metrics.micro_batch_successes, 1)
        for result in out.results:
            self.assertEqual(result.micro_batch_id, "micro_batch_0001")
            self.assertEqual(result.micro_batch_region_ids, region_ids)

    def test_invalid_segment_structures_fallback_only_failed_block(self):
        invalid_outputs = {
            "missing": (
                "__WTSEG0001__ Bir.\n__WTSEG0002__ İki.\n__WTSEG0004__ Dört."
            ),
            "duplicate": (
                "__WTSEG0001__ Bir.\n__WTSEG0002__ İki.\n"
                "__WTSEG0002__ Yine.\n__WTSEG0004__ Dört."
            ),
            "reordered": (
                "__WTSEG0002__ İki.\n__WTSEG0001__ Bir.\n"
                "__WTSEG0003__ Üç.\n__WTSEG0004__ Dört."
            ),
            "unknown": (
                "__WTSEG0001__ Bir.\n__WTSEG0002__ İki.\n"
                "__WTSEG0003__ Üç.\n__WTSEG0005__ Beş."
            ),
        }
        sources = ["First line.", "Second line.", "Third line.", "Fourth line."]
        singles = ["Birinci.", "İkinci.", "Üçüncü.", "Dördüncü."]

        for failure_name, invalid_output in invalid_outputs.items():
            with self.subTest(failure_name=failure_name):
                provider = TranslateGemmaGGUFTranslationProvider(managed=False)
                query = MagicMock(
                    side_effect=[_response(invalid_output)]
                    + [_response(single) for single in singles]
                )
                with patch.object(provider, "_query_official_translation", query), patch.object(
                    provider, "_check_health", return_value=True
                ):
                    out = provider.translate(TranslationInput(items=_items(sources)))

                self.assertEqual(query.call_count, 5)
                self.assertEqual([result.translation for result in out.results], singles)
                self.assertEqual(provider.metrics.micro_batch_requests, 1)
                self.assertEqual(provider.metrics.micro_batch_successes, 0)
                self.assertEqual(provider.metrics.micro_batch_fallbacks, 1)
                self.assertEqual(provider.metrics.single_item_fallback_calls, 4)
                self.assertEqual(provider.metrics.generation_call_count, 5)
                self.assertEqual(provider.micro_batch_history[0]["status"], "FALLBACK")

    def test_segment_marker_leak_is_rejected(self):
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        with patch.object(
            provider,
            "_query_official_translation",
            return_value=_response("__WTSEG9999__ Çevrilmiş metin."),
        ), patch.object(provider, "_check_health", return_value=True):
            result = provider.translate(
                TranslationInput(items=_items(["Translate this ordinary line."]))
            ).results[0]

        self.assertIsNone(result.translation)
        self.assertTrue(result.requires_review)
        self.assertIn("segment_marker_leak", result.validation_warnings)

    def test_term_maps_and_approved_terms_remain_item_scoped(self):
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        profile = SeriesProfile(
            series_id="item_scope",
            glossary={
                "SPIRIT STONE": "Ruh Taşı",
                "GUILD MASTER": "Lonca Lideri",
            },
        )
        query = MagicMock(
            return_value=_response(
                "__WTSEG0001__ Bir __WTTERM0001__ kullan.\n"
                "__WTSEG0002__ __WTTERM0001__ geldi.\n"
                "__WTSEG0003__ Kapıyı kapat.\n"
                "__WTSEG0004__ Burada bekle."
            )
        )
        inp = TranslationInput(
            items=_items(
                [
                    "Use a Spirit Stone.",
                    "The Guild Master arrived.",
                    "Close the door.",
                    "Wait here.",
                ]
            ),
            profile=profile,
        )

        with patch.object(provider, "_query_official_translation", query), patch.object(
            provider, "_check_health", return_value=True
        ):
            results = provider.translate(inp).results

        request_text = query.call_args.args[0]
        self.assertEqual(request_text.count("__WTTERM0001__"), 2)
        self.assertNotIn("__WTTERM0002__", request_text)
        self.assertEqual(results[0].translation, "Bir Ruh Taşı kullan.")
        self.assertEqual(results[1].translation, "Lonca Lideri geldi.")
        self.assertEqual(results[0].validation_warnings, [])
        self.assertEqual(results[1].validation_warnings, [])

    def test_system_ui_flushes_and_next_items_start_fresh_batch(self):
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        query = MagicMock(
            side_effect=[
                _response(
                    "__WTSEG0001__ Bir.\n__WTSEG0002__ İki.\n__WTSEG0003__ Üç."
                ),
                _response(
                    "__WTSEG0001__ Dört.\n__WTSEG0002__ Beş.\n"
                    "__WTSEG0003__ Altı.\n__WTSEG0004__ Yedi."
                ),
            ]
        )
        inp = TranslationInput(
            items=_items(
                [
                    "Ordinary line one.",
                    "Ordinary line two.",
                    "Ordinary line three.",
                    "CLASS ADVANCEMENT AVAILABLE",
                    "Ordinary line four.",
                    "Ordinary line five.",
                    "Ordinary line six.",
                    "Ordinary line seven.",
                ]
            )
        )

        with patch.object(provider, "_query_official_translation", query), patch.object(
            provider, "_check_health", return_value=True
        ):
            out = provider.translate(inp)

        self.assertEqual(query.call_count, 2)
        self.assertNotIn("CLASS ADVANCEMENT", query.call_args_list[0].args[0])
        self.assertNotIn("CLASS ADVANCEMENT", query.call_args_list[1].args[0])
        bypass = out.results[3]
        self.assertEqual(bypass.translation, "Sınıf Gelişimi Mevcut")
        self.assertIsNone(bypass.micro_batch_id)
        self.assertEqual(bypass.micro_batch_region_ids, [])
        self.assertEqual(provider.metrics.system_ui_bypass_count, 1)
        self.assertEqual(
            [entry["region_ids"] for entry in provider.micro_batch_history],
            [[1, 2, 3], [5, 6, 7, 8]],
        )

    def test_term_only_bypass_never_enters_model_and_flushes_both_sides(self):
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        query = MagicMock(
            side_effect=[
                _response(
                    "__WTSEG0001__ Bir.\n__WTSEG0002__ İki.\n__WTSEG0003__ Üç."
                ),
                _response(
                    "__WTSEG0001__ Dört.\n__WTSEG0002__ Beş.\n__WTSEG0003__ Altı."
                ),
            ]
        )
        inp = TranslationInput(
            items=_items(
                [
                    "Ordinary line one.",
                    "Ordinary line two.",
                    "Ordinary line three.",
                    "Silent Chain?",
                    "Ordinary line four.",
                    "Ordinary line five.",
                    "Ordinary line six.",
                ]
            )
        )

        with patch.object(provider, "_query_official_translation", query), patch.object(
            provider, "_check_health", return_value=True
        ):
            out = provider.translate(inp)

        self.assertEqual(query.call_count, 2)
        for call in query.call_args_list:
            self.assertNotIn("Silent Chain?", call.args[0])
        bypass = out.results[3]
        self.assertEqual(bypass.translation, "Silent Chain?")
        self.assertIsNone(bypass.micro_batch_id)
        self.assertEqual(bypass.micro_batch_region_ids, [])
        self.assertEqual(provider.metrics.term_only_bypass_count, 1)
        self.assertEqual(
            [entry["region_ids"] for entry in provider.micro_batch_history],
            [[1, 2, 3], [5, 6, 7]],
        )

    def test_source_translation_wrapper_labels_are_detected(self):
        prepared = "They want twenty bottles for one box."
        labels = [
            "Çevirisi:",
            "Türkçe Çeviri:",
            "Translation:",
            "Turkish Translation:",
            "**Çevirisi:**",
        ]
        for label in labels:
            with self.subTest(label=label):
                output = f"{prepared}\n\n{label}\nBir kutu için yirmi şişe istiyorlar."
                self.assertTrue(is_source_translation_wrapper_output(output, prepared))

    def test_source_translation_wrapper_is_rejected_not_salvaged(self):
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        raw = (
            "They want twenty bottles for one box.\n\n"
            "**Çevirisi:**\nBir kutu için yirmi şişe istiyorlar."
        )
        with patch.object(
            provider, "_query_official_translation", return_value=_response(raw)
        ), patch.object(provider, "_check_health", return_value=True):
            result = provider.translate(
                TranslationInput(items=_items(["They want twenty bottles for one box."]))
            ).results[0]

        self.assertIsNone(result.translation)
        self.assertTrue(result.requires_review)
        self.assertIn("source_translation_wrapper", result.validation_warnings)

    def test_chatbot_explanation_output_is_still_rejected(self):
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        with patch.object(
            provider,
            "_query_official_translation",
            return_value=_response("As a large language model, I cannot help with that."),
        ), patch.object(provider, "_check_health", return_value=True):
            result = provider.translate(
                TranslationInput(items=_items(["Open the door."]))
            ).results[0]

        self.assertIsNone(result.translation)
        self.assertIn("chatbot_or_explanation_output", result.validation_warnings)

    def test_untranslated_ordinary_english_without_terms_is_flagged(self):
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        source = "Open the western gate now."
        with patch.object(
            provider, "_query_official_translation", return_value=_response(source)
        ), patch.object(provider, "_check_health", return_value=True):
            result = provider.translate(
                TranslationInput(items=_items([source]))
            ).results[0]

        self.assertEqual(result.translation, source)
        self.assertTrue(result.requires_review)
        self.assertIn("untranslated_source_prose", result.validation_warnings)

    def test_protected_english_ability_name_is_not_prose_leakage(self):
        provider = TranslateGemmaGGUFTranslationProvider(managed=False)
        with patch.object(
            provider,
            "_query_official_translation",
            return_value=_response("__WTTERM0001__'i etkinleştir."),
        ), patch.object(provider, "_check_health", return_value=True):
            result = provider.translate(
                TranslationInput(items=_items(["Activate Silent Chain."]))
            ).results[0]

        self.assertEqual(result.translation, "Silent Chain'i etkinleştir.")
        self.assertNotIn("untranslated_source_prose", result.validation_warnings)

    def test_cardinal_common_term_plural_is_safely_removed(self):
        cases = [
            (
                "They paid thirty Spirit Stones.",
                {"SPIRIT STONE": "Ruh Taşı"},
                "Otuz __WTTERM0001__ler ödediler.",
                "Otuz Ruh Taşı ödediler.",
            ),
            (
                "They found twenty Mana Cores.",
                {"MANA CORE": "Mana Çekirdeği"},
                "Yirmi __WTTERM0001__ler buldular.",
                "Yirmi Mana Çekirdeği buldular.",
            ),
        ]
        for source, glossary, model_output, expected in cases:
            with self.subTest(source=source):
                _, placeholder_map = protect_source_text(source, glossary, set())
                meta = placeholder_map["__WTTERM0001__"]
                self.assertTrue(meta.source_cardinal_quantified)
                self.assertEqual(
                    restore_protected_translation(model_output, placeholder_map),
                    expected,
                )

    def test_non_cardinal_plural_is_not_singularized(self):
        _, placeholder_map = protect_source_text(
            "The Spirit Stones were missing.",
            {"SPIRIT STONE": "Ruh Taşı"},
            set(),
        )
        meta = placeholder_map["__WTTERM0001__"]
        self.assertFalse(meta.source_cardinal_quantified)
        self.assertEqual(
            restore_protected_translation("__WTTERM0001__ kayıptı.", placeholder_map),
            "Ruh Taşları kayıptı.",
        )

    def test_proper_names_are_not_subject_to_cardinal_common_term_repair(self):
        _, placeholder_map = protect_source_text(
            "Three Kael Ardens arrived.",
            {"KAEL ARDEN": "Kael Arden"},
            set(),
            proper_name_terms={"KAEL ARDEN"},
        )
        meta = placeholder_map["__WTTERM0001__"]
        self.assertTrue(meta.proper_name)
        self.assertFalse(meta.source_cardinal_quantified)
        self.assertEqual(
            restore_protected_translation("Üç __WTTERM0001__ geldi.", placeholder_map),
            "Üç Kael Ardenler geldi.",
        )

    def test_micro_batch_size_accepts_only_three_or_four(self):
        production_default = TranslateGemmaGGUFTranslationProvider(managed=False)
        self.assertTrue(production_default.micro_batch_enabled)
        self.assertEqual(production_default.prompt_variant, "legacy")
        self.assertEqual(
            TranslateGemmaGGUFTranslationProvider(managed=False, micro_batch_size=3).micro_batch_size,
            3,
        )
        self.assertEqual(
            TranslateGemmaGGUFTranslationProvider(managed=False, micro_batch_size=4).micro_batch_size,
            4,
        )
        for invalid_size in (0, 2, 5):
            with self.subTest(invalid_size=invalid_size), self.assertRaises(ValueError):
                TranslateGemmaGGUFTranslationProvider(
                    managed=False,
                    micro_batch_size=invalid_size,
                )


if __name__ == "__main__":
    unittest.main()
