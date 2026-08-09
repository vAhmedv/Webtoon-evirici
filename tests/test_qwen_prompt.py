"""Unit tests for Qwen translation prompt invariants and assembly rules.
"""
import json
import unittest

from core.translation.profile_discovery import CandidateStore
from core.translation.series_profile import SeriesProfile
from providers.translation.base import TranslationInput, TranslationItem
from providers.translation.qwen_prompt import (
    _SYSTEM_PROMPT,
    build_qwen_translation_prompt,
)


class TestQwenPromptInvariants(unittest.TestCase):

    def test_system_prompt_no_invention_rule(self):
        self.assertIn("Never invent information", _SYSTEM_PROMPT)
        self.assertIn("Do NOT invent or intensify:", _SYSTEM_PROMPT)

    def test_system_prompt_neutral_address_protection(self):
        self.assertIn(
            "A neutral form of address must not become an insult or affectionate nickname",
            _SYSTEM_PROMPT,
        )

    def test_context_reference_only(self):
        inp = TranslationInput(
            items=[TranslationItem(region_id=1, source="Hello.", reading_order=1)],
            context_items=[TranslationItem(region_id=0, source="Hi there.", reading_order=0)],
        )
        prompt_str, _ = build_qwen_translation_prompt(inp)
        self.assertIn("CONTEXT ONLY (REFERENCE ONLY - DO NOT TRANSLATE", prompt_str)
        self.assertIn("[0] id=0 | Hi there.", prompt_str)

    def test_approved_and_provisional_terminology_distinguished(self):
        profile = SeriesProfile(
            series_id="test_series",
            known_names={"LUO TIAN": "Luo Tian"},
            glossary={"SECRET REALM": "Gizli Diyar"},
        )
        store = CandidateStore(series_id="test_series")

        inp = TranslationInput(
            items=[
                TranslationItem(
                    region_id=1,
                    source="LUO TIAN entering the SECRET REALM.",
                    reading_order=1,
                )
            ],
            profile=profile,
            candidate_store=store,
        )

        prompt_str, term_maps = build_qwen_translation_prompt(inp)
        self.assertIn("APPROVED TERMS (AUTHORITATIVE GUIDANCE", prompt_str)
        self.assertIn("- LUO TIAN => Luo Tian", prompt_str)
        self.assertIn("- SECRET REALM => Gizli Diyar", prompt_str)

    def test_ids_and_schema_present(self):
        inp = TranslationInput(
            items=[TranslationItem(region_id=42, source="Where are we?", reading_order=1)]
        )
        prompt_str, _ = build_qwen_translation_prompt(inp)
        self.assertIn("ITEMS TO TRANSLATE:", prompt_str)
        self.assertIn("[1] id=42 | Where are we?", prompt_str)
        self.assertIn("OUTPUT SCHEMA:", prompt_str)
        self.assertIn('"translations":', prompt_str)
        self.assertIn('"term_usages":', prompt_str)
        self.assertIn('"fidelity_flags":', prompt_str)

    def test_no_hardcoded_series_names_in_production_prompt(self):
        lower_prompt = _SYSTEM_PROMPT.lower()
        self.assertNotIn("koharu", lower_prompt)
        self.assertNotIn("luo tian", lower_prompt)
        self.assertNotIn("suwayomi", lower_prompt)

    def test_candidate_store_remains_unmodified_by_prompt_builder(self):
        store = CandidateStore(series_id="test_series")
        store_before_json = json.dumps(store.to_dict())

        inp = TranslationInput(
            items=[TranslationItem(region_id=1, source="Test source item.", reading_order=1)],
            candidate_store=store,
        )
        build_qwen_translation_prompt(inp)

        store_after_json = json.dumps(store.to_dict())
        self.assertEqual(store_before_json, store_after_json, "CandidateStore was mutated by prompt builder!")


if __name__ == "__main__":
    unittest.main()
