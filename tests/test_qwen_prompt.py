"""Unit tests for Qwen translation prompt invariants and assembly rules (Compact v3).
"""
import json
import unittest

from core.translation.profile_discovery import CandidateStore
from core.translation.series_profile import SeriesProfile
from providers.translation.base import TranslationInput, TranslationItem
from providers.translation.qwen_prompt import (
    _SYSTEM_PROMPT,
    build_qwen_translation_prompt,
    build_qwen_translation_user_prompt,
)


class TestQwenPromptInvariants(unittest.TestCase):

    def test_system_prompt_length_reduction(self):
        # Compact v3 prompt must be materially shorter than v2 (under 4000 characters and under 80 lines)
        self.assertLess(len(_SYSTEM_PROMPT), 4000, "System prompt v3 is too long!")
        self.assertLess(len(_SYSTEM_PROMPT.splitlines()), 80, "System prompt v3 has too many lines!")

    def test_system_prompt_no_invention_rule(self):
        self.assertIn("Never invent information", _SYSTEM_PROMPT)

    def test_system_prompt_v3_principles(self):
        # Core translation concept
        self.assertIn("translate what the English means in context", _SYSTEM_PROMPT)
        self.assertIn("professional Turkish localization editor", _SYSTEM_PROMPT)

        # 1. Idiomatic & Contextual
        self.assertIn("Idiomatic Meaning & Context:", _SYSTEM_PROMPT)

        # 2. Natural Turkish Syntax & Semantic Restraint
        self.assertIn("Natural Turkish Syntax:", _SYSTEM_PROMPT)
        self.assertIn("Semantic Restraint & State vs. Action:", _SYSTEM_PROMPT)

        # 3. Fidelity & Intensity Control
        self.assertIn("Fidelity & Intensity Control:", _SYSTEM_PROMPT)

        # 4. Formality / Register
        self.assertIn("Register & Formality (Sen / Siz):", _SYSTEM_PROMPT)

        # 5. Negation, Quantifiers & Logical Scope
        self.assertIn("Negation, Quantifiers & Logical Scope:", _SYSTEM_PROMPT)

        # 6. Names, Glossary & Morphology
        self.assertIn("Names, Glossary & Turkish Morphology:", _SYSTEM_PROMPT)

        # 7. Narration & Concision
        self.assertIn("Narration:", _SYSTEM_PROMPT)
        self.assertIn("Concision & Context:", _SYSTEM_PROMPT)

    def test_context_reference_only(self):
        inp = TranslationInput(
            items=[TranslationItem(region_id=1, source="Hello.", reading_order=1)],
            context_items=[TranslationItem(region_id=0, source="Hi there.", reading_order=0)],
        )
        user_prompt, _ = build_qwen_translation_user_prompt(inp)
        self.assertIn("CONTEXT ONLY (REFERENCE ONLY - DO NOT TRANSLATE", user_prompt)
        self.assertIn("[0] id=0 | Hi there.", user_prompt)

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

        user_prompt, term_maps = build_qwen_translation_user_prompt(inp)
        self.assertIn("APPROVED TERMS (AUTHORITATIVE GUIDANCE", user_prompt)
        self.assertIn("- LUO TIAN => Luo Tian", user_prompt)
        self.assertIn("- SECRET REALM => Gizli Diyar", user_prompt)

    def test_unrelated_glossary_terms_filtered(self):
        profile = SeriesProfile(
            series_id="test_series",
            known_names={"LUO TIAN": "Luo Tian", "HU SAN": "Hu San"},
            glossary={"SECRET REALM": "Gizli Diyar", "MANA CORE": "Mana Çekirdeği"},
        )
        inp = TranslationInput(
            items=[
                TranslationItem(
                    region_id=1,
                    source="We entered the SECRET REALM.",
                    reading_order=1,
                )
            ],
            profile=profile,
        )
        user_prompt, _ = build_qwen_translation_user_prompt(inp)
        self.assertIn("SECRET REALM => Gizli Diyar", user_prompt)
        self.assertNotIn("MANA CORE", user_prompt)
        self.assertNotIn("HU SAN", user_prompt)

    def test_ids_and_schema_present(self):
        inp = TranslationInput(
            items=[TranslationItem(region_id=42, source="Where are we?", reading_order=1)]
        )
        user_prompt, _ = build_qwen_translation_user_prompt(inp)
        self.assertIn("ITEMS TO TRANSLATE:", user_prompt)
        self.assertIn("[1] id=42 | Where are we?", user_prompt)
        self.assertIn("OUTPUT SCHEMA:", user_prompt)
        self.assertIn('"translations":', user_prompt)
        self.assertIn('"term_usages":', user_prompt)
        self.assertIn('"fidelity_flags":', user_prompt)

    def test_no_hardcoded_series_names_in_production_prompt(self):
        lower_prompt = _SYSTEM_PROMPT.lower()
        self.assertNotIn("koharu", lower_prompt)
        self.assertNotIn("suwayomi", lower_prompt)

    def test_candidate_store_remains_unmodified_by_prompt_builder(self):
        store = CandidateStore(series_id="test_series")
        store_before_json = json.dumps(store.to_dict())

        inp = TranslationInput(
            items=[TranslationItem(region_id=1, source="Test source item.", reading_order=1)],
            candidate_store=store,
        )
        build_qwen_translation_user_prompt(inp)

        store_after_json = json.dumps(store.to_dict())
        self.assertEqual(store_before_json, store_after_json, "CandidateStore was mutated by prompt builder!")


if __name__ == "__main__":
    unittest.main()
