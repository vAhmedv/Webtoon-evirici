"""Qwen3.5-9B GGUF Direct Translator Sanity Test (ID 011 V3 Source).

Executes one real Qwen GGUF translation request on port 8083.
"""
import time

from providers.translation.base import TranslationInput, TranslationItem
from providers.translation.qwen_gguf_translation_v2 import QwenGGUFTranslationProviderV2

# ID 011 V3 Selected English Source
ID_011_V3_SOURCE = "Frost Chain can keep three targets from moving at the same time."


def run_sanity():
    print("=== STARTING QWEN TRANSLATOR V2 SANITY TEST (ID 011) ===")

    provider = QwenGGUFTranslationProviderV2(server_url="http://127.0.0.1:8083")

    t0 = time.perf_counter()
    print("Loading Qwen GGUF translator on port 8083...")
    provider.load()
    load_sec = time.perf_counter() - t0
    print(f"Loaded in {load_sec:.2f}s")

    from core.translation.series_profile import SeriesProfile

    profile = SeriesProfile(
        series_id="shootout_v1",
        glossary={"FROST CHAIN": "Frost Chain"},
    )

    inp = TranslationInput(
        items=[
            TranslationItem(
                region_id=11,
                source=ID_011_V3_SOURCE,
                reading_order=11,
            )
        ],
        profile=profile,
    )

    print(f"\nSOURCE: {ID_011_V3_SOURCE}")
    output = provider.translate(inp)

    provider.unload()

    res = output.results[0]
    print("\n--- TRANSLATION RESULT ---")
    print(f"Translation: {res.translation}")
    print(f"Raw Response: {res.raw_model_response}")
    print(f"Validation Warnings: {res.validation_warnings}")
    print(f"Requires Review: {res.requires_review}")

    # Sanity Assertions
    assert res.translation is not None, "Translation is None"
    assert "Frost Chain" in res.translation or "frost chain" in res.translation.lower(), "Frost Chain lost"
    assert "üç" in res.translation.lower() or "3" in res.translation, "Number 3 lost"
    assert res.requires_review is False or "chatbot_or_explanation_output" not in res.validation_warnings

    print("\n=== SANITY TEST PASSED ===")


if __name__ == "__main__":
    run_sanity()
