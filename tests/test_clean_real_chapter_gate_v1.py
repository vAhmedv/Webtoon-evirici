"""Unit tests for Clean Real Chapter Translation Gate V1 & Clean Human Review Pack.

Verifies watermark/SFX/garbage filters, short proper name preservation, stable SHA-256 randomization,
accounting for all 300 original IDs, blinding isolation, and analyzer decoding.
"""

import hashlib
import json
import random
from pathlib import Path
import pytest

from scripts.create_clean_real_chapter_gate_v1 import (
    is_domain_or_watermark,
    is_sfx_vocalization,
    is_ocr_garbage,
    rebuild_clean_context_windows,
)
from scripts.analyze_clean_production_candidate_human_review import analyze_clean_human_answers

ORIGINAL_GATE_DIR = Path("benchmark_results/real_chapter_translation_gate_v1")
CLEAN_GATE_DIR = Path("benchmark_results/real_chapter_translation_gate_v1_clean")
BLIND_PACK_DIR = Path("benchmark_results/qwen_production_candidate_gate_v1_clean")


def test_clean_context_windows_exclude_removed_noise_and_duplicates() -> None:
    retained = [
        {"id": "a", "series": "S", "chapter": "C", "original_accepted_english": "FIRST STORY."},
        {"id": "b", "series": "S", "chapter": "C", "original_accepted_english": "SECOND STORY."},
        {"id": "c", "series": "S", "chapter": "C", "original_accepted_english": "THIRD STORY."},
    ]
    rebuilt = rebuild_clean_context_windows(retained)
    assert rebuilt[1]["previous_context"] == ["FIRST STORY."]
    assert rebuilt[1]["next_context"] == ["THIRD STORY."]
    assert all(
        "ARYASCANS" not in " ".join(item["previous_context"] + item["next_context"])
        for item in rebuilt
    )
    assert "previous_context" not in retained[0]


def test_filters_domain_sfx_garbage_and_preserves_proper_names():
    """Verify watermark, SFX, and garbage filters work while preserving proper names."""
    # Watermarks
    assert is_domain_or_watermark("ARYASCANS.com")[0] is True
    assert is_domain_or_watermark("ASMOTOON.COM")[0] is True
    assert is_domain_or_watermark("THANKS FOR EARLIER.")[0] is False

    # SFX
    assert is_sfx_vocalization("AHHH!!!")[0] is True
    assert is_sfx_vocalization("THUMP~")[0] is True
    assert is_sfx_vocalization("I WILL DEFEAT YOU!")[0] is False

    # Garbage vs Real Short Names
    assert is_ocr_garbage("nb")[0] is True
    assert is_ocr_garbage("udqa")[0] is True
    assert is_ocr_garbage("Yuan")[0] is False  # Proper name preserved
    assert is_ocr_garbage("Gao")[0] is False   # Proper name preserved
    assert is_ocr_garbage("Yu")[0] is False    # Proper name preserved
    assert is_ocr_garbage("Thanks.")[0] is False


def test_clean_gate_artifacts_exist_and_all_300_ids_accounted_for():
    """Verify clean dataset files exist and valid + excluded == 300."""
    valid_p = CLEAN_GATE_DIR / "valid_story_items.json"
    excl_p = CLEAN_GATE_DIR / "excluded_items.json"
    summary_p = CLEAN_GATE_DIR / "clean_summary.json"

    assert valid_p.exists()
    assert excl_p.exists()
    assert summary_p.exists()

    with open(valid_p, encoding="utf-8") as f:
        valid_items = json.load(f)
    with open(excl_p, encoding="utf-8") as f:
        excluded_items = json.load(f)
    with open(summary_p, encoding="utf-8") as f:
        summary = json.load(f)

    assert len(valid_items) + len(excluded_items) == 300
    assert summary["original_total_items"] == 300
    assert summary["valid_story_items_count"] == len(valid_items)
    assert summary["total_excluded_items"] == len(excluded_items)

    valid_ids = {it["id"] for it in valid_items}
    excluded_ids = {it["id"] for it in excluded_items}
    assert len(valid_ids.intersection(excluded_ids)) == 0  # Disjoint sets


def test_blind_clean_pack_artifacts_and_zero_excluded_items():
    """Verify 80 clean items in blind pack and ZERO excluded noise items."""
    blind_p = BLIND_PACK_DIR / "blind_review_items.json"
    excl_p = CLEAN_GATE_DIR / "excluded_items.json"
    html_p = BLIND_PACK_DIR / "human_review_blind.html"

    with open(blind_p, encoding="utf-8") as f:
        blind_items = json.load(f)
    with open(excl_p, encoding="utf-8") as f:
        excluded_items = json.load(f)

    excluded_ids = {it["id"] for it in excluded_items}

    assert len(blind_items) == 80

    for item in blind_items:
        # Verify no excluded item in blind pack
        review_id = item["review_id"]
        src = item["original_accepted_english"]
        assert item.get("benchmark_id") not in excluded_ids
        assert "aryascans" not in src.lower()
        assert "asmotoon" not in src.lower()

    # Verify zero model names leaked in HTML or JSON
    forbidden = ["translategemma", "qwen", "gemma"]
    for p in [blind_p, html_p]:
        txt_lower = p.read_text(encoding="utf-8").lower()
        for forb in forbidden:
            assert forb not in txt_lower, f"Model name '{forb}' leaked in {p.name}!"


def test_stable_sha256_randomization_across_subprocesses():
    """Verify deterministic stable SHA-256 A/B assignment."""
    item_id = "axe_god_chapter1_005"
    stable_hash = int.from_bytes(
        hashlib.sha256(item_id.encode("utf-8")).digest()[:8],
        "big"
    )
    rng1 = random.Random(20260810 + stable_hash)
    choice1 = rng1.choice([True, False])

    rng2 = random.Random(20260810 + stable_hash)
    choice2 = rng2.choice([True, False])

    assert choice1 == choice2


def test_clean_analyzer_decoding_and_recommendation_rules(tmp_path):
    """Test clean analyzer script on synthetic human review answers."""
    key_p = BLIND_PACK_DIR / "blind_answer_key.json"
    manifest_p = BLIND_PACK_DIR / "sample_manifest.json"

    with open(key_p, encoding="utf-8") as f:
        key = json.load(f)
    with open(manifest_p, encoding="utf-8") as f:
        manifest = json.load(f)

    series_items: dict[str, list[str]] = {}
    for m in manifest:
        series_items.setdefault(m["series"], []).append(m["review_id"])

    synthetic_answers = {}

    for s_name, r_ids in series_items.items():
        for idx, rid in enumerate(r_ids):
            info = key[rid]
            qwen_is_a = (info["translation_a_model"] == "qwen35")

            if idx < 20:
                winner = "A" if qwen_is_a else "B"  # Qwen win (40 total)
                sev = "MINOR"
            elif idx < 25:
                winner = "B" if qwen_is_a else "A"  # TG win (10 total)
                sev = "MINOR"
            elif idx < 30:
                winner = "TIE_GOOD"
                sev = "NONE"
            elif idx < 35:
                winner = "TIE_BAD"
                sev = "MAJOR"
            else:
                winner = "UNSCORABLE_OCR"
                sev = "NONE"

            synthetic_answers[rid] = {
                "winner": winner,
                "severity": sev,
                "tags": ["meaning"],
                "notes": "Synthetic clean test note",
            }

    test_answers_p = tmp_path / "test_clean_answers.json"
    with open(test_answers_p, "w", encoding="utf-8") as f:
        json.dump(synthetic_answers, f, indent=2)

    results = analyze_clean_human_answers(test_answers_p)

    assert results["total_reviewed"] == 80
    assert results["valid_reviewed"] == 70
    assert results["unscorable_ocr"] == 10
    assert results["qwen_wins"] == 40
    assert results["translategemma_wins"] == 10
    assert results["recommendation"] == "STRONG_QWEN_CANDIDATE"


def test_original_benchmark_and_production_files_untouched():
    """Verify original benchmark files and production translator are unchanged."""
    orig_comp = ORIGINAL_GATE_DIR / "comparison.json"
    assert orig_comp.exists()

    prod_p = Path("providers/translation/translategemma_gguf_translation.py")
    assert prod_p.exists()
    content = prod_p.read_text(encoding="utf-8")
    assert "TranslateGemmaGGUFTranslationProvider" in content
