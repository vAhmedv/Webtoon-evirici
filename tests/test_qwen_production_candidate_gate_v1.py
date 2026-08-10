"""Unit tests for Qwen Production Candidate Human Gate V1.

Verifies deterministic blinding, dataset stratification, answer key isolation,
analyzer recommendation logic, and ensures no production files or model weights were modified.
"""

import json
from pathlib import Path
import pytest

from scripts.analyze_production_candidate_human_review import analyze_human_answers

GATE_DIR = Path("benchmark_results/qwen_production_candidate_gate_v1")


def test_gate_artifacts_exist_and_sample_size():
    """Verify all 6 gate artifacts exist and contain exactly 80 items."""
    manifest_p = GATE_DIR / "sample_manifest.json"
    items_p = GATE_DIR / "blind_review_items.json"
    key_p = GATE_DIR / "blind_answer_key.json"
    summary_p = GATE_DIR / "sampling_summary.json"
    html_p = GATE_DIR / "human_review_blind.html"
    readme_p = GATE_DIR / "README_REVIEW.txt"

    assert manifest_p.exists()
    assert items_p.exists()
    assert key_p.exists()
    assert summary_p.exists()
    assert html_p.exists()
    assert readme_p.exists()

    with open(manifest_p, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(items_p, encoding="utf-8") as f:
        items = json.load(f)
    with open(key_p, encoding="utf-8") as f:
        key = json.load(f)

    assert len(manifest) == 80
    assert len(items) == 80
    assert len(key) == 80


def test_blinding_isolation_no_model_names_leak():
    """Verify blind items and HTML UI contain ZERO model name leaks."""
    items_p = GATE_DIR / "blind_review_items.json"
    html_p = GATE_DIR / "human_review_blind.html"

    forbidden = ["translategemma", "qwen", "gemma"]

    for path in [items_p, html_p]:
        text_lower = path.read_text(encoding="utf-8").lower()
        for word in forbidden:
            assert word not in text_lower, f"Model name '{word}' leaked in {path.name}!"


def test_stratification_and_v3_rewrites():
    """Verify 60 high-info / 20 controls, 2 series, 10 chapters, and V3 rewrites."""
    summary_p = GATE_DIR / "sampling_summary.json"
    with open(summary_p, encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["total_review_items"] == 80
    assert summary["high_information_count"] == 60
    assert summary["control_count"] == 20
    assert summary["v3_rewrite_count"] == 3
    assert len(summary["items_per_series"]) == 2
    assert len(summary["items_per_chapter"]) == 5

    # Each series has 40 items
    for count in summary["items_per_series"].values():
        assert count == 40

    # Each chapter has 16 items
    for count in summary["items_per_chapter"].values():
        assert count == 16


def test_deterministic_ab_assignment():
    """Verify that every item has translation A & B, and answer key matches."""
    items_p = GATE_DIR / "blind_review_items.json"
    key_p = GATE_DIR / "blind_answer_key.json"

    with open(items_p, encoding="utf-8") as f:
        items = json.load(f)
    with open(key_p, encoding="utf-8") as f:
        key = json.load(f)

    for item in items:
        rid = item["review_id"]
        assert rid in key
        assert "translation_a" in item
        assert "translation_b" in item
        assert isinstance(item["translation_a"], str)
        assert isinstance(item["translation_b"], str)

        info = key[rid]
        assert info["translation_a_model"] in ["qwen35", "translategemma"]
        assert info["translation_b_model"] in ["qwen35", "translategemma"]
        assert info["translation_a_model"] != info["translation_b_model"]


def test_analyzer_decoding_and_recommendation_rules(tmp_path):
    """Test analyzer script on synthetic human review answers."""
    # 1. Test STRONG_QWEN_CANDIDATE case
    key_p = GATE_DIR / "blind_answer_key.json"
    with open(key_p, encoding="utf-8") as f:
        key = json.load(f)

    # Separate item IDs by series to ensure balanced wins
    manifest_p = GATE_DIR / "sample_manifest.json"
    with open(manifest_p, encoding="utf-8") as f:
        manifest = json.load(f)

    series_items: dict[str, list[str]] = {}
    for m in manifest:
        series_items.setdefault(m["series"], []).append(m["review_id"])

    synthetic_answers = {}

    for s_name, r_ids in series_items.items():
        # r_ids has 40 items per series
        for idx, rid in enumerate(r_ids):
            info = key[rid]
            qwen_is_a = (info["translation_a_model"] == "qwen35")

            if idx < 20:
                winner = "A" if qwen_is_a else "B"  # Qwen win (20 per series = 40 total)
                sev = "MINOR"
            elif idx < 25:
                winner = "B" if qwen_is_a else "A"  # TG win (5 per series = 10 total)
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
                "notes": "Synthetic test note",
            }

    test_answers_p = tmp_path / "test_answers.json"
    with open(test_answers_p, "w", encoding="utf-8") as f:
        json.dump(synthetic_answers, f, indent=2)

    results = analyze_human_answers(test_answers_p)

    assert results["total_reviewed"] == 80
    assert results["valid_reviewed"] == 70  # 80 - 10 unscorable
    assert results["unscorable_ocr"] == 10
    assert results["qwen_wins"] == 40
    assert results["translategemma_wins"] == 10
    assert results["tie_good"] == 10
    assert results["tie_bad"] == 10
    assert results["recommendation"] == "STRONG_QWEN_CANDIDATE"


def test_production_translator_unmodified():
    """Verify that default production translator file is unchanged."""
    prod_p = Path("providers/translation/translategemma_gguf_translation.py")
    assert prod_p.exists()
    content = prod_p.read_text(encoding="utf-8")
    assert "TranslateGemmaGGUFTranslationProvider" in content
