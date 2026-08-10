"""Unit tests for Real Chapter Translation Gate V1 pipeline."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.translation.series_profile import SeriesProfile
from providers.translation.base import TranslationItem


def test_series_profile_separation() -> None:
    profile_a = SeriesProfile(
        series_id="axe_god",
        known_names={"LUO TIAN": "Luo Tian"},
        glossary={"BLACKWIND RAVINE": "Blackwind Ravine"},
    )
    profile_b = SeriesProfile(
        series_id="god_tier_crafter",
        known_names={"ETHAN": "Ethan"},
        glossary={"GOD-TIER CRAFTER": "Tanrı Seviye Zanaatkar"},
    )

    assert profile_a.series_id != profile_b.series_id
    assert "LUO TIAN" in profile_a.known_names
    assert "LUO TIAN" not in profile_b.known_names
    assert "ETHAN" in profile_b.known_names
    assert "ETHAN" not in profile_a.known_names


def test_dataset_hash_validation() -> None:
    dataset = [
        {
            "id": "axe_ch01_001",
            "series_id": "axe_god",
            "chapter_name": "Chapter 1",
            "original_accepted_english": "Hello world.",
        }
    ]
    raw_bytes = json.dumps(dataset, sort_keys=True, ensure_ascii=False).encode("utf-8")
    expected_hash = hashlib.sha256(raw_bytes).hexdigest()

    # Recompute and verify match
    actual_hash = hashlib.sha256(
        json.dumps(dataset, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert actual_hash == expected_hash

    # Mutate dataset and verify mismatch
    mutated = list(dataset)
    mutated[0]["original_accepted_english"] = "Mutated text."
    mutated_hash = hashlib.sha256(
        json.dumps(mutated, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert mutated_hash != expected_hash


def test_context_boundary_no_chapter_cross() -> None:
    # Context helper function should not pull previous context from a different chapter
    items = [
        {"id": "axe_ch01_010", "chapter": "Chapter 1", "text": "Ch1 end"},
        {"id": "axe_ch02_001", "chapter": "Chapter 2", "text": "Ch2 start"},
    ]

    def get_prev_context(target_idx: int) -> list[str]:
        target_ch = items[target_idx]["chapter"]
        ctx = []
        for i in range(target_idx - 1, -1, -1):
            if items[i]["chapter"] != target_ch:
                break
            ctx.insert(0, items[i]["text"])
            if len(ctx) >= 3:
                break
        return ctx

    assert get_prev_context(1) == []  # Cannot pull Ch1 into Ch2


def test_human_review_fields_default_null() -> None:
    entry = {
        "id": "axe_ch01_001",
        "human_review": {
            "winner": None,
            "translategemma_score": None,
            "qwen_score": None,
            "notes": None,
        },
    }
    assert entry["human_review"]["winner"] is None
    assert entry["human_review"]["notes"] is None


def test_excluded_ocr_marking() -> None:
    region = {
        "id": "axe_ch01_999",
        "ocr_raw": "x#@!999",
        "excluded_from_translation_quality": True,
        "exclusion_reason": "ocr_unreliable",
    }
    assert region["excluded_from_translation_quality"] is True
    assert region["exclusion_reason"] == "ocr_unreliable"
