"""Tests for Gated Verifier (conditional OCR verification) routing logic."""

import pytest
from core.detection import BBox, Region, RegionStatus, RegionType
from providers.ocr.agreement import should_run_verifier
from providers.ocr.base import OCRResult


def test_should_run_verifier_clean_high_confidence():
    primary = OCRResult(
        text="I must forge the legendary sword at once.",
        confidence=0.96,
        raw_text="I must forge the legendary sword at once.",
    )
    should_run, reason = should_run_verifier(primary, region=None, min_confidence=0.92)
    assert should_run is False
    assert reason == "high_confidence_clean_primary"


def test_should_run_verifier_low_confidence():
    primary = OCRResult(
        text="I must forge the legendary sword at once.",
        confidence=0.87,
        raw_text="I must forge the legendary sword at once.",
    )
    should_run, reason = should_run_verifier(primary, region=None, min_confidence=0.92)
    assert should_run is True
    assert "low_confidence" in reason


def test_should_run_verifier_empty_text():
    primary = OCRResult(
        text="",
        confidence=0.95,
        raw_text="",
    )
    should_run, reason = should_run_verifier(primary, region=None, min_confidence=0.92)
    assert should_run is True
    assert reason == "primary_empty"


def test_should_run_verifier_cjk_characters():
    primary = OCRResult(
        text="伝説の鍛冶屋",
        confidence=0.98,
        raw_text="伝説の鍛冶屋",
    )
    should_run, reason = should_run_verifier(primary, region=None, min_confidence=0.92)
    assert should_run is True
    assert "cjk" in reason


def test_should_run_verifier_digit_fused_word():
    primary = OCRResult(
        text="YOU RECEIVED LEVEL20 REWARD",
        confidence=0.97,
        raw_text="YOU RECEIVED LEVEL20 REWARD",
    )
    should_run, reason = should_run_verifier(primary, region=None, min_confidence=0.92)
    assert should_run is True
    assert "digit_fused_word" in reason


def test_should_run_verifier_invalid_consonant_cluster():
    primary = OCRResult(
        text="WHAT IS THIS CDANTED POWER",
        confidence=0.95,
        raw_text="WHAT IS THIS CDANTED POWER",
    )
    should_run, reason = should_run_verifier(primary, region=None, min_confidence=0.92)
    assert should_run is True
    assert "invalid_start_consonants" in reason


def test_should_run_verifier_ambiguous_cjk_region():
    primary = OCRResult(
        text="SLASH",
        confidence=0.95,
        raw_text="SLASH",
    )
    region = Region(
        id=1,
        global_bbox=BBox(10, 10, 100, 100),
        detection_confidence=0.9,
        source_window_ids=[1],
        ocr_confidence=0.9,
        type=RegionType.UNKNOWN,
        status=RegionStatus.REVIEW,
        review_reason="ambiguous_cjk_review",
    )
    should_run, reason = should_run_verifier(primary, region=region, min_confidence=0.92)
    assert should_run is True
    assert reason == "ambiguous_cjk_review"


def test_ocr_provider_recognize_batch_default():
    from providers.ocr.base import OCRProvider

    class DummyOCR(OCRProvider):
        def load(self): pass
        def unload(self): pass
        @property
        def name(self): return "Dummy"
        def recognize(self, image, region_bbox=None):
            return OCRResult(text="DUMMY", confidence=0.99)

    provider = DummyOCR()
    results = provider.recognize_batch(["img1", "img2", "img3"])
    assert len(results) == 3
    assert all(r.text == "DUMMY" for r in results)
    assert provider.recognize_batch([]) == []
