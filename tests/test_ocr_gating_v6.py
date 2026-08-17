"""Unit tests for OCR Confidence Gating (0.85) and High Performance Batching."""

from __future__ import annotations

import pytest

from core.system.adaptive_batcher import BatchConfig, get_batch_config, set_batch_config
from providers.ocr.agreement import should_run_verifier
from providers.ocr.base import OCRResult


def test_batch_config_defaults():
    """Verifies that BatchConfig defaults are set for high-throughput GPU execution."""
    cfg = BatchConfig()
    assert cfg.detector_tile_batch == 16
    assert cfg.ocr_vl_batch == 64
    assert cfg.lama_batch == 24
    assert cfg.vram_ceiling == 0.95


def test_should_run_verifier_gating_085():
    """Verifies that confidence >= 0.85 cleanly bypasses secondary verifier."""
    # High confidence clean text (0.88)
    clean_res = OCRResult(text="Hello world!", confidence=0.88, raw_text="Hello world!")
    should_run, reason = should_run_verifier(clean_res, min_confidence=0.85)
    assert should_run is False
    assert reason == "high_confidence_clean_primary"

    # Borderline confidence (0.85)
    borderline_res = OCRResult(text="Good morning", confidence=0.85, raw_text="Good morning")
    should_run_b, reason_b = should_run_verifier(borderline_res, min_confidence=0.85)
    assert should_run_b is False
    assert reason_b == "high_confidence_clean_primary"

    # Low confidence (0.78)
    low_res = OCRResult(text="Unclear text", confidence=0.78, raw_text="Unclear text")
    should_run_l, reason_l = should_run_verifier(low_res, min_confidence=0.85)
    assert should_run_l is True
    assert "low_confidence" in reason_l

    # Empty text
    empty_res = OCRResult(text="", confidence=0.99, raw_text="")
    should_run_e, reason_e = should_run_verifier(empty_res, min_confidence=0.85)
    assert should_run_e is True
    assert reason_e == "primary_empty"
