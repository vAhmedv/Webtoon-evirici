"""V5.3 Production OCR State Machine & Classification Regression Tests.

Verifies:
1. UNKNOWN detections with text-like OCR remain eligible for REVIEW/Qwen repair when OCR is suspicious.
2. Clean UNKNOWN detections with valid story text proceed to AUTO and are preserved for classification.
3. True non-text UNKNOWN detections (empty/noise) become SKIP.
4. Absolute removal of hardcoded CRITICAL_NAMES dependency from OCR agreement.
"""

from __future__ import annotations

from core.detection import BBox, Region, RegionStatus, RegionType
from providers.ocr.agreement import decide_ocr_agreement
from providers.ocr.base import OCRResult


def test_no_critical_names_in_agreement_module() -> None:
    """Verify CRITICAL_NAMES is completely removed from agreement module."""
    import providers.ocr.agreement as agreement_module

    assert not hasattr(agreement_module, "CRITICAL_NAMES")
    assert not hasattr(agreement_module, "_critical_name_mismatch")


def test_unknown_suspicious_text_remains_review_and_repairable() -> None:
    """CTD UNKNOWN region with suspicious text (e.g. word mismatch) must NOT be skipped."""
    primary_res = OCRResult(text="MY NAME IS LHO TIAN", raw_text="MY NAME IS LHO TIAN", confidence=0.85)
    verifier_res = OCRResult(text="MY NAME IS LUO TIAN", raw_text="MY NAME IS LUO TIAN", confidence=0.92)

    verdict = decide_ocr_agreement(primary_res, verifier_res)

    assert verdict.requires_review is True
    assert verdict.needs_repair is True
    assert verdict.reason == "word_difference"

    # Simulate ChapterAnalyzer status assignment logic for UNKNOWN
    reg = Region(
        id=101,
        global_bbox=BBox(10, 10, 200, 50),
        type=RegionType.UNKNOWN,
        detection_confidence=0.9,
        source_window_ids=(1,),
    )

    accepted = verdict.accepted_text or verdict.provisional_text or primary_res.text or ""
    has_text_content = bool(accepted and accepted.strip() and any(c.isalnum() for c in accepted))

    status = RegionStatus.SKIP if (reg.type == RegionType.UNKNOWN and not has_text_content) else (
        RegionStatus.REVIEW if verdict.requires_review else RegionStatus.AUTO
    )

    assert status == RegionStatus.REVIEW
    assert reg.type == RegionType.UNKNOWN
    # Must be eligible for Qwen repair queue
    assert (
        status == RegionStatus.REVIEW
        and reg.type in (RegionType.DIALOGUE, RegionType.NARRATION, RegionType.UNKNOWN)
        and verdict.needs_repair
        and bool(accepted)
    )


def test_unknown_clean_story_text_becomes_auto() -> None:
    """CTD UNKNOWN region with clean matching text becomes AUTO."""
    primary_res = OCRResult(text="WHAT ARE YOU DOING HERE?", raw_text="WHAT ARE YOU DOING HERE?", confidence=0.98)
    verifier_res = OCRResult(text="WHAT ARE YOU DOING HERE?", raw_text="WHAT ARE YOU DOING HERE?", confidence=0.97)

    verdict = decide_ocr_agreement(primary_res, verifier_res)

    assert verdict.requires_review is False
    assert verdict.needs_repair is False

    reg = Region(
        id=102,
        global_bbox=BBox(10, 60, 200, 100),
        type=RegionType.UNKNOWN,
        detection_confidence=0.95,
        source_window_ids=(1,),
    )

    accepted = verdict.accepted_text or verdict.provisional_text or primary_res.text or ""
    has_text_content = bool(accepted and accepted.strip() and any(c.isalnum() for c in accepted))

    status = RegionStatus.SKIP if (reg.type == RegionType.UNKNOWN and not has_text_content) else (
        RegionStatus.REVIEW if verdict.requires_review else RegionStatus.AUTO
    )

    assert status == RegionStatus.AUTO


def test_unknown_true_non_text_becomes_skip() -> None:
    """CTD UNKNOWN region with empty text or no alphanumeric characters becomes SKIP."""
    # Empty text
    primary_res_empty = OCRResult(text="", raw_text="", confidence=0.0)
    verdict_empty = decide_ocr_agreement(primary_res_empty, None)

    reg = Region(
        id=103,
        global_bbox=BBox(10, 110, 200, 150),
        type=RegionType.UNKNOWN,
        detection_confidence=0.5,
        source_window_ids=(1,),
    )

    accepted = verdict_empty.accepted_text or verdict_empty.provisional_text or primary_res_empty.text or ""
    has_text_content = bool(accepted and accepted.strip() and any(c.isalnum() for c in accepted))

    status = RegionStatus.SKIP if (reg.type == RegionType.UNKNOWN and not has_text_content) else (
        RegionStatus.REVIEW if verdict_empty.requires_review else RegionStatus.AUTO
    )

    assert status == RegionStatus.SKIP

    # Punctuation/noise only text
    primary_res_noise = OCRResult(text="...", raw_text="...", confidence=0.3)
    verdict_noise = decide_ocr_agreement(primary_res_noise, None)

    accepted_noise = verdict_noise.accepted_text or verdict_noise.provisional_text or primary_res_noise.text or ""
    has_text_content_noise = bool(accepted_noise and accepted_noise.strip() and any(c.isalnum() for c in accepted_noise))

    status_noise = RegionStatus.SKIP if (reg.type == RegionType.UNKNOWN and not has_text_content_noise) else (
        RegionStatus.REVIEW if verdict_noise.requires_review else RegionStatus.AUTO
    )

    assert status_noise == RegionStatus.SKIP
