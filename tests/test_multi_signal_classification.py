"""Unit tests for multi-signal safe classification helper."""

from pathlib import Path
from PIL import Image

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.detection.classification import classify_regions, _is_multi_signal_non_text_noise
from core.detection.detection import BBox, Region, RegionStatus, RegionType
from core.models import Page


def _make_dummy_coords(tmp_path: Path) -> GlobalCoordinateSystem:
    page_path = tmp_path / "001.png"
    Image.new("RGB", (800, 1000), "white").save(page_path)
    return GlobalCoordinateSystem((Page(0, page_path, 800, 1000, 0),))


def test_multi_signal_noise_skips_when_primary_empty_and_verifier_weak(tmp_path: Path) -> None:
    """Multi-signal noise (empty primary OCR + weak verifier single char) safely transitions to SKIP."""
    coords = _make_dummy_coords(tmp_path)
    region = Region(
        id=8,
        global_bbox=BBox(68, 309, 107, 325),
        type=RegionType.UNKNOWN,
        detection_confidence=0.5,
        source_window_ids=(1,),
        status=RegionStatus.REVIEW,
        text="2",
        ocr_confidence=0.0,
        metadata={
            "region_validity": {"valid": True, "primary_alnum_count": 0},
            "ocr_verdict": {"reason": "primary_empty_verifier_filled", "second_pass_invoked": True},
            "repair_eligibility": {"reason": "verifier_only_weak_text_geometry"},
        },
    )

    [classified] = classify_regions([region], coords)

    assert classified.status is RegionStatus.SKIP
    assert classified.type is RegionType.UNKNOWN
    assert classified.review_reason == "non_text_noise_skip"


def test_single_letter_with_positive_primary_ocr_stays_review(tmp_path: Path) -> None:
    """Single letter with positive primary OCR confidence must stay in REVIEW (zero story loss)."""
    coords = _make_dummy_coords(tmp_path)
    region = Region(
        id=101,
        global_bbox=BBox(100, 100, 150, 150),
        type=RegionType.UNKNOWN,
        detection_confidence=0.8,
        source_window_ids=(1,),
        status=RegionStatus.REVIEW,
        text="I",
        ocr_confidence=0.92,
        metadata={
            "region_validity": {"valid": True, "primary_alnum_count": 1},
            "ocr_verdict": {"reason": None, "second_pass_invoked": False},
            "repair_eligibility": {"reason": "no_unresolved_ocr_ambiguity"},
        },
    )

    [classified] = classify_regions([region], coords)

    assert classified.status is RegionStatus.REVIEW
    assert classified.type is RegionType.UNKNOWN
    assert classified.review_reason == "ambiguous_unknown_review"


def test_dialogue_is_never_touched_by_multi_signal_filter(tmp_path: Path) -> None:
    """DIALOGUE region is never filtered by the UNKNOWN noise helper."""
    coords = _make_dummy_coords(tmp_path)
    region = Region(
        id=201,
        global_bbox=BBox(50, 50, 200, 100),
        type=RegionType.DIALOGUE,
        detection_confidence=0.95,
        source_window_ids=(1,),
        status=RegionStatus.AUTO,
        text="1",
        ocr_confidence=0.0,
        metadata={
            "region_validity": {"valid": True, "primary_alnum_count": 0},
            "ocr_verdict": {"reason": "primary_empty_verifier_filled"},
            "repair_eligibility": {"reason": "verifier_only_weak_text_geometry"},
        },
    )

    [classified] = classify_regions([region], coords)

    assert classified.status is RegionStatus.AUTO
    assert classified.type is RegionType.DIALOGUE


def test_narration_is_never_touched_by_multi_signal_filter(tmp_path: Path) -> None:
    """NARRATION region is never filtered by the UNKNOWN noise helper."""
    coords = _make_dummy_coords(tmp_path)
    region = Region(
        id=301,
        global_bbox=BBox(50, 50, 200, 100),
        type=RegionType.NARRATION,
        detection_confidence=0.95,
        source_window_ids=(1,),
        status=RegionStatus.AUTO,
        text="A",
        ocr_confidence=0.0,
        metadata={
            "region_validity": {"valid": True, "primary_alnum_count": 0},
            "ocr_verdict": {"reason": "primary_empty_verifier_filled"},
            "repair_eligibility": {"reason": "verifier_only_weak_text_geometry"},
        },
    )

    [classified] = classify_regions([region], coords)

    assert classified.status is RegionStatus.AUTO
    assert classified.type is RegionType.NARRATION


def test_multi_char_unknown_story_text_is_not_skipped(tmp_path: Path) -> None:
    """Multi-char UNKNOWN story text is not skipped."""
    coords = _make_dummy_coords(tmp_path)
    region = Region(
        id=401,
        global_bbox=BBox(100, 100, 300, 200),
        type=RegionType.UNKNOWN,
        detection_confidence=0.7,
        source_window_ids=(1,),
        status=RegionStatus.AUTO,
        text="SWORD",
        ocr_confidence=0.88,
    )

    [classified] = classify_regions([region], coords)

    assert classified.status is RegionStatus.AUTO
    assert classified.type is RegionType.UNKNOWN


def test_validity_rejected_region_strictly_maintains_rejection(tmp_path: Path) -> None:
    """Validity rejection outranks heuristic classification."""
    coords = _make_dummy_coords(tmp_path)
    region = Region(
        id=501,
        global_bbox=BBox(100, 100, 200, 200),
        type=RegionType.UNKNOWN,
        detection_confidence=0.5,
        source_window_ids=(1,),
        status=RegionStatus.SKIP,
        text="2",
        ocr_confidence=0.0,
        review_reason="invalid_boundary_noise",
        metadata={"region_validity": {"valid": False, "reason": "invalid_boundary_noise"}},
    )

    [classified] = classify_regions([region], coords)

    assert classified.status is RegionStatus.SKIP
    assert classified.review_reason == "invalid_boundary_noise"


def test_credit_metadata_on_boundary_pages_is_skipped(tmp_path: Path) -> None:
    """Wide banner / credit line on cover page or end card page is classified as credit SKIP."""
    p1 = tmp_path / "001.png"
    p2 = tmp_path / "002.png"
    p3 = tmp_path / "003.png"
    for p in (p1, p2, p3):
        Image.new("RGB", (800, 1000), "white").save(p)
    
    coords = GlobalCoordinateSystem((
        Page(0, p1, 800, 1000, 0),
        Page(1, p2, 800, 1000, 1000),
        Page(2, p3, 800, 1000, 2000),
    ))

    # Cover page banner element
    cover_region = Region(
        id=601,
        global_bbox=BBox(482, 35, 725, 101),  # width=243, height=66 on page 0
        type=RegionType.UNKNOWN,
        detection_confidence=0.8,
        source_window_ids=(0,),
        status=RegionStatus.REVIEW,
        text="7",
        ocr_confidence=0.89,
    )

    # End page credit line element
    end_region = Region(
        id=602,
        global_bbox=BBox(160, 2100, 647, 2126),  # width=487, height=26 on page 2 (Y in [2000..3000])
        type=RegionType.UNKNOWN,
        detection_confidence=0.8,
        source_window_ids=(2,),
        status=RegionStatus.REVIEW,
        text="M",
        ocr_confidence=0.72,
    )

    classified = classify_regions([cover_region, end_region], coords)
    assert classified[0].status is RegionStatus.SKIP
    assert classified[0].type is RegionType.WATERMARK
    assert classified[0].review_reason == "credit_metadata_skip"

    assert classified[1].status is RegionStatus.SKIP
    assert classified[1].type is RegionType.WATERMARK
    assert classified[1].review_reason == "credit_metadata_skip"


def test_isolated_drawing_sfx_is_skipped(tmp_path: Path) -> None:
    """Large drawing glyph with short vocalization text is classified as SFX SKIP."""
    coords = _make_dummy_coords(tmp_path)
    sfx_region = Region(
        id=701,
        global_bbox=BBox(176, 100, 442, 470),  # width=266, height=370, area=98,420
        type=RegionType.UNKNOWN,
        detection_confidence=0.9,
        source_window_ids=(1,),
        status=RegionStatus.REVIEW,
        text="1",
        ocr_confidence=0.54,
    )

    [classified] = classify_regions([sfx_region], coords)

    assert classified.status is RegionStatus.SKIP
    assert classified.type is RegionType.SFX
    assert classified.review_reason == "sfx_skip"


def test_middle_page_small_dialogue_candidate_never_skipped(tmp_path: Path) -> None:
    """Middle page small text (e.g. single digit or letter) with positive OCR stays in REVIEW."""
    p1 = tmp_path / "001.png"
    p2 = tmp_path / "002.png"
    p3 = tmp_path / "003.png"
    for p in (p1, p2, p3):
        Image.new("RGB", (800, 1000), "white").save(p)
    
    coords = GlobalCoordinateSystem((
        Page(0, p1, 800, 1000, 0),
        Page(1, p2, 800, 1000, 1000),
        Page(2, p3, 800, 1000, 2000),
    ))

    # Middle page small text (width=70, height=70) on page 1 (Y in [1000..2000])
    region = Region(
        id=801,
        global_bbox=BBox(300, 1400, 370, 1470),
        type=RegionType.UNKNOWN,
        detection_confidence=0.8,
        source_window_ids=(1,),
        status=RegionStatus.REVIEW,
        text="3",
        ocr_confidence=0.85,
    )

    [classified] = classify_regions([region], coords)

    assert classified.status is RegionStatus.REVIEW
    assert classified.type is RegionType.UNKNOWN
    assert classified.review_reason == "ambiguous_unknown_review"


def test_real_story_dialogue_with_exclamation_stays_auto(tmp_path: Path) -> None:
    """Real story dialogue words ('Wait!', 'No!') are always preserved as AUTO."""
    coords = _make_dummy_coords(tmp_path)
    dialogue = Region(
        id=901,
        global_bbox=BBox(200, 200, 400, 300),
        type=RegionType.DIALOGUE,
        detection_confidence=0.98,
        source_window_ids=(1,),
        status=RegionStatus.AUTO,
        text="Wait!",
        ocr_confidence=0.99,
    )

    [classified] = classify_regions([dialogue], coords)

    assert classified.status is RegionStatus.AUTO
    assert classified.type is RegionType.DIALOGUE


def test_cjk_stylized_sfx_is_skipped(tmp_path: Path) -> None:
    """Large drawing CJK glyph (e.g. Katakana onomatopoeia) is skipped as SFX."""
    coords = _make_dummy_coords(tmp_path)
    cjk_sfx = Region(
        id=1001,
        global_bbox=BBox(100, 100, 406, 398),  # 306x298 px, area=91,188
        type=RegionType.UNKNOWN,
        detection_confidence=0.85,
        source_window_ids=(1,),
        status=RegionStatus.REVIEW,
        text="チッ",
        ocr_confidence=0.38,
    )

    [classified] = classify_regions([cjk_sfx], coords)

    assert classified.status is RegionStatus.SKIP
    assert classified.type is RegionType.SFX
    assert classified.review_reason == "cjk_stylized_sfx_skip"


def test_cjk_multi_word_story_text_stays_review(tmp_path: Path) -> None:
    """Multi-word CJK text (e.g. titles/dialogue candidate) is preserved in REVIEW."""
    coords = _make_dummy_coords(tmp_path)
    cjk_story = Region(
        id=1002,
        global_bbox=BBox(50, 50, 400, 200),
        type=RegionType.UNKNOWN,
        detection_confidence=0.90,
        source_window_ids=(1,),
        status=RegionStatus.REVIEW,
        text="転生 神級 クラス",
        ocr_confidence=0.64,
    )

    [classified] = classify_regions([cjk_story], coords)

    assert classified.status is RegionStatus.REVIEW
    assert classified.type is RegionType.UNKNOWN
    assert classified.review_reason == "ambiguous_cjk_review"


def test_cjk_texture_noise_is_skipped(tmp_path: Path) -> None:
    """Primary empty OCR + verifier single CJK hallucination on texture is skipped as noise."""
    coords = _make_dummy_coords(tmp_path)
    noise_cjk = Region(
        id=1003,
        global_bbox=BBox(100, 100, 140, 130),  # 40x30 px small texture
        type=RegionType.UNKNOWN,
        detection_confidence=0.5,
        source_window_ids=(1,),
        status=RegionStatus.REVIEW,
        text="中",
        ocr_confidence=0.0,
        metadata={
            "region_validity": {"valid": True, "primary_alnum_count": 0},
            "ocr_verdict": {"reason": "primary_empty_verifier_filled"},
            "repair_eligibility": {"reason": "verifier_only_weak_text_geometry"},
        },
    )

    [classified] = classify_regions([noise_cjk], coords)

    assert classified.status is RegionStatus.SKIP
    assert classified.type is RegionType.UNKNOWN
    assert classified.review_reason == "non_text_noise_skip"
