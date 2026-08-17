from __future__ import annotations

import json

from PIL import Image

from application.chapter_analyzer import ChapterAnalyzer
from core.detection import BBox, Detection, Region, RegionStatus, RegionType
from core.detection.region_validity import evaluate_region_validity
from core.detection.classification import classify_regions
from core.coordinate.global_coords import GlobalCoordinateSystem
from core.models import Page
from providers.detector.base import DetectorProvider
from providers.ocr.base import OCRProvider, OCRResult
from providers.ocr.repair import OCRRepairProvider


def _region(
    bbox: BBox,
    *,
    lines: list | None = None,
    segments: list | None = None,
) -> Region:
    return Region(
        id=1,
        global_bbox=bbox,
        type=RegionType.UNKNOWN,
        detection_confidence=0.8,
        source_window_ids=(1,),
        metadata={
            "line_polygons": lines or [],
            "segmentation_polygons": segments or [],
        },
    )


def test_validity_preserves_readable_text_and_recovers_retained_geometry() -> None:
    readable = _region(BBox(10, 10, 110, 50))
    assert evaluate_region_validity(readable, "HELLO").is_valid

    clipped = _region(
        BBox(20, 20, 80, 50),
        lines=[[[10, 20], [100, 20], [100, 50], [10, 50]]],
        segments=[[[12, 24], [98, 24], [98, 46], [12, 46]]],
    )
    decision = evaluate_region_validity(clipped, "ELL")
    assert decision.is_valid
    assert decision.reason == "ctd_geometry_recovery"
    assert decision.recovered_bbox == BBox(10, 20, 100, 50)


def test_validity_rejects_artwork_and_clipped_empty_fragment() -> None:
    artwork = _region(
        BBox(0, 0, 100, 100),
        lines=[[[0, 0], [75, 0], [75, 75], [0, 75]]],
        segments=[[[40, 40], [50, 40], [50, 50], [40, 50]]],
    )
    assert evaluate_region_validity(artwork, "CAAI").reason == "artwork_like_text_geometry"

    fragment = _region(
        BBox(0, 0, 80, 40),
        lines=[[[0, 5], [80, 5], [80, 35], [0, 35]]],
        segments=[[[0, 0], [12, 0], [12, 40], [0, 40]]],
    )
    assert evaluate_region_validity(fragment, "").reason == "clipped_junk_fragment"


class _ArtworkDetector(DetectorProvider):
    @property
    def name(self) -> str:
        return "artwork-detector"

    @property
    def version(self) -> str:
        return "1"

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def detect(self, image: Image.Image, window_id: int = 1) -> list[Detection]:
        return [
            Detection(
                bbox=BBox(20, 20, 120, 120),
                confidence=0.9,
                type=RegionType.UNKNOWN,
                source_window_id=window_id,
                metadata={
                    "line_polygons": [[[20, 20], [95, 20], [95, 95], [20, 95]]],
                    "segmentation_polygons": [[[55, 55], [65, 55], [65, 65], [55, 65]]],
                },
            )
        ]


class _CountingOCR(OCRProvider):
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.calls = 0

    @property
    def name(self) -> str:
        return "counting-ocr"

    @property
    def version(self) -> str:
        return "1"

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def language(self) -> str:
        return "en"

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def recognize(self, image: Image.Image, region_bbox: BBox | None = None) -> OCRResult:
        self.calls += 1
        return OCRResult(self.text, 0.0, raw_text=self.text)


class _CountingRepair(OCRRepairProvider):
    def __init__(self) -> None:
        self.load_calls = 0
        self.repair_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def unload(self) -> None:
        pass

    def repair(self, *args, **kwargs):
        self.repair_calls += 1
        raise AssertionError("invalid CTD region reached visual repair")


def test_invalid_region_never_reaches_verifier_or_qwen(tmp_path) -> None:
    chapter = tmp_path / "chapter"
    chapter.mkdir()
    Image.new("RGB", (200, 200), "white").save(chapter / "001.png")
    output = tmp_path / "output"
    primary = _CountingOCR()
    verifier = _CountingOCR("hallucination")
    repair = _CountingRepair()

    ChapterAnalyzer().process_chapter(
        chapter_path=chapter,
        output_path=output,
        detector=_ArtworkDetector(),
        primary_ocr=primary,
        verifier_ocr=verifier,
        qwen_repair=repair,
    )

    assert primary.calls > 0
    assert verifier.calls == 0
    assert repair.load_calls == 0
    assert repair.repair_calls == 0
    report = json.loads((output / "analysis" / "regions.json").read_text(encoding="utf-8"))
    assert all(region["status"] == RegionStatus.SKIP.value for region in report["regions"])
    assert all(
        region["metadata"]["region_validity"]["reason"] == "artwork_like_text_geometry"
        for region in report["regions"]
    )


def test_validity_rejection_has_precedence_over_later_classification(tmp_path) -> None:
    page_path = tmp_path / "001.png"
    Image.new("RGB", (200, 200), "white").save(page_path)
    coords = GlobalCoordinateSystem((Page(0, page_path, 200, 200, 0),))
    rejected = Region(
        id=7,
        global_bbox=BBox(20, 20, 80, 80),
        type=RegionType.UNKNOWN,
        detection_confidence=0.9,
        source_window_ids=(1,),
        status=RegionStatus.SKIP,
        text="漢",
        review_reason="artwork_like_text_geometry",
        metadata={
            "region_validity": {
                "valid": False,
                "reason": "artwork_like_text_geometry",
            }
        },
    )

    [classified] = classify_regions([rejected], coords)

    assert classified.status is RegionStatus.SKIP
    assert classified.type is RegionType.UNKNOWN
    assert classified.review_reason == "artwork_like_text_geometry"
    assert classified.metadata["classification_diagnostic"] == {
        "proposed_type": RegionType.UNKNOWN.value,
        "proposed_status": RegionStatus.REVIEW.value,
        "proposed_reason": "ambiguous_cjk_review",
    }


def test_short_cjk_without_stylized_geometry_stays_review(tmp_path) -> None:
    page_path = tmp_path / "001.png"
    Image.new("RGB", (200, 200), "white").save(page_path)
    coords = GlobalCoordinateSystem((Page(0, page_path, 200, 200, 0),))
    region = Region(
        id=8,
        global_bbox=BBox(20, 20, 80, 80),
        type=RegionType.UNKNOWN,
        detection_confidence=0.9,
        source_window_ids=(1,),
        status=RegionStatus.REVIEW,
        text="漢",
    )

    [classified] = classify_regions([region], coords)

    assert classified.status is RegionStatus.REVIEW
    assert classified.type is RegionType.UNKNOWN
    assert classified.review_reason == "ambiguous_cjk_review"


def test_single_digit_unknown_stays_review(tmp_path) -> None:
    page_path = tmp_path / "001.png"
    Image.new("RGB", (200, 200), "white").save(page_path)
    coords = GlobalCoordinateSystem((Page(0, page_path, 200, 200, 0),))
    region = Region(
        id=10,
        global_bbox=BBox(50, 50, 100, 100),
        type=RegionType.UNKNOWN,
        detection_confidence=0.5,
        source_window_ids=(1,),
        status=RegionStatus.REVIEW,
        text="7",
    )

    [classified] = classify_regions([region], coords)

    assert classified.status is RegionStatus.REVIEW
    assert classified.type is RegionType.UNKNOWN
    assert classified.review_reason == "ambiguous_unknown_review"


def test_single_letter_unknown_stays_review(tmp_path) -> None:
    page_path = tmp_path / "001.png"
    Image.new("RGB", (200, 200), "white").save(page_path)
    coords = GlobalCoordinateSystem((Page(0, page_path, 200, 200, 0),))
    region = Region(
        id=11,
        global_bbox=BBox(50, 50, 100, 100),
        type=RegionType.UNKNOWN,
        detection_confidence=0.5,
        source_window_ids=(1,),
        status=RegionStatus.REVIEW,
        text="A",
    )

    [classified] = classify_regions([region], coords)

    assert classified.status is RegionStatus.REVIEW
    assert classified.type is RegionType.UNKNOWN
    assert classified.review_reason == "ambiguous_unknown_review"


def test_short_numeric_unknown_stays_review(tmp_path) -> None:
    page_path = tmp_path / "001.png"
    Image.new("RGB", (200, 200), "white").save(page_path)
    coords = GlobalCoordinateSystem((Page(0, page_path, 200, 200, 0),))
    region = Region(
        id=12,
        global_bbox=BBox(50, 50, 100, 100),
        type=RegionType.UNKNOWN,
        detection_confidence=0.5,
        source_window_ids=(1,),
        status=RegionStatus.REVIEW,
        text="11",
    )

    [classified] = classify_regions([region], coords)

    assert classified.status is RegionStatus.REVIEW
    assert classified.type is RegionType.UNKNOWN
    assert classified.review_reason == "ambiguous_unknown_review"


def test_short_alphabetic_unknown_stays_review(tmp_path) -> None:
    page_path = tmp_path / "001.png"
    Image.new("RGB", (200, 200), "white").save(page_path)
    coords = GlobalCoordinateSystem((Page(0, page_path, 200, 200, 0),))
    region = Region(
        id=13,
        global_bbox=BBox(50, 50, 100, 100),
        type=RegionType.UNKNOWN,
        detection_confidence=0.5,
        source_window_ids=(1,),
        status=RegionStatus.REVIEW,
        text="OK",
        review_reason="word_difference",
    )

    [classified] = classify_regions([region], coords)

    assert classified.status is RegionStatus.REVIEW
    assert classified.type is RegionType.UNKNOWN
    assert classified.review_reason == "word_difference"


def test_dialogue_unknown_is_not_affected_by_non_story_skip(tmp_path) -> None:
    page_path = tmp_path / "001.png"
    Image.new("RGB", (200, 200), "white").save(page_path)
    coords = GlobalCoordinateSystem((Page(0, page_path, 200, 200, 0),))
    region = Region(
        id=14,
        global_bbox=BBox(50, 50, 100, 100),
        type=RegionType.DIALOGUE,
        detection_confidence=0.9,
        source_window_ids=(1,),
        status=RegionStatus.AUTO,
        text="HELLO",
    )

    [classified] = classify_regions([region], coords)

    assert classified.status is RegionStatus.AUTO
    assert classified.type is RegionType.DIALOGUE


def test_sfx_region_is_not_affected_by_non_story_skip(tmp_path) -> None:
    page_path = tmp_path / "001.png"
    Image.new("RGB", (200, 200), "white").save(page_path)
    coords = GlobalCoordinateSystem((Page(0, page_path, 200, 200, 0),))
    region = Region(
        id=15,
        global_bbox=BBox(50, 50, 100, 100),
        type=RegionType.SFX,
        detection_confidence=0.9,
        source_window_ids=(1,),
        status=RegionStatus.REVIEW,
        text="BOOM",
    )

    [classified] = classify_regions([region], coords)

    assert classified.status is RegionStatus.SKIP
    assert classified.type is RegionType.SFX
    assert classified.review_reason == "detector_sfx_watermark_skip"
