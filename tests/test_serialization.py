"""Region/Detection serileştirme testleri."""

from __future__ import annotations

import pytest

from core.detection import BBox, Detection, Region, RegionStatus, RegionType
from core.serialization import detection_to_dict, dict_to_detection, region_to_dict, dict_to_region


def _sample_region() -> Region:
    return Region(
        id=1,
        global_bbox=BBox(x1=10, y1=20, x2=110, y2=120),
        type=RegionType.DIALOGUE,
        detection_confidence=0.85,
        source_window_ids=(0, 1, 2),
        status=RegionStatus.AUTO,
        text="Hello",
        ocr_confidence=0.9,
        translation="Merhaba",
        review_reason=None,
    )


def test_region_roundtrip() -> None:
    """Region -> dict -> Region kayıpsız dönüşüm."""
    reg = _sample_region()
    data = region_to_dict(reg)
    restored = dict_to_region(data)
    assert restored == reg


def test_region_dict_contains_expected_fields() -> None:
    """Dict'te tüm alanlar var."""
    reg = _sample_region()
    data = region_to_dict(reg)
    assert data["id"] == 1
    assert data["global_bbox"] == {"x1": 10, "y1": 20, "x2": 110, "y2": 120}
    assert data["type"] == "dialogue"
    assert data["status"] == "auto"
    assert data["source_window_ids"] == [0, 1, 2]


def test_detection_roundtrip() -> None:
    """Detection -> dict -> Detection dönüşümü."""
    det = Detection(
        bbox=BBox(x1=5, y1=5, x2=15, y2=15),
        confidence=0.7,
        type=RegionType.NARRATION,
        source_window_id=3,
        metadata={"provider": "dummy"},
    )
    data = detection_to_dict(det)
    restored = dict_to_detection(data)
    assert restored == det


def test_region_none_text_and_confidence() -> None:
    """None alanlar korunmalı."""
    reg = Region(
        id=0,
        global_bbox=BBox(x1=0, y1=0, x2=10, y2=10),
        type=RegionType.UNKNOWN,
        detection_confidence=0.5,
        source_window_ids=(),
        status=RegionStatus.REVIEW,
        text=None,
        ocr_confidence=None,
        translation=None,
        review_reason="low confidence",
    )
    data = region_to_dict(reg)
    restored = dict_to_region(data)
    assert restored.text is None
    assert restored.ocr_confidence is None
    assert restored.translation is None
    assert restored.review_reason == "low confidence"


def test_region_status_defaults_to_auto_when_missing() -> None:
    """Status eksikse AUTO kullanılmalı."""
    data = {
        "id": 1,
        "global_bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        "type": "dialogue",
        "detection_confidence": 0.5,
        "source_window_ids": [],
    }
    reg = dict_to_region(data)
    assert reg.status == RegionStatus.AUTO