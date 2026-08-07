"""Duplicate merger testleri."""

from __future__ import annotations

import pytest

from core.detection import BBox, Detection, RegionStatus, RegionType
from core.detection.merge import merge_duplicates


def _make_detection(bbox: tuple[int, int, int, int], window_id: int, confidence: float = 0.9, rtype: RegionType = RegionType.DIALOGUE) -> Detection:
    return Detection(
        bbox=BBox.from_tuple(bbox),
        confidence=confidence,
        type=rtype,
        source_window_id=window_id,
    )


def test_merge_two_overlapping_detections_same_window() -> None:
    """Aynı window'daki iki detection merge edilmeli."""
    dets = [
        _make_detection((100, 100, 200, 200), window_id=0),
        _make_detection((120, 120, 220, 220), window_id=0),
    ]
    regions = merge_duplicates(dets, iou_threshold=0.4)
    assert len(regions) == 1
    reg = regions[0]
    assert reg.source_window_ids == (0,)
    assert reg.type == RegionType.DIALOGUE
    assert reg.detection_confidence == 0.9
    assert reg.status == RegionStatus.AUTO


def test_merge_different_regions_stay_separate() -> None:
    """Farklı konumdaki detection'lar ayrı kalmalı."""
    dets = [
        _make_detection((100, 100, 200, 200), window_id=0),
        _make_detection((1000, 1000, 1100, 1100), window_id=0),
    ]
    regions = merge_duplicates(dets, iou_threshold=0.5)
    assert len(regions) == 2


def test_iou_threshold_blocks_merge() -> None:
    """IoU eşiğinin altındaki çiftler merge edilmemeli."""
    dets = [
        _make_detection((100, 100, 200, 200), window_id=0),
        _make_detection((500, 500, 600, 600), window_id=0),
    ]
    regions = merge_duplicates(dets, iou_threshold=0.5)
    assert len(regions) == 2


def test_merge_from_different_windows() -> None:
    """Farklı window'lardan gelen aynı bölge merge edilmeli."""
    dets = [
        _make_detection((100, 100, 200, 200), window_id=0),
        _make_detection((105, 105, 205, 205), window_id=1),
    ]
    regions = merge_duplicates(dets, iou_threshold=0.5)
    assert len(regions) == 1
    assert set(regions[0].source_window_ids) == {0, 1}


def test_source_window_ids_preserved() -> None:
    """source_window_ids merge sonrası korunmalı."""
    dets = [
        _make_detection((100, 100, 200, 200), window_id=0),
        _make_detection((110, 110, 210, 210), window_id=1),
        _make_detection((105, 105, 215, 215), window_id=2),
    ]
    regions = merge_duplicates(dets, iou_threshold=0.5)
    assert len(regions) == 1
    assert set(regions[0].source_window_ids) == {0, 1, 2}


def test_sfx_skipped() -> None:
    """SFX tipi SKIP durumunda olmalı."""
    dets = [
        _make_detection((100, 100, 200, 200), window_id=0, rtype=RegionType.SFX),
    ]
    regions = merge_duplicates(dets, iou_threshold=0.5)
    assert len(regions) == 1
    assert regions[0].status == RegionStatus.SKIP


def test_low_confidence_review() -> None:
    """Düşük confidence → REVIEW."""
    dets = [
        _make_detection((100, 100, 200, 200), window_id=0, confidence=0.3),
    ]
    regions = merge_duplicates(dets, iou_threshold=0.5)
    assert regions[0].status == RegionStatus.REVIEW