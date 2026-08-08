
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


def test_containment_merge_keeps_larger_bbox() -> None:
    """Window sınırı partial/full detection: küçük bbox büyüğün içindeyse
    daha büyük bbox korunmalı."""
    # ID 9 bbox = [169,8935,584,8998] (küçük)
    # ID 13 bbox = [155,8937,591,9119] (büyük)
    # Küçük bbox ~%96.8 içinde, IoU ~0.316
    dets = [
        _make_detection((169, 8935, 584, 8998), window_id=0, confidence=0.9),
        _make_detection((155, 8937, 591, 9119), window_id=1, confidence=0.95),
    ]
    regions = merge_duplicates(dets, iou_threshold=0.5, center_distance_threshold=200)
    assert len(regions) == 1
    reg = regions[0]
    # Büyük bbox korunmalı
    assert reg.global_bbox == BBox(x1=155, y1=8937, x2=591, y2=9119)
    assert reg.detection_confidence == 0.95
    assert set(reg.source_window_ids) == {0, 1}


def test_separate_balloons_not_merged_by_containment() -> None:
    """Yakın ama ayrı balonlar containment eşiğinin altındaysa birleşmez."""
    # İki ayrı balon, IoU düşük ve containment da %95'in altında
    dets = [
        _make_detection((100, 100, 200, 200), window_id=0),
        _make_detection((210, 100, 310, 200), window_id=0),
    ]
    regions = merge_duplicates(dets, iou_threshold=0.5)
    assert len(regions) == 2
