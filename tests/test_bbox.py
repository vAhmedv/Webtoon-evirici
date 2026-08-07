"""BBox testleri."""

from __future__ import annotations

import pytest

from core.detection import BBox


def test_bbox_properties() -> None:
    """width/height/area hesaplanmalı."""
    bbox = BBox(x1=10, y1=20, x2=110, y2=120)
    assert bbox.width == 100
    assert bbox.height == 100
    assert bbox.area == 10000


def test_bbox_center() -> None:
    """Merkez koordinat doğru."""
    bbox = BBox(x1=0, y1=0, x2=10, y2=20)
    assert bbox.center == (5.0, 10.0)


def test_bbox_invalid_raises() -> None:
    """x2<=x1 veya y2<=y1 ValueError fırlatmalı."""
    with pytest.raises(ValueError):
        BBox(x1=10, y1=20, x2=10, y2=30)
    with pytest.raises(ValueError):
        BBox(x1=10, y1=20, x2=20, y2=20)


def test_bbox_intersection() -> None:
    """Kesişim hesaplama."""
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    b = BBox(x1=5, y1=5, x2=15, y2=15)
    inter = a.intersection(b)
    assert inter is not None
    assert inter.x1 == 5 and inter.y1 == 5 and inter.x2 == 10 and inter.y2 == 10


def test_bbox_no_intersection() -> None:
    """Kesişim yoksa None."""
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    b = BBox(x1=20, y1=20, x2=30, y2=30)
    assert a.intersection(b) is None


def test_bbox_full_overlap() -> None:
    """Tam örtüşme."""
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    b = BBox(x1=0, y1=0, x2=10, y2=10)
    inter = a.intersection(b)
    assert inter is not None
    assert inter.area == 100


def test_bbox_iou() -> None:
    """IoU hesabı."""
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    b = BBox(x1=5, y1=5, x2=15, y2=15)
    # Kesişim alanı 25, birleşim alanı 175
    expected_iou = 25 / 175
    assert abs(a.iou(b) - expected_iou) < 1e-6


def test_bbox_iou_no_overlap() -> None:
    """Kesişim yoksa IoU 0."""
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    b = BBox(x1=20, y1=20, x2=30, y2=30)
    assert a.iou(b) == 0.0


def test_bbox_clip_inside() -> None:
    """Kırpma sınırı içinde kalırsa değişmez."""
    bbox = BBox(x1=5, y1=5, x2=15, y2=15)
    clipped = bbox.clip(0, 0, 20, 20)
    assert clipped is not None
    assert clipped == bbox


def test_bbox_clip_outside() -> None:
    """Tamamen dışarıda kalırsa None."""
    bbox = BBox(x1=0, y1=0, x2=10, y2=10)
    assert bbox.clip(20, 20, 30, 30) is None


def test_bbox_clip_partial() -> None:
    """Kısmi kırpma."""
    bbox = BBox(x1=5, y1=5, x2=15, y2=15)
    clipped = bbox.clip(10, 10, 20, 20)
    assert clipped is not None
    assert clipped == BBox(x1=10, y1=10, x2=15, y2=15)


def test_bbox_to_tuple_roundtrip() -> None:
    """Tuple dönüşümü roundtrip."""
    bbox = BBox(x1=1, y1=2, x2=3, y2=4)
    assert BBox.from_tuple(bbox.to_tuple()) == bbox