"""Local/global bbox dönüşüm testleri."""

from __future__ import annotations

import pytest

from core.detection import BBox
from core.detection.coordinate import global_bbox_to_window, window_bbox_to_global


def test_window_to_global_basic() -> None:
    """Window-local bbox global'a doğru çevrilmeli."""
    local = BBox(x1=10, y1=20, x2=110, y2=120)
    window_y_start = 5000
    global_bbox = window_bbox_to_global(local, window_y_start)
    assert global_bbox == BBox(x1=10, y1=5020, x2=110, y2=5120)


def test_global_to_window_basic() -> None:
    """Global bbox window-local'a doğru çevrilmeli."""
    global_bbox = BBox(x1=10, y1=5020, x2=110, y2=5120)
    window_y_start = 5000
    local = global_bbox_to_window(global_bbox, window_y_start)
    assert local == BBox(x1=10, y1=20, x2=110, y2=120)


def test_roundtrip() -> None:
    """Gidiş-dönüş tutarlı."""
    original = BBox(x1=5, y1=15, x2=25, y2=35)
    window_y_start = 8000
    roundtripped = global_bbox_to_window(
        window_bbox_to_global(original, window_y_start), window_y_start
    )
    assert roundtripped == original


def test_zero_offset() -> None:
    """Offset 0 ise aynı kalmalı."""
    bbox = BBox(x1=0, y1=0, x2=10, y2=10)
    assert window_bbox_to_global(bbox, 0) == bbox
    assert global_bbox_to_window(bbox, 0) == bbox


def test_negative_window_start() -> None:
    """Negatif window_y_start (sınır durumu)."""
    local = BBox(x1=0, y1=0, x2=10, y2=10)
    global_bbox = window_bbox_to_global(local, -100)
    assert global_bbox == BBox(x1=0, y1=-100, x2=10, y2=-90)