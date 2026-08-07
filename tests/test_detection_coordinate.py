"""Local/global bbox ve polygon dönüşüm testleri."""

from __future__ import annotations

import pytest

from core.detection import BBox
from core.detection.coordinate import (
    global_bbox_to_window,
    global_polygon_to_window,
    window_bbox_to_global,
    window_polygon_to_global,
)


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


# ---------------------------------------------------------------------------
# Polygon tests
# ---------------------------------------------------------------------------

def test_window_polygon_to_global_basic() -> None:
    """Window-local polygon global'a doğru çevrilmeli."""
    local_polygon = [[10.0, 20.0], [110.0, 20.0], [110.0, 120.0], [10.0, 120.0]]
    global_polygon = window_polygon_to_global(local_polygon, 5000)
    assert global_polygon == [
        [10.0, 5020.0],
        [110.0, 5020.0],
        [110.0, 5120.0],
        [10.0, 5120.0],
    ]


def test_global_polygon_to_window_basic() -> None:
    """Global polygon window-local'a doğru çevrilmeli."""
    global_polygon = [[10.0, 5020.0], [110.0, 5020.0], [110.0, 5120.0], [10.0, 5120.0]]
    local_polygon = global_polygon_to_window(global_polygon, 5000)
    assert local_polygon == [
        [10.0, 20.0],
        [110.0, 20.0],
        [110.0, 120.0],
        [10.0, 120.0],
    ]


def test_polygon_roundtrip() -> None:
    """Polygon gidiş-dönüş tutarlı."""
    original = [[5.0, 15.0], [25.0, 15.0], [25.0, 35.0], [5.0, 35.0]]
    window_y_start = 8000
    roundtripped = global_polygon_to_window(
        window_polygon_to_global(original, window_y_start), window_y_start
    )
    assert roundtripped == original


def test_polygon_zero_offset() -> None:
    """Offset 0 ise polygon aynı kalmalı."""
    polygon = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    assert window_polygon_to_global(polygon, 0) == polygon
    assert global_polygon_to_window(polygon, 0) == polygon


def test_polygon_high_offset_window_5() -> None:
    """Window y_start=20000 ile polygon y'si 20000 artmalı."""
    local_polygon = [[595.7, 1030.2], [650.3, 1030.2], [650.3, 1240.8], [595.7, 1240.8]]
    global_polygon = window_polygon_to_global(local_polygon, 20000)
    assert global_polygon == [
        [595.7, 21030.2],
        [650.3, 21030.2],
        [650.3, 21240.8],
        [595.7, 21240.8],
    ]


def test_polygon_high_offset_window_8() -> None:
    """Window y_start=32000 ile polygon y'si 32000 artmalı."""
    local_polygon = [[37.0, 50.0], [181.0, 50.0], [181.0, 112.0], [37.0, 112.0]]
    global_polygon = window_polygon_to_global(local_polygon, 32000)
    assert global_polygon == [
        [37.0, 32050.0],
        [181.0, 32050.0],
        [181.0, 32112.0],
        [37.0, 32112.0],
    ]


def test_polygon_empty_list() -> None:
    """Boş polygon listesi boş döner."""
    assert window_polygon_to_global([], 5000) == []
    assert global_polygon_to_window([], 5000) == []


def test_polygon_single_point() -> None:
    """Tek nokta polygon da çalışır."""
    polygon = [[100.0, 200.0]]
    assert window_polygon_to_global(polygon, 4000) == [[100.0, 4200.0]]
    assert global_polygon_to_window([[100.0, 4200.0]], 4000) == [[100.0, 200.0]]