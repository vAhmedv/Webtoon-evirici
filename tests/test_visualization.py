"""Görselleştirme testleri."""

from __future__ import annotations

from PIL import Image

from core.detection import BBox, Detection, Region, RegionStatus, RegionType
from core.visualization import draw_detections, draw_regions


def test_draw_detections_returns_rgb_image() -> None:
    """draw_detections RGB görüntü döndürmeli."""
    img = Image.new("RGB", (100, 100))
    dets = [
        Detection(
            bbox=BBox(x1=10, y1=10, x2=50, y2=50),
            confidence=0.8,
            type=RegionType.DIALOGUE,
            source_window_id=0,
        )
    ]
    out = draw_detections(img, dets)
    assert out.mode == "RGB"
    assert out.size == (100, 100)


def test_draw_regions_returns_rgb_image() -> None:
    """draw_regions RGB görüntü döndürmeli."""
    img = Image.new("RGB", (100, 100))
    regs = [
        Region(
            id=0,
            global_bbox=BBox(x1=10, y1=10, x2=50, y2=50),
            type=RegionType.DIALOGUE,
            detection_confidence=0.8,
            source_window_ids=(0,),
            status=RegionStatus.AUTO,
        )
    ]
    out = draw_regions(img, regs, window_y_start=0)
    assert out.mode == "RGB"
    assert out.size == (100, 100)


def test_draw_regions_outside_window_skipped() -> None:
    """Pencerenin dışındaki region'lar çizilmemeli."""
    img = Image.new("RGB", (100, 100))
    regs = [
        Region(
            id=0,
            global_bbox=BBox(x1=10, y1=1000, x2=50, y2=1050),
            type=RegionType.DIALOGUE,
            detection_confidence=0.8,
            source_window_ids=(0,),
            status=RegionStatus.AUTO,
        )
    ]
    out = draw_regions(img, regs, window_y_start=0)
    assert out.mode == "RGB"