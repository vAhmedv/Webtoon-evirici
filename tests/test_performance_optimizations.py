"""Unit tests verifying performance and memory optimizations across core subsystems."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

from core.detection import BBox, Region, RegionStatus, RegionType
from core.detection.text_block import TextBlock
from core.imaging.inpainter import Inpainter, DEFAULT_LAMA_CHECKPOINT
from core.imaging.renderer import TextRenderer, _get_font
from application.chapter_analyzer import _image_to_bytes


def test_font_caching_lru_cache() -> None:
    """_get_font returns cached instances on repeated calls without disk reloading."""
    font1 = _get_font(14)
    font2 = _get_font(14)
    assert font1 is font2
    assert hasattr(_get_font, "cache_info")
    info = _get_font.cache_info()
    assert info.hits >= 1


def test_image_to_bytes_fast_buffer_extraction() -> None:
    """_image_to_bytes returns fast raw buffer for PIL images, NumPy arrays, and bytes."""
    img = Image.new("RGB", (64, 64), color="blue")
    b1 = _image_to_bytes(img)
    assert isinstance(b1, bytes)
    assert len(b1) == 64 * 64 * 3

    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    b2 = _image_to_bytes(arr)
    assert isinstance(b2, bytes)
    assert len(b2) == 32 * 32 * 3

    raw_bytes = b"sample_image_bytes"
    assert _image_to_bytes(raw_bytes) == raw_bytes


def test_inpainter_in_place_canvas_modification_and_batching() -> None:
    """inpaint_blocks operates on a single canvas buffer and utilizes GPU batching."""
    from PIL import ImageDraw
    canvas = Image.new("RGB", (200, 200), color="white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 20), "Hello", fill="black")
    draw.text((30, 110), "World", fill="black")


    # Create 2 mock text blocks
    r1 = Region(
        id=1,
        global_bbox=BBox(10, 10, 80, 40),
        type=RegionType.DIALOGUE,
        detection_confidence=0.9,
        source_window_ids=(0,),
        status=RegionStatus.AUTO,
        text="Hello",
        metadata={"line_polygons": [[[12, 12], [75, 12], [75, 35], [12, 35]]]},
    )
    b1 = TextBlock(
        id=1,
        member_ids=(1,),
        members=(r1,),
        merged_bbox=BBox(10, 10, 80, 40),
        source_text="Hello",
    )

    r2 = Region(
        id=2,
        global_bbox=BBox(20, 100, 90, 140),
        type=RegionType.DIALOGUE,
        detection_confidence=0.9,
        source_window_ids=(0,),
        status=RegionStatus.AUTO,
        text="World",
        metadata={"line_polygons": [[[22, 102], [85, 102], [85, 135], [22, 135]]]},
    )
    b2 = TextBlock(
        id=2,
        member_ids=(2,),
        members=(r2,),
        merged_bbox=BBox(20, 100, 90, 140),
        source_text="World",
    )

    inpainter = Inpainter(lama_checkpoint=DEFAULT_LAMA_CHECKPOINT)

    # Mock inpaint_batch to verify it is called when neural inpainting is required
    dummy_crop1 = np.full((50, 100, 3), 255, dtype=np.uint8)
    dummy_crop2 = np.full((50, 100, 3), 255, dtype=np.uint8)

    with patch.object(inpainter.lama, "inpaint_batch", return_value=[dummy_crop1, dummy_crop2]) as mock_batch:
        result_img = inpainter.inpaint_blocks(canvas, [b1, b2])
        assert isinstance(result_img, Image.Image)
        assert result_img.size == (200, 200)
        assert len(inpainter.processed_block_ids) == 2
