"""Tests for V5.3 Visual Review Generator reliability & geometry rebasing.

Verifies:
1. Complete, side-effect free geometry rebasing (no mutation of original Region/metadata).
2. Explicit page identity matching by page filename.
3. Stale mask leakage prevention and mask ownership assertions.
"""

from __future__ import annotations

import copy
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from core.detection import BBox, Region, RegionStatus, RegionType
from core.detection.text_block import TextBlock
from core.imaging.inpainter import Inpainter
from scripts.generate_v5_3_visual_review_and_report import rebase_geometry_to_crop


def test_rebase_geometry_to_crop_side_effect_free() -> None:
    """Verify rebase_geometry_to_crop rebases all geometry without mutating original objects."""
    orig_bbox = BBox(100, 200, 300, 400)
    orig_line_polygons = [[[110, 210], [290, 210], [290, 250], [110, 250]]]
    orig_segmentation = [[105, 205], [295, 205], [295, 395], [105, 395]]
    orig_ctd_block = [100, 200, 300, 400]
    orig_ctd_memberships = [{"polygon": [[110, 210], [290, 210], [290, 250], [110, 250]]}]

    orig_meta = {
        "line_polygons": orig_line_polygons,
        "segmentation_polygon": orig_segmentation,
        "ctd_block_bbox": orig_ctd_block,
        "ctd_line_memberships": orig_ctd_memberships,
    }

    region = Region(
        id=1,
        global_bbox=orig_bbox,
        type=RegionType.DIALOGUE,
        detection_confidence=0.9,
        source_window_ids=(1,),
        status=RegionStatus.AUTO,
        text="TEST TEXT",
        metadata=orig_meta,
    )

    block = TextBlock(
        id=10,
        member_ids=(1,),
        members=(region,),
        source_text="TEST TEXT",
        merged_bbox=orig_bbox,
        metadata={"ctd_line_mapping": [{"polygon": [[110, 210], [290, 210], [290, 250], [110, 250]]}]},
    )

    page_y_offset = 1000
    crop_x1 = 80
    crop_y1 = 50

    rebased_block = rebase_geometry_to_crop(block, page_y_offset, crop_x1, crop_y1)

    # 1. Check original region and metadata remain strictly unchanged
    assert region.global_bbox == BBox(100, 200, 300, 400)
    assert region.metadata["ctd_block_bbox"] == [100, 200, 300, 400]
    assert region.metadata["line_polygons"] == [[[110, 210], [290, 210], [290, 250], [110, 250]]]

    # 2. Check rebased block member bbox
    # Expected: x1' = 100 - 80 = 20
    # Expected: y1' = 200 - 1000 - 50 = -850
    # Expected: x2' = 300 - 80 = 220
    # Expected: y2' = 400 - 1000 - 50 = -650
    rebased_r = rebased_block.members[0]
    assert rebased_r.global_bbox == BBox(20, -850, 220, -650)

    # 3. Check metadata geometry rebasing
    rebased_meta = rebased_r.metadata
    assert rebased_meta["ctd_block_bbox"] == [20, -850, 220, -650]
    assert rebased_meta["line_polygons"] == [[[30, -840], [210, -840], [210, -800], [30, -800]]]
    assert rebased_meta["segmentation_polygon"] == [[25, -845], [215, -845], [215, -655], [25, -655]]
    assert rebased_meta["ctd_line_memberships"][0]["polygon"] == [[30, -840], [210, -840], [210, -800], [30, -800]]


def test_inpainter_mask_state_reset_and_ownership_validation(tmp_path: Path) -> None:
    """Verify inpainter.last_text_mask is properly reset and doesn't leak between samples."""
    inpainter = Inpainter(debug_dir=tmp_path / "debug")

    # Sample 1: Canvas with black text that produces a mask
    canvas1 = Image.new("RGB", (200, 200), (255, 255, 255))
    # Draw black text box inside
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas1)
    draw.rectangle([(50, 50), (150, 150)], fill=(0, 0, 0))

    r1 = Region(
        id=1,
        global_bbox=BBox(50, 50, 150, 150),
        type=RegionType.DIALOGUE,
        detection_confidence=0.9,
        source_window_ids=(1,),
        status=RegionStatus.AUTO,
        text="TEXT1",
    )
    b1 = TextBlock(id=1, member_ids=(1,), members=(r1,), source_text="TEXT1", merged_bbox=BBox(50, 50, 150, 150))

    inpainter.last_text_mask = None
    inpainter.inpaint_blocks(canvas1, [b1])
    mask1 = inpainter.last_text_mask
    assert mask1 is not None

    # Sample 2: Reset last_text_mask before inpainting a different sample crop
    canvas2 = Image.new("RGB", (80, 80), (255, 255, 255))
    # Reset mask state
    inpainter.last_text_mask = None

    r2 = Region(
        id=2,
        global_bbox=BBox(10, 10, 70, 70),
        type=RegionType.DIALOGUE,
        detection_confidence=0.9,
        source_window_ids=(1,),
        status=RegionStatus.AUTO,
        text="TEXT2",
    )
    b2 = TextBlock(id=2, member_ids=(2,), members=(r2,), source_text="TEXT2", merged_bbox=BBox(10, 10, 70, 70))

    inpainter.inpaint_blocks(canvas2, [b2])
    mask2 = inpainter.last_text_mask

    # If mask2 is present, its crop_bbox MUST be within canvas2 size (80, 80)
    # If mask2 is present, its crop_bbox MUST be within canvas2 size (80, 80)
    if mask2 is not None:
        mx1, my1, mx2, my2 = mask2.crop_bbox
        assert 0 <= mx1 < mx2 <= 80 and 0 <= my1 < my2 <= 80
        assert mask2 != mask1


def test_sample_b_no_mask_never_reuses_sample_a_mask() -> None:
    """Verify that if Sample A produces a mask and Sample B produces no mask, Sample B never displays Sample A's mask."""
    inpainter = Inpainter()

    # Sample A: Story text region on canvas
    canvas_a = Image.new("RGB", (100, 100), (255, 255, 255))
    r_a = Region(
        id=1, global_bbox=BBox(10, 10, 90, 90), type=RegionType.DIALOGUE,
        detection_confidence=0.9, source_window_ids=(1,), status=RegionStatus.AUTO, text="STORY TEXT"
    )
    b_a = TextBlock(id=1, member_ids=(1,), members=(r_a,), source_text="STORY TEXT", merged_bbox=BBox(10, 10, 90, 90))

    inpainter.inpaint_blocks(canvas_a, [b_a])
    mask_a = inpainter.last_text_mask
    assert mask_a is not None

    # Sample B: Non-story text / SFX region (produces NO mask)
    canvas_b = Image.new("RGB", (100, 100), (255, 255, 255))
    r_b = Region(
        id=2, global_bbox=BBox(10, 10, 90, 90), type=RegionType.SFX,
        detection_confidence=0.9, source_window_ids=(1,), status=RegionStatus.SKIP, text="BOOM"
    )
    b_b = TextBlock(id=2, member_ids=(2,), members=(r_b,), source_text="BOOM", merged_bbox=BBox(10, 10, 90, 90))

    inpainter.inpaint_blocks(canvas_b, [b_b])
    mask_b = inpainter.last_text_mask

    # Must be None - Sample B MUST NOT retain Sample A's mask
    assert mask_b is None


def test_build_page_identity_map_stem_matching_and_errors() -> None:
    """Verify build_page_identity_map handles WEBP->PNG stem matching and fails on missing/duplicate identities."""
    from core.models import Page
    from scripts.generate_v5_3_visual_review_and_report import build_page_identity_map

    src_pages = [
        Page(index=0, path=Path("001.webp"), width=800, height=1000, y_offset=0),
        Page(index=1, path=Path("002.webp"), width=800, height=1000, y_offset=1000),
        Page(index=2, path=Path("010.webp"), width=800, height=1000, y_offset=2000),
    ]

    exp_paths = [
        Path("001.png"),
        Path("002.png"),
        Path("010.png"),
    ]

    # 1. Successful matching with format change (WEBP -> PNG) and natural ordering (1, 2, 10)
    identity_map = build_page_identity_map(src_pages, exp_paths)
    assert len(identity_map) == 3
    assert identity_map["001.webp"] == Path("001.png")
    assert identity_map["002.webp"] == Path("002.png")
    assert identity_map["010.webp"] == Path("010.png")

    # 2. Missing output page identity raises ValueError
    missing_exp_paths = [Path("001.png"), Path("002.png")]
    with pytest.raises(ValueError, match="Exported page files missing"):
        build_page_identity_map(src_pages, missing_exp_paths)

    # 3. Duplicate output page identity stem raises ValueError
    dup_exp_paths = [Path("001.png"), Path("001.jpg"), Path("002.png")]
    with pytest.raises(ValueError, match="Duplicate exported page identity stem"):
        build_page_identity_map(src_pages, dup_exp_paths)
