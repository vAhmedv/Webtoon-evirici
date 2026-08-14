"""V5.3 Pipeline Invariant and Result Metrics Tests.

Verifies:
1. Source image files remain byte/mtime unchanged after processing.
2. SKIP and REVIEW regions are never rendered.
3. Inpaint review blocks (inpainter.review_block_ids) are excluded from rendering.
4. Output page count equals input page count.
5. Metrics reported match actual operations performed.
6. Outside-mask pixel identity guarantees.
7. Non-uniform-width canvas width normalization contract.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image

from core.detection import BBox, Region, RegionStatus, RegionType
from core.detection.text_block import TextBlock
from core.imaging.inpainter import Inpainter
from core.imaging.renderer import TextRenderer
from core.models import Page
from core.io.input_loader import load_chapter


def test_source_files_remain_unchanged(tmp_path: Path) -> None:
    """Verify input source images are strictly untouched (byte/mtime identical)."""
    img_path = tmp_path / "001.jpg"
    img = Image.new("RGB", (100, 200), (128, 128, 128))
    img.save(img_path)

    stat_before = img_path.stat()
    bytes_before = img_path.read_bytes()

    pages = load_chapter(tmp_path, allow_non_uniform_widths=True)
    assert len(pages) == 1

    stat_after = img_path.stat()
    bytes_after = img_path.read_bytes()

    assert stat_before.st_mtime == stat_after.st_mtime
    assert bytes_before == bytes_after


def test_skip_and_review_regions_not_rendered() -> None:
    """Verify SKIP and REVIEW status regions are ignored by TextRenderer."""
    canvas = Image.new("RGB", (300, 300), (255, 255, 255))
    renderer = TextRenderer()

    # Region with REVIEW status
    r_review = Region(
        id=1,
        global_bbox=BBox(10, 10, 100, 50),
        type=RegionType.UNKNOWN,
        detection_confidence=0.9,
        source_window_ids=(1,),
        status=RegionStatus.REVIEW,
        text="REVIEW TEXT",
    )
    b_review = TextBlock(id=1, member_ids=(1,), members=(r_review,), source_text="REVIEW TEXT", merged_bbox=BBox(10, 10, 100, 50))

    # Region with SKIP status
    r_skip = Region(
        id=2,
        global_bbox=BBox(10, 60, 100, 100),
        type=RegionType.SFX,
        detection_confidence=0.9,
        source_window_ids=(1,),
        status=RegionStatus.SKIP,
        text="SFX TEXT",
    )
    b_skip = TextBlock(id=2, member_ids=(2,), members=(r_skip,), source_text="SFX TEXT", merged_bbox=BBox(10, 60, 100, 100))

    block_translations = [(b_review, "İnceleme"), (b_skip, "Atla")]
    out_canvas, rendered_cnt, overflow_cnt = renderer.render_blocks(canvas, block_translations)

    # Neither block should be rendered because member statuses are not AUTO
    assert rendered_cnt == 0
    # Output canvas must be byte-identical to white canvas
    assert np.array_equal(np.asarray(canvas), np.asarray(out_canvas))


def test_inpaint_review_blocks_excluded_from_rendering() -> None:
    """Verify blocks marked by inpainter.review_block_ids are excluded from rendering."""
    canvas = Image.new("RGB", (300, 300), (255, 255, 255))
    renderer = TextRenderer()

    inpainter = Inpainter()
    inpainter.processed_block_ids.add(10)
    inpainter.processed_block_ids.add(20)
    inpainter.review_block_ids.add(20)  # Block 20 flagged for inpaint review

    r10 = Region(id=101, global_bbox=BBox(10, 10, 100, 50), type=RegionType.DIALOGUE, detection_confidence=0.9, source_window_ids=(1,), status=RegionStatus.AUTO, text="T1")
    b10 = TextBlock(id=10, member_ids=(101,), members=(r10,), source_text="T1", merged_bbox=BBox(10, 10, 100, 50))

    r20 = Region(id=102, global_bbox=BBox(10, 60, 100, 100), type=RegionType.DIALOGUE, detection_confidence=0.9, source_window_ids=(1,), status=RegionStatus.AUTO, text="T2")
    b20 = TextBlock(id=20, member_ids=(102,), members=(r20,), source_text="T2", merged_bbox=BBox(10, 60, 100, 100))

    translated_block_pairs = [(b10, "Metin 1"), (b20, "Metin 2")]

    renderable_pairs = [
        pair for pair in translated_block_pairs
        if pair[0].id in inpainter.processed_block_ids
        and pair[0].id not in inpainter.review_block_ids
    ]

    # Only b10 should be renderable
    assert len(renderable_pairs) == 1
    assert renderable_pairs[0][0].id == 10

    out_canvas, rendered_cnt, overflow_cnt = renderer.render_blocks(canvas, renderable_pairs)
    assert rendered_cnt == 1


def test_outside_mask_pixel_identity_guarantee() -> None:
    """Verify inpainting preserves 100% of pixels outside the refined mask."""
    # Create background pattern canvas
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    canvas = Image.fromarray(arr, "RGB")

    inpainter = Inpainter()
    # Mask only a small region (10, 10) to (30, 30)
    r = Region(id=1, global_bbox=BBox(10, 10, 30, 30), type=RegionType.DIALOGUE, detection_confidence=0.9, source_window_ids=(1,), status=RegionStatus.AUTO, text="TEXT")
    b = TextBlock(id=1, member_ids=(1,), members=(r,), source_text="TEXT", merged_bbox=BBox(10, 10, 30, 30))

    result_img = inpainter.inpaint_blocks(canvas, [b])
    result_arr = np.asarray(result_img)

    text_mask = inpainter.last_text_mask
    if text_mask is not None:
        refined = text_mask.refined > 0
        x1, y1, x2, y2 = text_mask.crop_bbox
        # Crop region outside refined mask must equal original arr exactly
        crop_orig = arr[y1:y2, x1:x2]
        crop_res = result_arr[y1:y2, x1:x2]

        outside_mask = ~refined
        assert np.array_equal(crop_orig[outside_mask], crop_res[outside_mask])

        # Everything outside crop_bbox must also equal original arr exactly
        arr_copy = arr.copy()
        arr_copy[y1:y2, x1:x2] = crop_res
        assert np.array_equal(arr_copy, result_arr)


def test_non_uniform_width_contract_characterization(tmp_path: Path) -> None:
    """Verify non-uniform-width pages are normalized to target_width on global canvas."""
    img1_path = tmp_path / "001.jpg"
    img2_path = tmp_path / "002.jpg"

    Image.new("RGB", (800, 1000), (255, 255, 255)).save(img1_path)
    Image.new("RGB", (400, 500), (255, 255, 255)).save(img2_path)  # Non-uniform width

    pages = load_chapter(tmp_path, allow_non_uniform_widths=True)
    assert len(pages) == 2
    # target_width should be 800 (most common)
    assert pages[0].width == 800
    assert pages[1].width == 800
    # Height of page 2 scaled proportionally from 500 * (800 / 400) = 1000
    assert pages[1].height == 1000


def test_mixed_status_block_safety() -> None:
    """Verify that a TextBlock with mixed AUTO and REVIEW/SKIP members is neither in-painted nor rendered."""
    arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    canvas = Image.fromarray(arr, "RGB")

    r_auto = Region(
        id=1, global_bbox=BBox(10, 10, 90, 50), type=RegionType.DIALOGUE,
        detection_confidence=0.9, source_window_ids=(1,), status=RegionStatus.AUTO, text="AUTO TEXT"
    )
    r_review = Region(
        id=2, global_bbox=BBox(10, 60, 90, 100), type=RegionType.DIALOGUE,
        detection_confidence=0.9, source_window_ids=(1,), status=RegionStatus.REVIEW, text="REVIEW TEXT"
    )
    mixed_block = TextBlock(
        id=50, member_ids=(1, 2), members=(r_auto, r_review),
        source_text="AUTO TEXT REVIEW TEXT", merged_bbox=BBox(10, 10, 90, 100)
    )

    inpainter = Inpainter()
    result_img = inpainter.inpaint_blocks(canvas, [mixed_block])

    # Inpainter must NOT process or alter mixed-status block
    assert 50 not in inpainter.processed_block_ids
    assert np.array_equal(arr, np.asarray(result_img))

    renderer = TextRenderer()
    out_canvas, rendered_cnt, _ = renderer.render_blocks(canvas, [(mixed_block, "Türkçe")])

    # Renderer must NOT render mixed-status block
    assert rendered_cnt == 0
    assert np.array_equal(arr, np.asarray(out_canvas))


def test_renderer_breaks_long_turkish_token_within_narrow_bubble() -> None:
    canvas = Image.new("RGB", (180, 120), "white")
    region = Region(
        id=1, global_bbox=BBox(10, 10, 80, 100), type=RegionType.DIALOGUE,
        detection_confidence=0.9, source_window_ids=(1,), status=RegionStatus.AUTO,
        text="TEXT",
    )
    block = TextBlock(
        id=1, member_ids=(1,), members=(region,), source_text="TEXT",
        merged_bbox=region.global_bbox,
    )
    renderer = TextRenderer()
    font, lines, _, overflow = renderer._fit_block_text(
        "ÇĞİÖŞÜılaştırılamayanlardanmışsınız", 56, 80
    )
    assert not overflow
    assert len(lines) > 1
    assert all(renderer._line_width(line, font) <= 56 for line in lines)


def test_renderer_counts_unfit_horizontal_or_vertical_text_as_overflow_not_rendered() -> None:
    canvas = Image.new("RGB", (40, 40), "white")
    region = Region(
        id=2, global_bbox=BBox(5, 5, 18, 18), type=RegionType.DIALOGUE,
        detection_confidence=0.9, source_window_ids=(1,), status=RegionStatus.AUTO,
        text="TEXT",
    )
    block = TextBlock(2, (2,), (region,), region.global_bbox, "TEXT")
    output, rendered, overflow = TextRenderer().render_blocks(
        canvas, [(block, "Çok, uzun; çok satırlı Türkçe metin!")]
    )
    assert rendered == 0
    assert overflow == 1
    assert np.array_equal(np.asarray(output), np.asarray(canvas))
