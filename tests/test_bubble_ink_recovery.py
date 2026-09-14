"""Düz-balon mürekkep kurtarma: uzak küçük kelime maskeye katılır (sentetik)."""

import numpy as np
from PIL import Image
from unittest.mock import patch

from core.detection import BBox, Region, RegionStatus, RegionType
from core.detection.text_block import TextBlock
from core.imaging.inpainter import (
    Inpainter,
    _coverage_min_area,
    _expand_refined_to_bubble_ink,
    _mask_coverage_ok,
)
from core.imaging.text_mask import TextMask


def _story_member(rid, x1, y1, x2, y2):
    return Region(
        id=rid, global_bbox=BBox(x1, y1, x2, y2),
        type=RegionType.DIALOGUE, detection_confidence=0.9,
        source_window_ids=(1,), status=RegionStatus.AUTO,
        text="TEXT", metadata={},
    )


def _bbox_block(bid, members, x1, y1, x2, y2):
    return TextBlock(
        id=bid, member_ids=tuple(m.id for m in members), members=tuple(members),
        source_text="TEXT", merged_bbox=BBox(x1, y1, x2, y2),
    )


def _uniform_mask_with_far_word():
    # 60x200 düz beyaz balon: maske altta, IN üstte uzakta.
    h, w = 60, 200
    source = np.full((h, w, 3), 255, dtype=np.uint8)
    source[8:20, 20:50] = (0, 0, 0)      # uzak küçük kelime (12x30=360px)
    source[38:52, 90:170] = (0, 0, 0)    # ana yazı
    refined = np.zeros((h, w), dtype=np.uint8)
    refined[36:54, 88:172] = 255          # yalnız ana yazı maskede
    bubble = np.full((h, w), 255, dtype=np.uint8)
    return source, refined, bubble


def test_coverage_min_area_scales_with_bubble() -> None:
    small = _coverage_min_area(60 * 120)
    big = _coverage_min_area(800 * 400)
    assert small < big
    assert small >= 80  # toz tabanı
    # B19 toz sınıfı (53px) küçük balonda bile veto üretmez.
    assert small > 53


def test_far_word_recovered_in_uniform_bubble() -> None:
    src, ref, bub = _uniform_mask_with_far_word()
    assert _mask_coverage_ok(src, ref, (255, 255, 255), bub) is False
    grown = _expand_refined_to_bubble_ink(src, ref, (255, 255, 255), bub)
    assert int(np.count_nonzero((grown > 0) & ~(ref > 0))) > 0
    assert _mask_coverage_ok(src, grown, (255, 255, 255), bub) is True


def test_dust_and_lines_not_recovered() -> None:
    h, w = 60, 200
    source = np.full((h, w, 3), 255, dtype=np.uint8)
    source[10:12, 20:40] = (0, 0, 0)     # toz 2x20=40px < taban
    source[30:34, 10:150] = (0, 0, 0)    # çizgi 4x140 oran 35 > 6
    source[40:52, 90:170] = (0, 0, 0)    # ana yazı
    refined = np.zeros((h, w), dtype=np.uint8)
    refined[38:54, 88:172] = 255
    bubble = np.full((h, w), 255, dtype=np.uint8)
    grown = _expand_refined_to_bubble_ink(source, refined, (255, 255, 255), bubble)
    added = (grown > 0) & ~(refined > 0)
    # Toz ve çizgi katılmaz.
    assert int(np.count_nonzero(added[8:14, 15:45])) == 0
    assert int(np.count_nonzero(added[28:36, 5:155])) == 0


def test_uniform_block_passes_end_to_end() -> None:
    src, ref, bub = _uniform_mask_with_far_word()
    h, w = ref.shape
    tm = TextMask(
        crop_bbox=(0, 0, w, h), source=src.copy(), raw=ref.copy(),
        refined=ref.copy(), background_color=(255, 255, 255),
        is_uniform_background=True,
    )
    object.__setattr__(tm, "bubble_interior", bub)
    canvas = Image.fromarray(np.full((h, w, 3), 255, dtype=np.uint8), "RGB")
    # Canvas'a uzak kelime + ana yazıyı işle (blok kutusu tamamı).
    arr = np.asarray(canvas).copy()
    arr[8:20, 20:50] = (0, 0, 0)
    arr[38:52, 90:170] = (0, 0, 0)
    canvas = Image.fromarray(arr, "RGB")
    member = _story_member(101, 0, 0, w, h)
    block = _bbox_block(77, [member], 0, 0, w, h)
    inpainter = Inpainter()
    with patch.object(inpainter.mask_builder, "_build", return_value=tm):
        inpainter.inpaint_blocks(canvas, [block])
    assert 77 not in inpainter.review_block_ids


def test_non_uniform_block_stays_review() -> None:
    # Sanatlı balon: kurtarma çalışmaz, kapı veto eder (sanat korunur).
    src, ref, bub = _uniform_mask_with_far_word()
    # Zemine gradyan koy (tekdüze değil).
    grad = np.tile(np.linspace(0, 60, src.shape[1]).astype(np.uint8), (src.shape[0], 1))
    src = src.copy()
    for c in range(3):
        src[:, :, c] = np.clip(src[:, :, c].astype(int) - grad, 0, 255).astype(np.uint8)
    h, w = ref.shape
    tm = TextMask(
        crop_bbox=(0, 0, w, h), source=src.copy(), raw=ref.copy(),
        refined=ref.copy(), background_color=(255, 255, 255),
        is_uniform_background=False,
    )
    object.__setattr__(tm, "bubble_interior", bub)
    canvas = Image.fromarray(np.full((h, w, 3), 255, dtype=np.uint8), "RGB")
    member = _story_member(102, 0, 0, w, h)
    block = _bbox_block(78, [member], 0, 0, w, h)
    inpainter = Inpainter()
    with patch.object(inpainter.mask_builder, "_build", return_value=tm):
        with patch("core.imaging.inpainter._mask_coverage_ok", return_value=False):
            inpainter.inpaint_blocks(canvas, [block])
    assert 78 in inpainter.review_block_ids
    assert inpainter.review_causes.get(78) == "coverage"
