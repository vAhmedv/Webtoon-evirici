"""Düz-balon mürekkep kurtarma: uzak küçük kelime maskeye katılır (sentetik)."""

import numpy as np
from PIL import Image
from unittest.mock import patch

from core.detection import BBox, Region, RegionStatus, RegionType
from core.detection.text_block import TextBlock
from core.imaging.inpainter import (
    Inpainter,
    _absorb_nearband_ink,
    _absorb_nearblock_ink,
    _bubble_fill_safe,
    _catch_soft_halo,
    _coverage_min_area,
    _expand_refined_to_bubble_ink,
    _fill_uniform_bubble_interior,
    _lama_mask_for,
    _mask_coverage_ok,
)
from core.imaging.text_mask import TextMask, TextMaskBuilder, _yolo_box_interior


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
    huge = _coverage_min_area(10**6)
    assert small < big
    assert small >= 100  # r3 dersi: eski davranış tabanı (daraltma yok)
    assert huge == 300  # B19 boşluğu tavanı (toz 53 < 300 < M' 561)
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


def _halo_fixture():
    # Düz beyaz balon: ana yazı + bitişik soluk hale + uzak soluk toz.
    h, w = 80, 200
    source = np.full((h, w, 3), 255, dtype=np.uint8)
    source[30:55, 80:160] = (0, 0, 0)       # ana yazı
    source[28:30, 80:160] = (225, 225, 225)  # bitişik soluk hale (30 kontrast)
    source[5:8, 10:40] = (230, 230, 230)     # uzak soluk toz (alınmamalı)
    refined = np.zeros((h, w), dtype=np.uint8)
    refined[30:55, 80:160] = 255
    bubble = np.full((h, w), 255, dtype=np.uint8)
    return source, refined, bubble


def test_soft_halo_near_mask_is_caught() -> None:
    src, ref, bub = _halo_fixture()
    grown = _catch_soft_halo(src, ref, (255, 255, 255), bub)
    added = (grown > 0) & ~(ref > 0)
    assert int(np.count_nonzero(added[27:30, 80:160])) > 0


def test_soft_halo_far_dust_is_ignored() -> None:
    src, ref, bub = _halo_fixture()
    grown = _catch_soft_halo(src, ref, (255, 255, 255), bub)
    added = (grown > 0) & ~(ref > 0)
    assert int(np.count_nonzero(added[4:9, 8:42])) == 0


def test_soft_halo_needs_bubble() -> None:
    src, ref, _ = _halo_fixture()
    assert np.array_equal(_catch_soft_halo(src, ref, (255, 255, 255), None), ref)


def _rect_box_fixture():
    # Eksen-paralel anlatım kutusu: siyah çerçeve + içte yazı.
    import cv2

    h, w = 120, 200
    source = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(source, (20, 10), (180, 110), (0, 0, 0), 2)
    source[40:60, 50:150] = (0, 0, 0)  # yazı
    raw = np.zeros((h, w), dtype=np.uint8)
    raw[40:60, 50:150] = 255
    return source, raw


def test_rect_box_found_when_canny_misses() -> None:
    src, raw = _rect_box_fixture()
    bubble = TextMaskBuilder._extract_bubble(src, raw)
    assert bubble is not None and bool(np.any(bubble))
    # Yazı balonun içinde kalır.
    assert int(np.count_nonzero((raw > 0) & (bubble == 0))) == 0


def test_chamfered_octagon_box_found() -> None:
    # Köşesi kesik anlatım kutusu (TODAY sınıfı: 8 köşe).
    import cv2

    h, w = 120, 220
    src = np.full((h, w, 3), 255, dtype=np.uint8)
    pts = np.array([[40, 10], [180, 10], [200, 30], [200, 90], [180, 110], [40, 110], [20, 90], [20, 30]], np.int32)
    cv2.polylines(src, [pts], True, (0, 0, 0), 2)
    src[45:75, 60:160] = (0, 0, 0)
    raw = np.zeros((h, w), dtype=np.uint8)
    raw[45:75, 60:160] = 255
    bubble = TextMaskBuilder._extract_bubble(src, raw)
    assert bubble is not None and bool(np.any(bubble))
    assert int(np.count_nonzero((raw > 0) & (bubble == 0))) == 0


def test_no_box_no_bubble() -> None:
    src = np.full((60, 100, 3), 255, dtype=np.uint8)
    src[20:40, 30:70] = (0, 0, 0)
    raw = np.zeros((60, 100), dtype=np.uint8)
    raw[20:40, 30:70] = 255
    assert TextMaskBuilder._extract_bubble(src, raw) is None


def test_tight_frame_fallback_stays_quiet() -> None:    # Yazıya-yapışık çerçeve: YEDEK sessiz kalır (ana yolun eski davranışı
    # aynen durur; bu test yalnız yedeği bağlar).
    import cv2

    h, w = 60, 100
    src = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(src, (28, 18), (72, 42), (0, 0, 0), 1)
    src[22:38, 32:68] = (0, 0, 0)
    raw = np.zeros((h, w), dtype=np.uint8)
    raw[22:38, 32:68] = 255
    blurred = cv2.GaussianBlur(src, (3, 3), 0)
    edges = cv2.Canny(blurred, 70, 140, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    edges[raw > 0] = 0
    cv2.rectangle(edges, (0, 0), (w - 1, h - 1), 255, 1)
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    tp = cv2.findNonZero(raw)
    assert TextMaskBuilder._extract_rectangular_box(src, raw, contours, tp) is None


def _nearband_fixture():
    # Düz balon: ana maske + 4px ötede 8x20 artığı + 30px ötede artığı.
    h, w = 80, 200
    source = np.full((h, w, 3), 255, dtype=np.uint8)
    source[30:55, 80:140] = (0, 0, 0)    # ana yazı
    source[30:50, 144:152] = (0, 0, 0)   # yakın artık (4px, 8x20=160px)
    source[30:50, 175:183] = (0, 0, 0)   # uzak artık (35px)
    refined = np.zeros((h, w), dtype=np.uint8)
    refined[30:55, 80:140] = 255
    bubble = np.full((h, w), 255, dtype=np.uint8)
    return source, refined, bubble


def test_nearband_shard_absorbed() -> None:
    src, ref, bub = _nearband_fixture()
    grown = _absorb_nearband_ink(src, ref, (255, 255, 255), bub)
    added = (grown > 0) & ~(ref > 0)
    assert int(np.count_nonzero(added[28:52, 143:153])) > 0


def test_nearband_far_shard_ignored() -> None:
    src, ref, bub = _nearband_fixture()
    grown = _absorb_nearband_ink(src, ref, (255, 255, 255), bub)
    added = (grown > 0) & ~(ref > 0)
    assert int(np.count_nonzero(added[28:52, 174:184])) == 0


def test_nearband_needs_bubble() -> None:
    src, ref, _ = _nearband_fixture()
    assert np.array_equal(_absorb_nearband_ink(src, ref, (255, 255, 255), None), ref)


def test_uniform_interior_fill_covers_all_ink() -> None:
    # Düz balonda uzak kelime + hale + bant artığı: hepsi maskeye girer.
    h, w = 80, 220
    source = np.full((h, w, 3), 255, dtype=np.uint8)
    source[8:20, 20:50] = (0, 0, 0)      # uzak kelime
    source[38:52, 90:170] = (0, 0, 0)    # ana yazı
    source[36:38, 90:170] = (225, 225, 225)  # hale
    refined = np.zeros((h, w), dtype=np.uint8)
    refined[38:52, 90:170] = 255
    bubble = np.zeros((h, w), dtype=np.uint8)
    bubble[4:76, 10:210] = 255
    grown = _fill_uniform_bubble_interior(refined, bubble)
    assert int(np.count_nonzero((grown > 0)[8:20, 20:50])) > 0
    assert int(np.count_nonzero((grown > 0)[36:38, 90:170])) > 0
    # Çerçeve payı korunur (7x7 aşındırma): en dış 3px boyanmaz.
    assert int(np.count_nonzero((grown > 0)[0:3, :])) == 0
    assert int(np.count_nonzero((grown > 0)[:, 0:8])) == 0


def test_uniform_interior_fill_needs_bubble() -> None:
    ref = np.zeros((40, 40), dtype=np.uint8)
    ref[10:20, 10:20] = 255
    assert np.array_equal(_fill_uniform_bubble_interior(ref, None), ref)


def test_fill_safe_passes_white_box() -> None:
    src = np.full((80, 200, 3), 255, dtype=np.uint8)
    src[30:55, 80:140] = (0, 0, 0)
    bub = np.full((80, 200), 255, dtype=np.uint8)
    ok, color = _bubble_fill_safe(src, bub)
    assert ok is True
    assert color == (255, 255, 255)


def test_fill_safe_fails_gradient() -> None:
    src = np.zeros((80, 200, 3), dtype=np.uint8)
    for c in range(3):
        src[:, :, c] = np.tile(np.linspace(0, 200, 200).astype(np.uint8), (80, 1))
    bub = np.full((80, 200), 255, dtype=np.uint8)
    ok, _ = _bubble_fill_safe(src, bub)
    assert ok is False


def test_fill_safe_needs_bubble() -> None:
    src = np.full((40, 40, 3), 255, dtype=np.uint8)
    ok, _ = _bubble_fill_safe(src, None)
    assert ok is False


def _nearblock_fixture():
    # Balonsuz düz kırpıntı: ana yazı + 10px ötede artığı + 45px ötede artığı.
    h, w = 100, 220
    source = np.full((h, w, 3), 255, dtype=np.uint8)
    source[40:65, 80:160] = (0, 0, 0)    # ana yazı
    source[40:60, 164:172] = (0, 0, 0)   # yakın artık (4px, 8x20=160px)
    source[40:60, 205:213] = (0, 0, 0)   # uzak artık (45px)
    refined = np.zeros((h, w), dtype=np.uint8)
    refined[40:65, 80:160] = 255
    return source, refined


def test_nearblock_shard_absorbed() -> None:
    src, ref = _nearblock_fixture()
    grown = _absorb_nearblock_ink(src, ref, (255, 255, 255), 3)
    added = (grown > 0) & ~(ref > 0)
    assert int(np.count_nonzero(added[38:62, 163:173])) > 0


def test_nearblock_far_shard_ignored() -> None:
    src, ref = _nearblock_fixture()
    grown = _absorb_nearblock_ink(src, ref, (255, 255, 255), 3)
    added = (grown > 0) & ~(ref > 0)
    assert int(np.count_nonzero(added[38:62, 204:214])) == 0


def test_nearblock_art_band_ignored() -> None:
    # Bantta gerçek sanat (güçlü varyasyon) varsa emme yok.
    src, ref = _nearblock_fixture()
    grad = np.tile(np.linspace(0, 120, ref.shape[1]).astype(np.uint8), (ref.shape[0], 1))
    for c in range(3):
        src[:, :, c] = np.clip(src[:, :, c].astype(int) - grad, 0, 255).astype(np.uint8)
    grown = _absorb_nearblock_ink(src, ref, (255, 255, 255), 3)
    added = (grown > 0) & ~(ref > 0)
    assert int(np.count_nonzero(added[38:62, 163:173])) == 0


def test_lama_mask_clipped_to_bubble() -> None:
    base = np.zeros((60, 120), dtype=np.uint8)
    base[10:50, 10:110] = 255
    bub = np.zeros((60, 120), dtype=np.uint8)
    bub[10:50, 10:60] = 255  # balon yalnız sol yarı
    out = _lama_mask_for(base, bub)
    assert int(np.count_nonzero((out > 0)[:, 61:])) == 0
    assert int(np.count_nonzero((out > 0)[:, :60])) > 0


def test_lama_mask_footprint_preserved_without_bubble() -> None:
    import cv2

    base = np.zeros((120, 160), dtype=np.uint8)
    base[30:90, 40:120] = 255
    out = _lama_mask_for(base, None)
    from core.imaging.inpainter import lama_kernel_for_height

    kh = lama_kernel_for_height(120)
    plain = cv2.dilate(base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kh, kh)))
    diff = abs(int(np.count_nonzero(out)) - int(np.count_nonzero(plain)))
    assert diff / max(1, int(np.count_nonzero(plain))) < 0.05


def _yolo_crop():
    # 200x120 kırpıntı, ortada yazı; YOLO kutusu (global) balonu sarar.
    raw = np.zeros((120, 200), dtype=np.uint8)
    raw[45:75, 60:160] = 255
    return (1000, 2000, 1200, 2120), raw


def test_yolo_box_interior_found() -> None:
    crop, raw = _yolo_crop()
    # Kutu: yazıyı sarar, kırpıntıdan küçüktür.
    out = _yolo_box_interior(crop, raw, [(1040, 2030, 1180, 2110)])
    assert out is not None and bool(np.any(out))
    assert int(np.count_nonzero((raw > 0) & (out == 0))) == 0


def test_yolo_box_no_overlap_is_none() -> None:
    crop, raw = _yolo_crop()
    assert _yolo_box_interior(crop, raw, [(0, 0, 50, 50)]) is None
    assert _yolo_box_interior(crop, raw, []) is None
    assert _yolo_box_interior(crop, raw, None) is None


def test_yolo_page_box_rejected() -> None:
    # Sayfa-kadar kutu sahtekarlığı elenir (0.92 tavanı).
    crop, raw = _yolo_crop()
    assert _yolo_box_interior(crop, raw, [(1000, 2000, 1200, 2120)]) is None


def test_yolo_provider_fail_open_without_download() -> None:
    # Model yoksa indirme DENENMEZ, FileNotFoundError gelir.
    from providers.detector.yolo8_bubble import YoloBubbleDetector

    det = YoloBubbleDetector(model_path="yok/boyle/bir/model.pt")
    try:
        det.load()
        raise AssertionError("yuklenmemeliydi")
    except FileNotFoundError:
        pass
