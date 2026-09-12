"""Faz 4 inpaint guard testleri: tamamı sentetik numpy, model/GPU yok.

Anti-overfit: hiçbir test bölüm/sayfa verisi kullanmaz; eşikler göreli
oranlardır (çekirdek = maske yüksekliğinin işlevi, kontrast = medyandan sapma).
"""

import numpy as np

from core.imaging.inpainter import (
    FILL_RING_LUMA_GAP,
    GHOST_MIN_COMPONENT_AREA,
    Inpainter,
    lama_kernel_for_height,
)
from core.imaging.text_mask import TextMask


def _mask(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> TextMask:
    refined = np.zeros((h, w), dtype=np.uint8)
    refined[y1:y2, x1:x2] = 255
    source = np.full((h, w, 3), 255, dtype=np.uint8)
    return TextMask(
        crop_bbox=(0, 0, w, h),
        source=source,
        raw=refined.copy(),
        refined=refined,
        background_color=(255, 255, 255),
        is_uniform_background=True,
    )


def test_lama_kernel_scales_with_height() -> None:
    assert lama_kernel_for_height(10) == 3
    assert lama_kernel_for_height(0) == 3
    small = lama_kernel_for_height(40)
    big = lama_kernel_for_height(300)
    huge = lama_kernel_for_height(5000)
    assert small % 2 == 1 and big % 2 == 1 and huge % 2 == 1
    assert 3 <= small < big <= huge <= 21
    assert huge == 21


def test_interior_ghost_clean_uniform() -> None:
    crop = np.full((60, 100, 3), 255, dtype=np.uint8)
    mask = np.zeros((60, 100), dtype=np.uint8)
    mask[10:50, 10:90] = 255
    assert Inpainter._interior_ghost_area(crop, mask) == 0


def test_interior_ghost_flags_glyph_remnant() -> None:
    crop = np.full((60, 100, 3), 255, dtype=np.uint8)
    crop[20:40, 30:50] = (0, 0, 0)  # 20x20 = 400px siyah artık
    mask = np.zeros((60, 100), dtype=np.uint8)
    mask[10:50, 10:90] = 255
    assert Inpainter._interior_ghost_area(crop, mask) >= GHOST_MIN_COMPONENT_AREA


def test_interior_ghost_ignores_specks() -> None:
    crop = np.full((60, 100, 3), 255, dtype=np.uint8)
    crop[20:22, 30:32] = (0, 0, 0)  # 2x2 = 4px < eşik
    mask = np.zeros((60, 100), dtype=np.uint8)
    mask[10:50, 10:90] = 255
    assert Inpainter._interior_ghost_area(crop, mask) == 0


def test_interior_ghost_flags_scattered_specks() -> None:
    """P005 vakası: tekil eşik altı ama toplamda kirleten zerreler."""
    crop = np.full((60, 100, 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(7)
    for _ in range(25):  # 25x 2x2 zerre = 100px toplam
        y, x = int(rng.integers(12, 45)), int(rng.integers(12, 85))
        crop[y : y + 2, x : x + 2] = (0, 0, 0)
    mask = np.zeros((60, 100), dtype=np.uint8)
    mask[10:50, 10:90] = 255
    assert Inpainter._interior_ghost_area(crop, mask) >= 60


def test_fill_ring_mismatch_white_blob_on_dark() -> None:
    """P003 vakası: balonsuz beyaz dolgu, koyu çevre → REVIEW."""
    h, w = 120, 160
    source = np.full((h, w, 3), 25, dtype=np.uint8)  # koyu zemin
    refined = np.zeros((h, w), dtype=np.uint8)
    refined[30:90, 40:120] = 255
    inpainted = source.copy()
    inpainted[refined > 0] = (245, 245, 245)  # beyaz leke
    tm = TextMask(
        crop_bbox=(0, 0, w, h),
        source=source,
        raw=refined.copy(),
        refined=refined,
        background_color=(245, 245, 245),
        is_uniform_background=False,
    )
    assert Inpainter._fill_ring_mismatch(tm, inpainted) is True


def test_fill_ring_mismatch_white_bubble_clean() -> None:
    """Beyaz balon + beyaz dolgu + açık çevre → temiz."""
    tm = _mask(120, 160, 40, 30, 120, 90)
    inpainted = np.full((120, 160, 3), 255, dtype=np.uint8)
    assert Inpainter._fill_ring_mismatch(tm, inpainted) is False


def test_fill_ring_mismatch_bubble_found_skips() -> None:
    """Balon bulunduysa dolgu balon içindedir — denetim çalışmaz."""
    tm = _mask(120, 160, 40, 30, 120, 90)
    bubble = np.zeros((120, 160), dtype=np.uint8)
    bubble[20:100, 30:130] = 255
    object.__setattr__(tm, "bubble_interior", bubble)
    inpainted = np.full((120, 160, 3), 25, dtype=np.uint8)
    inpainted[30:90, 40:120] = (245, 245, 245)
    assert Inpainter._fill_ring_mismatch(tm, inpainted) is False


def test_fill_ring_gap_constant_sane() -> None:
    assert 30 <= FILL_RING_LUMA_GAP <= 120


def _member_with_flag(flag: bool):
    from core.detection import BBox, Region, RegionStatus, RegionType

    return Region(
        id=1,
        global_bbox=BBox(0, 0, 50, 30),
        type=RegionType.UNKNOWN,
        detection_confidence=0.38,
        source_window_ids=(1,),
        status=RegionStatus.REVIEW,
        text="X",
        metadata={"second_chance": flag},
    )


def test_is_second_chance_block() -> None:
    from core.imaging.inpainter import _is_second_chance_block
    from core.detection.text_block import TextBlock
    from core.detection import BBox

    def _block(*flags: bool) -> TextBlock:
        members = tuple(_member_with_flag(f) for f in flags)
        return TextBlock(
            id=1, member_ids=tuple(m.id for m in members), members=members,
            merged_bbox=BBox(0, 0, 50, 30), source_text="X",
        )

    assert _is_second_chance_block(_block(True))
    assert _is_second_chance_block(_block(True, True))
    assert not _is_second_chance_block(_block(True, False))
    assert not _is_second_chance_block(_block(False))


def test_expand_mask_in_bubble_grows_bounded() -> None:
    from core.imaging.inpainter import Inpainter

    tm = _mask(120, 160, 60, 50, 100, 70)  # 40x20 maske
    bubble = np.zeros((120, 160), dtype=np.uint8)
    bubble[10:110, 20:140] = 255  # geniş balon
    object.__setattr__(tm, "bubble_interior", bubble)
    before = int(np.count_nonzero(tm.refined))
    grown = Inpainter._expand_mask_in_bubble(tm)
    after = int(np.count_nonzero(grown.refined))
    assert after > before
    # Balon dışına taşmaz.
    assert bool(np.all((grown.refined == 0) | (bubble > 0)))


def test_expand_mask_without_bubble_grows_on_bg_color() -> None:
    """Balon yoksa bile zemin-rengi alanda büyür (DAMMIT vakası)."""
    from core.imaging.inpainter import Inpainter

    tm = _mask(120, 160, 60, 50, 100, 70)
    before = int(np.count_nonzero(tm.refined))
    grown = Inpainter._expand_mask_in_bubble(tm)
    assert int(np.count_nonzero(grown.refined)) > before


def test_expand_mask_keeps_glyph_cores() -> None:
    """Büyüme glif çekirdeklerini eksiltmez (siyah glif zemin-rengi değil)."""
    from core.imaging.inpainter import Inpainter

    tm = _mask(120, 160, 40, 30, 120, 90)
    refined = np.zeros((120, 160), dtype=np.uint8)
    refined[40:80, 50:110] = 255
    object.__setattr__(tm, "refined", refined)
    src = np.full((120, 160, 3), 255, dtype=np.uint8)
    src[50:70, 60:100] = (0, 0, 0)  # glif çekirdeği
    object.__setattr__(tm, "source", src)
    grown = Inpainter._expand_mask_in_bubble(tm)
    assert bool(np.all(grown.refined[50:70, 60:100] > 0))


def test_expand_mask_blocked_on_dark_art() -> None:
    """Koyu sanatta büyüme durur (P003 beyaz-leke riski yok)."""
    from core.imaging.inpainter import Inpainter

    tm = _mask(120, 160, 60, 50, 100, 70)
    dark = np.full((120, 160, 3), 25, dtype=np.uint8)
    object.__setattr__(tm, "source", dark)
    grown = Inpainter._expand_mask_in_bubble(tm)
    assert np.array_equal(grown.refined, tm.refined)
