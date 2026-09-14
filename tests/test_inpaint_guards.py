"""Faz 4 inpaint guard testleri: tamamı sentetik numpy, model/GPU yok.

Anti-overfit: hiçbir test bölüm/sayfa verisi kullanmaz; eşikler göreli
oranlardır (çekirdek = maske yüksekliğinin işlevi, kontrast = medyandan sapma).
"""

import numpy as np
from pathlib import Path
from PIL import Image

from core.detection import BBox, Region, RegionStatus, RegionType
from core.detection.text_block import TextBlock
from core.imaging.inpainter import (
    FILL_RING_LUMA_GAP,
    GHOST_MIN_COMPONENT_AREA,
    Inpainter,
    _mask_coverage_ok,
    _outside_mask_text,
    lama_kernel_for_height,
)
from core.imaging.text_mask import TextMask

_FIXTURE_B19 = Path(__file__).resolve().parent / "fixtures" / "inpaint_b19"


def _fixture_png(name: str) -> np.ndarray:
    return np.array(Image.open(_FIXTURE_B19 / name).convert("RGB"), dtype=np.uint8)


def _fixture_mask(name: str) -> np.ndarray:
    return np.array(Image.open(_FIXTURE_B19 / name).convert("L"))


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

    tm = _mask(120, 160, 40, 30, 100, 70)
    seed = np.zeros((120, 160), dtype=np.uint8)
    seed[40:80, 50:110] = 255  # 60x40 = 2400px tohum
    object.__setattr__(tm, "refined", seed)
    bubble = np.zeros((120, 160), dtype=np.uint8)
    bubble[20:100, 30:130] = 255  # 100x80 balon
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

    tm = _mask(120, 160, 40, 30, 100, 70)
    seed = np.zeros((120, 160), dtype=np.uint8)
    seed[40:80, 50:110] = 255
    object.__setattr__(tm, "refined", seed)
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
    """Tekdüze koyu zeminde taşma zararsızdır (no-op); parlak nesneye girmez."""
    from core.imaging.inpainter import Inpainter

    src = np.full((120, 160, 3), 25, dtype=np.uint8)
    src[90:110, 120:150] = (240, 240, 255)  # parlak ışıma nesnesi
    tm = TextMask(
        crop_bbox=(0, 0, 160, 120),
        source=src,
        raw=np.zeros((120, 160), dtype=np.uint8),
        refined=np.zeros((120, 160), dtype=np.uint8),
        background_color=(25, 25, 25),
        is_uniform_background=False,
    )
    seed = np.zeros((120, 160), dtype=np.uint8)
    seed[40:80, 50:110] = 255
    object.__setattr__(tm, "refined", seed)
    grown = Inpainter._expand_mask_in_bubble(tm)
    # Parlak nesneye taşmaz (P003 koruması).
    assert bool(np.all(grown.refined[90:110, 120:150] == 0))


def test_flood_fills_white_region() -> None:
    """Tohumdan beyaz-bitişik bölgenin tamamı kapsanır."""
    from core.imaging.inpainter import Inpainter

    src = np.full((120, 160, 3), 255, dtype=np.uint8)
    src[:, :20] = (10, 10, 10)  # solda koyu sanat şeridi
    src[50:60, 60:100] = (0, 0, 0)  # tohum içi glif çekirdeği
    tm = TextMask(
        crop_bbox=(0, 0, 160, 120),
        source=src,
        raw=np.zeros((120, 160), dtype=np.uint8),
        refined=np.zeros((120, 160), dtype=np.uint8),
        background_color=(255, 255, 255),
        is_uniform_background=False,
    )
    seed = np.zeros((120, 160), dtype=np.uint8)
    seed[40:80, 50:110] = 255
    object.__setattr__(tm, "refined", seed)
    grown = Inpainter._expand_mask_in_bubble(tm)
    assert int(np.count_nonzero(grown.refined)) > 5000
    # Koyu şeride taşmaz.
    assert bool(np.all(grown.refined[:, :20] == 0))
    # Glif çekirdeği korunur.
    assert bool(np.all(grown.refined[50:60, 60:100] > 0))


def test_flood_ignores_poisoned_background_color() -> None:
    """background_color koyu zehirliyse bile çevre medyanı kazanır (DAMMIT)."""
    from core.imaging.inpainter import Inpainter

    src = np.full((120, 160, 3), 255, dtype=np.uint8)
    tm = TextMask(
        crop_bbox=(0, 0, 160, 120),
        source=src,
        raw=np.zeros((120, 160), dtype=np.uint8),
        refined=np.zeros((120, 160), dtype=np.uint8),
        background_color=(10, 10, 10),  # zehirli: glif pikselinden gelmiş
        is_uniform_background=False,
    )
    seed = np.zeros((120, 160), dtype=np.uint8)
    seed[40:80, 50:110] = 255
    object.__setattr__(tm, "refined", seed)
    grown = Inpainter._expand_mask_in_bubble(tm)
    assert int(np.count_nonzero(grown.refined)) > int(np.count_nonzero(seed))


def test_expand_mask_stays_in_seed_compartment() -> None:
    """FAZ 4-KISIT kilidi (İŞ 1 sonucu): kısmi-kutu tohumu glif bariyerini
    aşamaz — flood tohumun beyaz kompartımanında kalır, komşu sanat korunur.

    Uzak glifler (DAMMIT `D`/`IT`) maske katmanıyla kapsanamaz
    (detector-recall işi); taşma dürüstçe sınırlı kalır, sessiz leke yok.
    """
    from core.imaging.inpainter import Inpainter

    from core.imaging.text_mask import TextMask

    h, w = 60, 120
    src = np.full((h, w, 3), 255, dtype=np.uint8)
    src[:, 58:62] = (0, 0, 0)  # 4px siyah glif bariyeri
    tm = TextMask(
        crop_bbox=(0, 0, w, h),
        source=src,
        raw=np.zeros((h, w), dtype=np.uint8),
        refined=np.zeros((h, w), dtype=np.uint8),
        background_color=(255, 255, 255),
        is_uniform_background=False,
    )
    seed = np.zeros((h, w), dtype=np.uint8)
    seed[20:40, 64:90] = 255  # bariyerin sağında beyaz tohum
    object.__setattr__(tm, "refined", seed)
    grown = Inpainter._expand_mask_in_bubble(tm)
    grown_mask = grown.refined > 0
    # Sağ kompartıman dolar, sol kompartımana (x<58) taşma yok.
    assert int(np.count_nonzero(grown_mask[:, 63:])) > int(np.count_nonzero(seed))
    assert int(np.count_nonzero(grown_mask[:, :57])) == 0


def test_flood_area_cap_trips_on_runaway() -> None:
    """40× üstü taşmada tavan devreye girer (hesapsal kaçak sigortası)."""
    from core.imaging.inpainter import Inpainter

    tm = _mask(120, 160, 40, 30, 100, 70)
    seed = np.zeros((120, 160), dtype=np.uint8)
    seed[55:65, 75:85] = 255  # 10x10 = 100px; beyaz bölge 19200 = 192×
    object.__setattr__(tm, "refined", seed)
    grown = Inpainter._expand_mask_in_bubble(tm)
    assert np.array_equal(grown.refined, tm.refined)


def test_outside_mask_band_catches_b19_shard() -> None:
    """S4 B19-kanıtı: maske-dışında kalan "M'" artığı bantta yakalanır."""
    source = _fixture_png("source.png")
    inpainted = _fixture_png("inpainted.png")
    refined = _fixture_mask("refined_mask.png")
    assert _outside_mask_text(source, inpainted, refined, (255, 255, 255)) is True


def test_outside_mask_band_clean_crop_passes() -> None:
    crop = np.full((60, 100, 3), 255, dtype=np.uint8)
    mask = np.zeros((60, 100), dtype=np.uint8)
    mask[10:50, 30:90] = 255
    assert _outside_mask_text(crop, crop, mask, (255, 255, 255)) is False


def test_outside_mask_band_near_glyph_flags() -> None:
    base_mask = np.zeros((60, 120), dtype=np.uint8)
    base_mask[10:50, 60:110] = 255
    near = np.full((60, 120, 3), 255, dtype=np.uint8)
    near[20:40, 48:56] = (0, 0, 0)  # maske kenarının 4px solunda
    assert _outside_mask_text(near, near, base_mask, (255, 255, 255)) is True


def test_outside_mask_band_far_glyph_out_of_scope() -> None:
    # Bant-dışı uzak leke bu fonksiyonun işi değil (dedektör-erişim
    # sorunu) — sessiz False, şişirme yok.
    base_mask = np.zeros((60, 120), dtype=np.uint8)
    base_mask[10:50, 60:110] = 255
    far = np.full((60, 120, 3), 255, dtype=np.uint8)
    far[20:40, 10:18] = (0, 0, 0)
    assert _outside_mask_text(far, far, base_mask, (255, 255, 255)) is False


def test_outside_mask_band_ignores_huge_art() -> None:
    crop = np.full((60, 120, 3), 255, dtype=np.uint8)
    crop[0:60, 0:40] = (0, 0, 0)  # kırpıntının 1/3ünden büyük yapı
    mask = np.zeros((60, 120), dtype=np.uint8)
    mask[10:50, 60:110] = 255
    assert _outside_mask_text(crop, crop, mask, (255, 255, 255)) is False


def test_outside_mask_band_ignores_dark_source_art() -> None:
    # Balon-dışı kaya dokusu: kaynakta koyu → bant elenir (dungeon b10
    # sınıfı sel düzeltmesi).
    art = np.full((60, 120, 3), 255, dtype=np.uint8)
    art[0:60, 0:30] = (90, 80, 110)  # koyu doku (kaynak + bitmiş aynı)
    art_mask = np.zeros((60, 120), dtype=np.uint8)
    art_mask[10:50, 60:110] = 255
    assert _outside_mask_text(art, art, art_mask, (255, 255, 255)) is False


def _story_member(rid: int, x1: int, y1: int, x2: int, y2: int) -> Region:
    return Region(
        id=rid,
        global_bbox=BBox(x1, y1, x2, y2),
        type=RegionType.DIALOGUE,
        detection_confidence=0.9,
        source_window_ids=(1,),
        status=RegionStatus.AUTO,
        text="TEXT",
        metadata={},
    )


def _bbox_block(bid: int, members: list[Region], x1: int, y1: int, x2: int, y2: int) -> TextBlock:
    return TextBlock(
        id=bid,
        member_ids=tuple(m.id for m in members),
        members=tuple(members),
        source_text="TEXT",
        merged_bbox=BBox(x1, y1, x2, y2),
    )


def test_review_revert_and_alias_guard() -> None:
    """P1: REVIEW geri-alması orijinali yazar; çakışan kırpıntı bakışı bozamaz."""
    arr = np.full((120, 120, 3), 255, dtype=np.uint8)
    arr[52:58, 10:110] = (0, 0, 0)  # X yazısı (uzun çubuk, oran>6 → elenir)
    arr[62:72, 62:108] = (0, 0, 0)  # Y yazısı (üye kutusu içi)
    arr[62:82, 46:54] = (0, 0, 0)  # Y dış artığı (6px ötede, 8x20=160px)
    canvas = Image.fromarray(arr, "RGB")
    x_block = _bbox_block(8, [_story_member(81, 10, 50, 110, 60)], 0, 50, 120, 60)
    y_block = _bbox_block(9, [_story_member(91, 60, 60, 110, 80)], 0, 60, 120, 80)

    inpainter = Inpainter()
    out = np.asarray(inpainter.inpaint_blocks(canvas, [x_block, y_block]))
    assert 9 in inpainter.review_block_ids
    assert 8 not in inpainter.review_block_ids
    # X temizlendi (beyaz), Y sapasağlam geri alındı (orijinal İngilizce).
    assert bool(np.all(out[52:58, 10:110] == 255))
    assert bool(np.all(out[62:72, 62:108] == 0))
    assert bool(np.all(out[62:82, 46:54] == 0))


def _interior_full(h: int, w: int) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def test_coverage_ok_when_mask_covers_ink() -> None:
    crop = np.full((60, 120, 3), 255, dtype=np.uint8)
    crop[20:40, 62:108] = (0, 0, 0)
    mask = np.zeros((60, 120), dtype=np.uint8)
    mask[18:42, 60:110] = 255
    assert _mask_coverage_ok(crop, mask, (255, 255, 255), _interior_full(60, 120)) is True


def test_coverage_fails_on_half_covered_glyph() -> None:
    # B19 sınıfı: mürekkebin yarısından azı maskede → temizliğe girilmez.
    crop = np.full((60, 120, 3), 255, dtype=np.uint8)
    crop[20:40, 20:80] = (0, 0, 0)  # 60x20 = 1200px bar
    mask = np.zeros((60, 120), dtype=np.uint8)
    mask[20:40, 60:110] = 255  # barın üçte birini kapsar
    assert _mask_coverage_ok(crop, mask, (255, 255, 255), _interior_full(60, 120)) is False


def test_coverage_defers_without_interior() -> None:
    # Balon-içi bilinmiyorsa sanat yanlış-alarm üretmesin diye kapı pas geçer.
    crop = np.full((60, 120, 3), 255, dtype=np.uint8)
    crop[20:40, 20:80] = (0, 0, 0)
    mask = np.zeros((60, 120), dtype=np.uint8)
    mask[20:40, 60:110] = 255
    assert _mask_coverage_ok(crop, mask, (255, 255, 255), None) is True


def test_coverage_clean_balloon_passes() -> None:
    crop = np.full((60, 120, 3), 255, dtype=np.uint8)
    mask = np.zeros((60, 120), dtype=np.uint8)
    mask[20:40, 60:110] = 255
    assert _mask_coverage_ok(crop, mask, (255, 255, 255), _interior_full(60, 120)) is True
