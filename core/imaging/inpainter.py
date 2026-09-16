"""Mask-only speech text removal with a uniform fast path and Big-LaMa fallback."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Any, Sequence

import numpy as np
from PIL import Image

from core.detection import Region, RegionStatus, RegionType
from core.imaging.lama import LaMaLargeInpainter
from core.imaging.text_mask import TextMask, TextMaskBuilder


DEFAULT_LAMA_CHECKPOINT = Path(r"C:\AI\Models\LaMa\lama_large_512px.ckpt")


# Faz 4 eşikleri (genel; piksel-sanatı varsayımı yok, hepsi göreli).
# LaMa maske çekirdeği maske yüksekliğine göre ölçeklenir (sabit 7x7 yerine).
LAMA_KERNEL_MIN = 3
LAMA_KERNEL_MAX = 21
# İç-artık: dolgu renginden bu kadar sapan, bu büyüklükte bileşenler glif artığıdır.
GHOST_CONTRAST_THRESHOLD = 40.0
GHOST_MIN_COMPONENT_AREA = 15
# Dağınık zerre artığı: tekil bileşen eşiğini aşamayan ama toplamda kirleten.
GHOST_MIN_TOTAL_AREA = 60
# Dolgu/zemin uyuşmazlığı: balon bulunamadıysa iç-ortanca ile dış-halka
# ortancası bu kadar ayrışamaz (P003 beyaz-leke vakası).
FILL_RING_LUMA_GAP = 60.0
FILL_RING_WIDTH = 6


def lama_kernel_for_height(mask_h: int) -> int:
    """Maske yüksekliğine göre tek-sayı morfolojik çekirdek boyutu.

    Küçük yazıda dar (artık bırakmaz), büyük yazıda geniş (hale yapmaz).
    Sınırlar: 3..21, her zaman tek sayı.
    """
    k = int(round(max(1, mask_h) * 0.06)) * 2 + 1
    return max(LAMA_KERNEL_MIN, min(LAMA_KERNEL_MAX, k))


def _luminance(pixel: np.ndarray) -> float:
    rgb = np.asarray(pixel, dtype=np.float32).reshape(-1, 3)
    return float(np.median(np.dot(rgb, [0.299, 0.587, 0.114])))


def _ring_luma_stats(
    img: np.ndarray, mask: np.ndarray, bubble_interior: np.ndarray | None
) -> tuple[float, float] | None:
    """2-8px halka ortanca ışıklık + standart sapma (yoksa None)."""
    import cv2

    d_outer = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    d_inner = cv2.dilate(mask, np.ones((2, 2), np.uint8))
    ring = (d_outer > 0) & (d_inner == 0)
    if bubble_interior is not None and np.any(bubble_interior):
        ring &= (bubble_interior > 0)
    px = img[ring]
    if len(px) < 16:
        px = img[mask == 0]
    if len(px) < 8:
        return None
    gray = cv2.cvtColor(
        np.ascontiguousarray(px.reshape(-1, 1, 3)), cv2.COLOR_RGB2GRAY
    ).astype(np.float32)
    return float(np.median(gray)), float(np.std(gray))


def _is_story_text(region: Region) -> bool:
    return (
        region.status == RegionStatus.AUTO
        and region.type not in (RegionType.SFX, RegionType.WATERMARK)
        and bool(region.text and region.text.strip())
    )


def _is_second_chance_block(block: Any) -> bool:
    """Blok üyelerinin tamamı eşik-altı ikinci-şans tespiti mi?"""
    members: tuple[Any, ...] = tuple(getattr(block, "members", ()) or ())
    if not members:
        return False
    for m in members:
        meta = getattr(m, "metadata", None)
        if not isinstance(meta, dict) or not meta.get("second_chance"):
            return False
    return True


# İkinci-şans maske taşması fasesi sabitleri.
SECOND_CHANCE_BG_TOLERANCE = 28
# Alan tavanı (koyu glifler grown'a giremez → komşu metin silinemez;
# tavan yalnız hesapsal kaçak içindir, beyaz-beyaza boyama zaten no-op'tur).
SECOND_CHANCE_MAX_AREA_RATIO = 40


# S4 maske-dışı bant sabitleri (B19-ölçümlü: "M'" 561px/32x37oran1.2,
# maskeye 4.8px, %98 aydınlık-komşu; dungeon b10-ölçümlü: balon çizgi
# artıkları 31x3..143x11 (oran≥10), toz 53px).
# Kural: maskeye ≤8px + komşu ≥%50 aydınlık + alan 100..4000 +
# en-boy-oranı ≤6 + kutu ≤1/3. Çizgi/toz elenir, glif parçası kalır.
OUTSIDE_MAX_DIST_PX = 8.0
OUTSIDE_MIN_LIGHT_FRAC = 0.5
OUTSIDE_MIN_COMPONENT_AREA = 100
OUTSIDE_MAX_COMPONENT_AREA = 4000
OUTSIDE_MAX_ASPECT = 6.0
OUTSIDE_DARK_MARGIN = 40


def _outside_mask_text(
    source_crop: np.ndarray,
    inpainted_crop: np.ndarray,
    refined_mask: np.ndarray,
    background_color: tuple[int, int, int] | list[int],
    bubble_interior: np.ndarray | None = None,
) -> bool:
    """Maske-dışı bantta kalmış glif parçası var mı (S4)?

    Refined maskenin ~12px dış bandında, BALON DOLGUSU üstünde (kaynak
    zemine yakın-açık) duran, bitmiş görüntüde zeminden koyu, glif-ölçekli
    (5..4000px, kırpıntının 1/3'ünden küçük kutulu) bağlı bileşen arar.
    Kaynak-koyu bant (balon çizgisi, dış sanat, kaya dokusu) elenir —
    artık hem kaynakta hem bitmiş görüntüde koyu olmak zorundadır ki
    inpaint lekesinden değil, temizlenmemiş gliften söz edelim.
    Balon-içi biliniyorsa bant onunla sınırlanır.
    Temizlenememiş tam-İngilizce de DAHİL bayraklanır: üzerine Türkçe
    basılırsa çakışır — blok İngilizce korunmalıdır (S0: ya tam temizle
    ya hiç dokunma).
    """
    import cv2

    refined = (np.asarray(refined_mask) > 0)
    if not np.any(refined):
        return False
    h, w = refined.shape[:2]
    src_gray = cv2.cvtColor(np.ascontiguousarray(source_crop), cv2.COLOR_RGB2GRAY).astype(np.float32)
    out_gray = cv2.cvtColor(np.ascontiguousarray(inpainted_crop), cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg = float(np.dot(np.asarray(background_color, dtype=np.float32), [0.299, 0.587, 0.114]))
    # Dokunulmamış artık: hem kaynakta hem bitmiş görüntüde koyu.
    leftover = ((bg - out_gray >= OUTSIDE_DARK_MARGIN)
                & (bg - src_gray >= OUTSIDE_DARK_MARGIN)
                & (~refined))
    if bubble_interior is not None:
        leftover &= (np.asarray(bubble_interior) > 0)
    if int(np.count_nonzero(leftover)) < OUTSIDE_MIN_COMPONENT_AREA:
        return False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(leftover.astype(np.uint8), 8)
    if count <= 1:
        return False
    # Maskeye uzaklık: arkaplan piksellerinin en yakın maske-piksele mesafesi
    # (ters-çevirme şart — distanceTransform sıfıra olanı ölçer).
    dist = cv2.distanceTransform((~refined).astype(np.uint8), cv2.DIST_L2, 3)
    src_light = src_gray >= bg - 25
    ring_kernel = np.ones((11, 11), np.uint8)
    labels = np.asarray(labels)
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if not (OUTSIDE_MIN_COMPONENT_AREA <= area <= OUTSIDE_MAX_COMPONENT_AREA):
            continue
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if bw > w // 3 or bh > h // 3:
            continue
        # Çizgi artıkları (balon kuyruğu/kontur: oran≥10) elenir;
        # glif parçaları topludur (B19 "M'" oranı 1.2).
        if max(bw, bh) / max(1, min(bw, bh)) > OUTSIDE_MAX_ASPECT:
            continue
        ys, xs = np.where(labels == idx)
        if dist[ys, xs].min() > OUTSIDE_MAX_DIST_PX:
            continue
        comp = (labels == idx).astype(np.uint8)
        ring = (cv2.dilate(comp, ring_kernel) > 0) & (comp == 0)
        if int(np.count_nonzero(ring)) == 0:
            continue
        light_frac = float(np.count_nonzero(src_light & (ring > 0))) / float(np.count_nonzero(ring))
        if light_frac >= OUTSIDE_MIN_LIGHT_FRAC:
            return True
    return False


def _outside_mask_boxes(
    source_crop: np.ndarray,
    inpainted_crop: np.ndarray,
    refined_mask: np.ndarray,
    background_color: tuple[int, int, int] | list[int],
    bubble_interior: np.ndarray | None = None,
) -> list[tuple[int, int, int, int]]:
    """_outside_mask_text ile AYNI kapılardan geçen bileşenlerin kutuları.

    Kırpıntı-içi (x1, y1, x2, y2) döndürür; kapı mantığı tek kaynaktır
    (bool sürüm bu listedeki ilk elemanın varlığına bakar).
    """
    import cv2

    refined = (np.asarray(refined_mask) > 0)
    if not np.any(refined):
        return []
    h, w = refined.shape[:2]
    src_gray = cv2.cvtColor(np.ascontiguousarray(source_crop), cv2.COLOR_RGB2GRAY).astype(np.float32)
    out_gray = cv2.cvtColor(np.ascontiguousarray(inpainted_crop), cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg = float(np.dot(np.asarray(background_color, dtype=np.float32), [0.299, 0.587, 0.114]))
    leftover = ((bg - out_gray >= OUTSIDE_DARK_MARGIN)
                & (bg - src_gray >= OUTSIDE_DARK_MARGIN)
                & (~refined))
    if bubble_interior is not None:
        leftover &= (np.asarray(bubble_interior) > 0)
    if int(np.count_nonzero(leftover)) < OUTSIDE_MIN_COMPONENT_AREA:
        return []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(leftover.astype(np.uint8), 8)
    if count <= 1:
        return []
    dist = cv2.distanceTransform((~refined).astype(np.uint8), cv2.DIST_L2, 3)
    src_light = src_gray >= bg - 25
    ring_kernel = np.ones((11, 11), np.uint8)
    labels = np.asarray(labels)
    boxes: list[tuple[int, int, int, int]] = []
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if not (OUTSIDE_MIN_COMPONENT_AREA <= area <= OUTSIDE_MAX_COMPONENT_AREA):
            continue
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if bw > w // 3 or bh > h // 3:
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > OUTSIDE_MAX_ASPECT:
            continue
        ys, xs = np.where(labels == idx)
        if dist[ys, xs].min() > OUTSIDE_MAX_DIST_PX:
            continue
        comp = (labels == idx).astype(np.uint8)
        ring = (cv2.dilate(comp, ring_kernel) > 0) & (comp == 0)
        if int(np.count_nonzero(ring)) == 0:
            continue
        light_frac = float(np.count_nonzero(src_light & (ring > 0))) / float(np.count_nonzero(ring))
        if light_frac >= OUTSIDE_MIN_LIGHT_FRAC:
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            boxes.append((x, y, x + bw, y + bh))
    return boxes


# P2 kapsama-kapısı sabitleri: her mürekkep parçasının en az yarısı
# maskede olmalı, yoksa temizliğe girilmez (B19 "MY" sınıfı).
# Alan eşiği balon-göreli: küçük balonda toz veto üretmesin, büyükte
# zerreler veto üretmesin. Taban 80px (B19 toz 53px altı kalır).
COVERAGE_MIN_COMPONENT_AREA = 100
COVERAGE_MIN_FRACTION = 0.5
# Balon-göreli gevşeme (B19 toz sınıfı korunur): büyük balonda toz veto
# üretmesin. Taban ESKİ 100 (küçük balonda davranış birebir aynı — daraltma
# YOK, r3 dersi: tabanı düşürmek veto artırır). Tavan 300: B19-ölçümlü boşluk
# (toz 53 < 300 < M' 561) — gerçek parça büyük balonda da veto yer.
COVERAGE_AREA_FLOOR = 100
COVERAGE_AREA_RATIO = 0.002
COVERAGE_AREA_CEIL = 300


def _coverage_min_area(bubble_pixels: int) -> int:
    """Balon büyüklüğüne göre en-küçük mürekkep alanı (genel, oranlı)."""
    return max(COVERAGE_AREA_FLOOR, min(COVERAGE_AREA_CEIL, int(round(bubble_pixels * COVERAGE_AREA_RATIO))))


def _bg_luma_tuple(background_color) -> tuple[int, int, int]:
    vals = [int(v) for v in np.asarray(background_color).ravel()[:3]]
    return (vals[0], vals[1], vals[2]) if len(vals) == 3 else (255, 255, 255)


def _mask_coverage_ok(
    source_crop: np.ndarray,
    refined_mask: np.ndarray,
    background_color: tuple[int, int, int] | list[int],
    bubble_interior: np.ndarray | None = None,
) -> bool:
    """P2: maske, balon-içi mürekkep parçalarını kapsıyor mu?

    Balon-göreli her kaynak-koyu bileşenin ≥%50'si refined içindeyse True.
    Balon-içi bilinmiyorsa True döner (sanat/kirpinti-kenarı yanlış
    alarm üretmesin diye — o durumda sonradan-kontroller devrededir).
    """
    import cv2

    refined = (np.asarray(refined_mask) > 0)
    if not np.any(refined):
        return True
    if bubble_interior is None:
        return True
    area = (np.asarray(bubble_interior) > 0)
    bubble_pixels = int(np.count_nonzero(area))
    min_area = _coverage_min_area(bubble_pixels)
    gray = cv2.cvtColor(np.ascontiguousarray(source_crop), cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg = float(np.dot(np.asarray(background_color, dtype=np.float32), [0.299, 0.587, 0.114]))
    dark = ((bg - gray) >= OUTSIDE_DARK_MARGIN) & area
    if int(np.count_nonzero(dark)) < min_area:
        return True
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    if count <= 1:
        return True
    refined_u8 = refined.astype(np.uint8)
    for idx in range(1, count):
        comp_area = int(stats[idx, cv2.CC_STAT_AREA])
        if comp_area < min_area:
            continue
        comp = (np.asarray(labels) == idx)
        inside = int(np.count_nonzero(comp & (refined_u8 > 0)))
        if inside / max(1, comp_area) < COVERAGE_MIN_FRACTION:
            return False
    return True


def _expand_refined_to_bubble_ink(
    source_crop: np.ndarray,
    refined_mask: np.ndarray,
    background_color: tuple[int, int, int] | list[int],
    bubble_interior: np.ndarray | None,
) -> np.ndarray:
    """Düz balonda maskeye girmemiş mürekkebi maskeye kat (genel kurtarma).

    YALNIZ düz-renkli balonlarda çağrılır (sanat korunur): balon-içi
    zeminle zıt her mürekkep parçası (koyu-açık iki yön) en-boy-oranı
    ve balon-göreli alan süzgecinden geçerse refined'a eklenir. Uzak
    küçük kelimeler (`IN` 116px kuzeyde) böyle kurtulur; toz/çizgi
    elenir. Renk-bağımsız, metin-bağımsız, bölüm-sayısı yok.
    """
    import cv2

    refined = (np.asarray(refined_mask) > 0)
    if not np.any(refined) or bubble_interior is None:
        return np.asarray(refined_mask)
    area = (np.asarray(bubble_interior) > 0)
    if not np.any(area):
        return np.asarray(refined_mask)
    bubble_pixels = int(np.count_nonzero(area))
    min_area = _coverage_min_area(bubble_pixels)
    max_area = max(min_area + 1, bubble_pixels // 3)
    gray = cv2.cvtColor(np.ascontiguousarray(source_crop), cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg = float(np.dot(np.asarray(background_color, dtype=np.float32), [0.299, 0.587, 0.114]))
    ink = (np.abs(gray - bg) >= OUTSIDE_DARK_MARGIN) & area
    ink = ink & (~refined)
    if int(np.count_nonzero(ink)) < min_area:
        return np.asarray(refined_mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    if count <= 1:
        return np.asarray(refined_mask)
    grown = refined.copy()
    for idx in range(1, count):
        comp_area = int(stats[idx, cv2.CC_STAT_AREA])
        if comp_area < min_area or comp_area > max_area:
            continue
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if min(w, h) <= 0:
            continue
        if max(w, h) / max(1, min(w, h)) > OUTSIDE_MAX_ASPECT:
            continue
        grown |= (np.asarray(labels) == idx)
    if not np.any(grown & (~refined)):
        return np.asarray(refined_mask)
    out = np.zeros_like(np.asarray(refined_mask), dtype=np.uint8)
    out[grown] = 255
    return out


# Yumuşak-hale eşiği (zemin-bağıl): bileşen filtresinden kaçan
# kenar-yumuşatma pikselleri. 40'lık glif eşiğinin altında, ama düz
# zeminde hâlâ kir (hayalet/bulanık kenar). YALNIZ tekdüze balonda.
SOFT_HALO_MARGIN = 12


def _catch_soft_halo(
    source_crop: np.ndarray,
    refined_mask: np.ndarray,
    background_color: tuple[int, int, int] | list[int],
    bubble_interior: np.ndarray | None,
) -> np.ndarray:
    """Düz balonda maske-kenarındaki soluk hale piksellerini maskeye kat.

    Bileşen-kurtarma (40 kontrast + alan süzgeci) uzak mürekkebi alır;
    bu, YAKINDAKİ soluk kenarları (yumuşatma/gölge, 12+ kontrast) toplar.
    Sınır maske-boyutuna göre (oranlı), yalnızca maskeye BAĞLI bileşenler
    alınır (uzak toz asla). Renk-bağıl, boyut-oranlı, bölüm-sayısı yok.
    """
    import cv2

    base = np.asarray(refined_mask)
    refined = (base > 0)
    if not np.any(refined) or bubble_interior is None:
        return base
    area = (np.asarray(bubble_interior) > 0)
    if not np.any(area):
        return base
    gray = cv2.cvtColor(np.ascontiguousarray(source_crop), cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg = float(np.dot(np.asarray(background_color, dtype=np.float32), [0.299, 0.587, 0.114]))
    soft = (np.abs(gray - bg) >= SOFT_HALO_MARGIN) & area & (~refined)
    min_add = max(8, int(round(int(np.count_nonzero(refined)) * 0.001)))
    if int(np.count_nonzero(soft)) < min_add:
        return base
    h, w = refined.shape[:2]
    k = min(32, max(8, int(round(max(h, w) * 0.08))))
    bound = cv2.dilate(refined.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    cand = (refined | (soft & bound)).astype(np.uint8)
    count, labels = cv2.connectedComponents(cand, 8)[:2]
    if count <= 1:
        return base
    seed_labels = set(np.unique(np.asarray(labels)[refined]))
    seed_labels.discard(0)
    if not seed_labels:
        return base
    kept = np.isin(np.asarray(labels), list(seed_labels))
    if not np.any(kept & (~refined)):
        return base
    out = np.zeros_like(base, dtype=np.uint8)
    out[kept] = 255
    return out


def _bubble_fill_safe(
    source_crop: np.ndarray,
    bubble_interior: np.ndarray | None,
) -> tuple[bool, tuple[int, int, int]]:
    """Balon-içi düz-dolguya uygun mu (balon-geneli sertifika)?

    r5 dersi: maske-istatistiği (is_uniform, std<=9) metin-yoğun kutuda
    haksız veto verir; oysa can_flat (düz-dolguyu seçen kapı) True'dur.
    Bu sertifika balonun TAMAMINI ölçer (mürekkep hariç): tekdüzeyse düz
    dolgu görsel no-op'tur — maske ne kadar büyürse büyüsün güvenlidir.
    Eşikler mevcut kapıların aynısı (14/12/20/40; yeni mutlak sayı yok).
    """
    import cv2

    if bubble_interior is None:
        return False, (255, 255, 255)
    area = (np.asarray(bubble_interior) > 0)
    if int(np.count_nonzero(area)) < 24:
        return False, (255, 255, 255)
    img = np.ascontiguousarray(source_crop)
    med = np.median(img[area].reshape(-1, 3).astype(np.float32), axis=0)
    dev = np.max(np.abs(img.reshape(-1, 3).astype(np.float32) - med), axis=1).reshape(img.shape[:2])
    bgpx = img[(area) & (dev < OUTSIDE_DARK_MARGIN)]
    if len(bgpx) < 24:
        return False, (255, 255, 255)
    std_rgb = np.std(bgpx, axis=0)
    if float(np.max(std_rgb)) > 14.0 or float(np.std(bgpx)) > 12.0:
        return False, (255, 255, 255)
    distances = np.linalg.norm(bgpx.astype(np.float32) - med, axis=1)
    if float(np.percentile(distances, 90)) > 20:
        return False, (255, 255, 255)
    median_color = tuple(int(round(v)) for v in med)
    median_color = (median_color[0], median_color[1], median_color[2]) if len(median_color) == 3 else (255, 255, 255)
    return True, median_color


def _fill_uniform_bubble_interior(    refined_mask: np.ndarray,
    bubble_interior: np.ndarray | None,
) -> np.ndarray:
    """Tekdüze balonun içini maskeye kat (düz-dolgu ile temizlenecek).

    Bileşen-kurtarma uzak mürekkebi, hale-yakalama kenarları, bant-emme
    yakın parçaları alır; ama S4'ün gördüğü (genişlemiş maskeye ≤8px)
    artıklar aradan kaçabilir. Tekdüze zeminde EN GENEL çözüm: için TAMAMINI
    maskelemek — düz-dolgu tekdüze pikseli aynen yazar (görsel no-op),
    mürekkep pikseli zemin rengine döner. Balon-çizgisi korunur (içerik
    7x7 aşındırılır). Çağıran kapı ÇİFT kilit ister: is_uniform (balon-geneli
    örneklemeli) VE can_flat — yoksa LaMa devasa maskeyle çağrılırdı.
    """
    import cv2

    base = np.asarray(refined_mask)
    if bubble_interior is None:
        return base
    area = (np.asarray(bubble_interior) > 0)
    if not np.any(area):
        return base
    safe = cv2.erode(area.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    grown = (base > 0) | safe
    if not np.any(grown & ~(base > 0)):
        return base
    out = np.zeros_like(base, dtype=np.uint8)
    out[grown] = 255
    return out


def _absorb_nearblock_ink(
    source_crop: np.ndarray,
    refined_mask: np.ndarray,
    background_color: tuple[int, int, int] | list[int],
    dilation_radius: int = 3,
) -> np.ndarray:
    """Balonsuz kutuda kutuya-bitişik mürekkebi maskeye kat (2b).

    Balon bulunamadığında (sivri/kenar-dayalı) S4'ün gördüğü yakın-artıklar
    (IS/THE sınıfı) ne kapsamaya girer ne banda takılır — sayfada kalır.
    Bu yedek, maske-kenar bandı TEKDÜZE ise (sanat yoksa) S4-aynalı
    parçaları önden emer; çağrı sonrası düz-dolgu sürüyorsa kalır, yoksa
    çağrıcı iade eder (LaMa-dev-maske YASAK). Yarıçap = S4'ün 8px'i +
    yazı-boyuna-göre genleşme payı (dilation_radius) — yeni mutlak yok.
    """
    import cv2

    base = np.asarray(refined_mask)
    refined = (base > 0)
    if not np.any(refined):
        return base
    radius = int(OUTSIDE_MAX_DIST_PX + max(0, dilation_radius))
    # Bant, mesafe-yarıçapının TAMAMINI kapsar (2r+1): parça bütün
    # ölçülür, kenardan-kırpık alan tabanı delinmez.
    band = (cv2.dilate(refined.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) > 0) & (~refined)
    if int(np.count_nonzero(band)) < 24:
        return base
    img = np.ascontiguousarray(source_crop)
    bandpx = img[band].reshape(-1, 3).astype(np.float32)
    med = np.median(bandpx, axis=0)
    dev = np.max(np.abs(bandpx - med), axis=1)
    clean = bandpx[dev < OUTSIDE_DARK_MARGIN]
    if len(clean) < 24:
        return base
    if float(np.max(np.std(clean, axis=0))) > 14.0 or float(np.std(clean)) > 12.0:
        return base
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg = float(np.dot(np.asarray(background_color, dtype=np.float32), [0.299, 0.587, 0.114]))
    dark = ((bg - gray) >= OUTSIDE_DARK_MARGIN) & band
    if int(np.count_nonzero(dark)) < OUTSIDE_MIN_COMPONENT_AREA:
        return base
    dist = cv2.distanceTransform((~refined).astype(np.uint8), cv2.DIST_L2, 3)
    h, w = refined.shape[:2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    if count <= 1:
        return base
    grown = refined.copy()
    labels = np.asarray(labels)
    for idx in range(1, count):
        comp_area = int(stats[idx, cv2.CC_STAT_AREA])
        if comp_area < OUTSIDE_MIN_COMPONENT_AREA or comp_area > OUTSIDE_MAX_COMPONENT_AREA:
            continue
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if bw > w // 3 or bh > h // 3:
            continue
        if min(bw, bh) <= 0 or max(bw, bh) / max(1, min(bw, bh)) > OUTSIDE_MAX_ASPECT:
            continue
        ys, xs = np.where(labels == idx)
        if float(np.min(dist[ys, xs])) > radius:
            continue
        grown |= (labels == idx)
    if not np.any(grown & (~refined)):
        return base
    out = np.zeros_like(base, dtype=np.uint8)
    out[grown] = 255
    return out


def _absorb_nearband_ink(    source_crop: np.ndarray,
    refined_mask: np.ndarray,
    background_color: tuple[int, int, int] | list[int],
    bubble_interior: np.ndarray | None,
) -> np.ndarray:
    """Düz balonda S4-bant mürekkebini veto yerine maskeye kat (ön-emme).

    S4 (maske-dışı bant) maskeye ≤8px, 100..4000px, oranı ≤6 parçayı veto
    eder (B19 "M'" sınıfı). Düz zeminde bu parçalar zararsızca silinebilir:
    veto yerine maskeye katılır, düz-dolgu temizler. Sanatlı balonda ASLA
    çalışmaz (çağıran kapı tekdüzelik ister). S4'ün kendi eşikleri aynen
    kullanılır (yeni mutlak sayı yok), karar SAMİMİ: veto→temizlik.
    """
    import cv2

    base = np.asarray(refined_mask)
    refined = (base > 0)
    if not np.any(refined) or bubble_interior is None:
        return base
    area = (np.asarray(bubble_interior) > 0)
    if not np.any(area):
        return base
    gray = cv2.cvtColor(np.ascontiguousarray(source_crop), cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg = float(np.dot(np.asarray(background_color, dtype=np.float32), [0.299, 0.587, 0.114]))
    dark = ((bg - gray) >= OUTSIDE_DARK_MARGIN) & area & (~refined)
    if int(np.count_nonzero(dark)) < OUTSIDE_MIN_COMPONENT_AREA:
        return base
    dist = cv2.distanceTransform((~refined).astype(np.uint8), cv2.DIST_L2, 3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    if count <= 1:
        return base
    grown = refined.copy()
    for idx in range(1, count):
        comp_area = int(stats[idx, cv2.CC_STAT_AREA])
        if comp_area < OUTSIDE_MIN_COMPONENT_AREA or comp_area > OUTSIDE_MAX_COMPONENT_AREA:
            continue
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if min(w, h) <= 0 or max(w, h) / max(1, min(w, h)) > OUTSIDE_MAX_ASPECT:
            continue
        comp = (np.asarray(labels) == idx)
        if float(np.min(dist[comp])) > OUTSIDE_MAX_DIST_PX:
            continue
        grown |= comp
    if not np.any(grown & (~refined)):
        return base
    out = np.zeros_like(base, dtype=np.uint8)
    out[grown] = 255
    return out


def _lama_mask_for(
    base_mask: np.ndarray,
    bubble_interior: np.ndarray | None = None,
) -> np.ndarray:
    """LaMa maskesi: genişlet + balonla-kırp + kenar-yumuşat (B).

    Kırpma: maske balon-dışına taşamaz (sanat-bulaşma biter; veto yönünde
    güvenli — kaçıran kapı REVIEW verir, kir basılmaz). Yumuşatma: 3x3
    bulanık + eşik (ayak-izi ~aynı, sert halka kenarı yumuşar).
    """
    import cv2

    kh = lama_kernel_for_height(base_mask.shape[0])
    grown = cv2.dilate(
        np.asarray(base_mask), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kh, kh))
    )
    if bubble_interior is not None and np.any(bubble_interior):
        grown = (
            np.logical_and(np.asarray(grown) > 0, np.asarray(bubble_interior) > 0)
            .astype(np.uint8)
            * 255
        )
    soft = cv2.GaussianBlur(grown, (3, 3), 0)
    _, out = cv2.threshold(soft, 128, 255, cv2.THRESH_BINARY)
    return out


class Inpainter:
    """Removes source glyphs while preserving every pixel outside the refined mask."""

    def __init__(
        self,
        lama_checkpoint: str | Path = DEFAULT_LAMA_CHECKPOINT,
        debug_dir: str | Path | None = None,
        context_scale: float = 1.7,
        bubble_boxes: Sequence[tuple[int, int, int, int]] | None = None,
    ) -> None:
        self.mask_builder = TextMaskBuilder(context_scale=context_scale, bubble_boxes=bubble_boxes)
        self.lama = LaMaLargeInpainter(lama_checkpoint)
        self.debug_dir = Path(debug_dir) if debug_dir is not None else None
        self.debug_records: list[dict[str, Any]] = []
        self.processed_block_ids: set[int] = set()
        self.review_block_ids: set[int] = set()
        # Madde 3: REVIEW'a düşüren alt-sebep (forensik; sebep string'i sabit).
        self.review_causes: dict[int, str] = {}
        # Artık-kutu tesisatı (ölçüm): boundary/outside sebepli bloklarda
        # kalan artığın global kutuları — render-kapsama kurtarması için.
        # Davranış değiştirmez (yalnız kayıt).
        self.review_residual_boxes: dict[int, list[list[int]]] = {}
        self.last_text_mask: TextMask | None = None

    def unload(self) -> None:
        self.lama.unload()

    def inpaint_batch(
        self,
        images: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        batch_size: int = 24,
    ) -> list[np.ndarray]:
        """Inpaint a list of image crops with corresponding masks using GPU batching."""
        return self.lama.inpaint_batch(images, masks, batch_size=batch_size)

    def inpaint_blocks(self, canvas: Image.Image, text_blocks: Sequence[Any]) -> Image.Image:
        self.last_text_mask = None
        result = np.array(canvas.convert("RGB"), dtype=np.uint8, copy=True)
        
        # 1. Build text masks for all eligible blocks
        prepared: list[tuple[Any, TextMask, str]] = []
        for block in text_blocks:
            members: tuple[Any, ...] = tuple(getattr(block, "members", ()))
            if not members or any(not _is_story_text(r) for r in members):
                continue
            eligible = members
            mask = self.mask_builder._build(result, block.merged_bbox, eligible)
            # İkinci-şans kutuları kısmi olur (düşük güven ↔ eksik geometri);
            # maske balon içinde görece genişletilir (sanat korunur).
            if _is_second_chance_block(block):
                _member_text = " ".join(
                    (getattr(m, "text", "") or "") for m in eligible
                )
                mask = self._expand_mask_in_bubble(mask, _member_text)
            # P1 kopya-bekçisi: tam-genişlik kırpıntılarda mask.source
            # canvas'a bakış (view) olabilir; sonraki blok uygulamaları
            # onu bozardı. Geri-alma için her maske kendi kopyasını tutar.
            if mask.source.base is not None:
                mask = replace(mask, source=mask.source.copy())
            block_id = int(getattr(block, "id", -1))
            if np.any(mask.refined):
                self.processed_block_ids.add(block_id)
            else:
                # No approved text mask means the translated block cannot be
                # safely rendered. Count it as inpaint REVIEW so lifecycle
                # totals remain explicit and the original pixels stay intact.
                self.review_block_ids.add(block_id)
                self.review_causes[block_id] = "empty_mask"
            debug_name = f"block_{getattr(block, 'id', len(self.debug_records) + len(prepared) + 1):04d}"
            # Düz-balon kurtarma (genel): sertifikalı-tekdüze zeminde maske-dışı
            # mürekkep maskeye katılır (uzak küçük kelimeler, hale, bant).
            # Sanatlı balonda ASLA çalışmaz (sertifika yoksa atlanır).
            # Sonrası aynı kapı; büyümüş maske düz-dolguyu kaybederse iade.
            if np.any(mask.refined) and mask.bubble_found:
                fill_safe, _ = _bubble_fill_safe(mask.source, mask.bubble_interior)
                if fill_safe:
                    original_refined = mask.refined
                    filled = _fill_uniform_bubble_interior(mask.refined, mask.bubble_interior)
                    if np.any((np.asarray(filled) > 0) & ~(np.asarray(mask.refined) > 0)):
                        mask = replace(mask, refined=np.ascontiguousarray(filled))
                    recovered = _expand_refined_to_bubble_ink(
                        mask.source,
                        mask.refined,
                        _bg_luma_tuple(mask.background_color),
                        mask.bubble_interior,
                    )
                    if np.any((np.asarray(recovered) > 0) & ~(np.asarray(mask.refined) > 0)):
                        mask = replace(mask, refined=np.ascontiguousarray(recovered))
                    halo_caught = _catch_soft_halo(
                        mask.source,
                        mask.refined,
                        _bg_luma_tuple(mask.background_color),
                        mask.bubble_interior,
                    )
                    if np.any((np.asarray(halo_caught) > 0) & ~(np.asarray(mask.refined) > 0)):
                        mask = replace(mask, refined=np.ascontiguousarray(halo_caught))
                    band_absorbed = _absorb_nearband_ink(
                        mask.source,
                        mask.refined,
                        _bg_luma_tuple(mask.background_color),
                        mask.bubble_interior,
                    )
                    if np.any((np.asarray(band_absorbed) > 0) & ~(np.asarray(mask.refined) > 0)):
                        mask = replace(mask, refined=np.ascontiguousarray(band_absorbed))
                    # LaMa-sigortası: büyümüş maske düz-dolguyu kaybederse
                    # (ilerideki _apply_mask LaMa'ya düşerdi) büyüme iade —
                    # dev LaMa maskesi YASAK.
                    still_flat, _ = self._can_use_flat_fill(
                        mask.source, mask.refined, mask.bubble_interior
                    )
                    if not still_flat:
                        mask = replace(mask, refined=np.ascontiguousarray(original_refined))
            # 2b balonsuz-kutu (genel): balon bulunamadığında S4-aynalı yakın
            # mürekkep tekdüze-bantta emilir (IS/THE sınıfı). Balonlu kutular
            # yukarıda halledilir (çift-çalışma yok). Düz-dolgu sürmezse iade.
            if np.any(mask.refined) and not mask.bubble_found:
                original_refined_2b = mask.refined
                absorbed_2b = _absorb_nearblock_ink(
                    mask.source,
                    mask.refined,
                    _bg_luma_tuple(mask.background_color),
                    mask.dilation_radius,
                )
                if np.any((np.asarray(absorbed_2b) > 0) & ~(np.asarray(mask.refined) > 0)):
                    mask = replace(mask, refined=np.ascontiguousarray(absorbed_2b))
                    still_flat_2b, _ = self._can_use_flat_fill(
                        mask.source, mask.refined, mask.bubble_interior
                    )
                    if not still_flat_2b:
                        mask = replace(mask, refined=np.ascontiguousarray(original_refined_2b))
            # P2 kapsama-kapısı: maske mürekkebi kapsamıyorsa temizliğe
            # girilmez (B19 "MY" sınıfı) — piksel aynen durur, blok REVIEW.
            if np.any(mask.refined) and not _mask_coverage_ok(
                mask.source,
                mask.refined,
                _bg_luma_tuple(mask.background_color),
                mask.bubble_interior,
            ):
                self.review_block_ids.add(block_id)
                self.review_causes[block_id] = "coverage"
                self._save_debug(debug_name, mask, mask.source, "coverage_skip", review=True)
                continue
            prepared.append((block, mask, debug_name))

        # 2. Batch GPU LaMa inference for all blocks requiring full neural inpainting
        lama_jobs: list[tuple[int, np.ndarray, np.ndarray]] = []
        for idx, (block, mask, _) in enumerate(prepared):
            if not np.any(mask.refined):
                continue
            can_flat, _ = self._can_use_flat_fill(mask.source, mask.refined, mask.bubble_interior)
            if not can_flat and not mask.is_uniform_background:
                lama_mask = _lama_mask_for(mask.refined, mask.bubble_interior)
                lama_jobs.append((idx, mask.source, lama_mask))

        precomputed_crops: dict[int, np.ndarray] = {}
        if lama_jobs:
            lama_sources = [job[1] for job in lama_jobs]
            lama_masks = [job[2] for job in lama_jobs]
            batch_results = self.lama.inpaint_batch(lama_sources, lama_masks, batch_size=24)
            for (prep_idx, _, _), res_crop in zip(lama_jobs, batch_results):
                precomputed_crops[prep_idx] = res_crop

        # 3. Apply masks in-place directly on the single canvas buffer
        for idx, (block, mask, debug_name) in enumerate(prepared):
            pre_crop = precomputed_crops.get(idx)
            self._apply_mask(result, mask, debug_name, in_place=True, precomputed_crop=pre_crop)

        return Image.fromarray(result, "RGB")

    def inpaint_regions(self, canvas: Image.Image, regions: Sequence[Region]) -> Image.Image:
        result = np.array(canvas.convert("RGB"), dtype=np.uint8, copy=True)
        for region in regions:
            if not _is_story_text(region):
                continue
            mask = self.mask_builder.build_for_region(result, region)
            self._apply_mask(result, mask, f"region_{region.id:04d}", in_place=True)
        return Image.fromarray(result, "RGB")

    def _apply_mask(
        self,
        full_source: np.ndarray,
        text_mask: TextMask,
        debug_name: str,
        in_place: bool = False,
        precomputed_crop: np.ndarray | None = None,
    ) -> np.ndarray:
        x1, y1, x2, y2 = text_mask.crop_bbox
        refined = text_mask.refined > 0
        if not np.any(refined):
            self.last_text_mask = text_mask
            self._save_debug(debug_name, text_mask, text_mask.source, "empty")
            return full_source if in_place else np.array(full_source, copy=True)

        can_flat, flat_color = self._can_use_flat_fill(
            text_mask.source, text_mask.refined, text_mask.bubble_interior
        )

        if can_flat:
            inpainted_crop = self._apply_flat_fill_with_soft_blend(
                text_mask.source, text_mask.refined, flat_color
            )
            method = "flat_fill_fast"
        elif text_mask.is_uniform_background:
            inpainted_crop = text_mask.source.copy()
            inpainted_crop[refined] = np.asarray(text_mask.background_color, dtype=np.uint8)
            method = "median"
        elif precomputed_crop is not None:
            inpainted_crop = precomputed_crop
            method = "lama_large"
        else:
            lama_mask = _lama_mask_for(text_mask.refined, text_mask.bubble_interior)
            inpainted_crop = self.lama.inpaint(text_mask.source, lama_mask)
            method = "lama_large"

        residual_expansion_passes = 0
        review = False
        # Outlined/anti-aliased glyphs can extend several pixels beyond the
        # first color-selected component.  Continue only while the existing
        # boundary detector proves text-like source pixels immediately adjacent
        # to the approved mask.  The raw CTD envelope, bubble interior and
        # protected structures still bound every pass.
        max_expansion_passes = max(2, min(4, text_mask.dilation_radius + 1))
        for _ in range(max_expansion_passes):
            expanded = self._residual_expansion(text_mask, inpainted_crop)
            if not np.any(expanded > text_mask.refined):
                break
            residual_expansion_passes += 1
            text_mask = replace(text_mask, refined=expanded)
            refined = expanded > 0
            if can_flat:
                inpainted_crop = self._apply_flat_fill_with_soft_blend(
                    text_mask.source, expanded, flat_color
                )
            elif text_mask.is_uniform_background:
                inpainted_crop = text_mask.source.copy()
                inpainted_crop[refined] = np.asarray(text_mask.background_color, dtype=np.uint8)
            else:
                lama_expanded = _lama_mask_for(expanded, text_mask.bubble_interior)
                inpainted_crop = self.lama.inpaint(text_mask.source, lama_expanded)
        review = self._has_boundary_residual(text_mask, inpainted_crop)
        review_cause: str | None = "boundary" if review else None
        # S4: maske-dışı bantta kalmış glif (B19 "M'" sınıfı) — iç-bakan
        # denetçilerin kör noktası. Varsa blok REVIEW (İngilizce korunur).
        _bg_vals = [int(v) for v in np.asarray(text_mask.background_color).ravel()[:3]]
        _bg: tuple[int, int, int] = (_bg_vals[0], _bg_vals[1], _bg_vals[2]) if len(_bg_vals) == 3 else (255, 255, 255)
        if not review and _outside_mask_text(
            text_mask.source,
            inpainted_crop,
            text_mask.refined,
            _bg,
            text_mask.bubble_interior,
        ):
            review = True
            review_cause = "outside"
        # Faz 4a: maske-içi hayalet (LaMa/ortanca artığı).
        if method != "flat_fill_fast":
            try:
                if self._interior_ghost_area(inpainted_crop, text_mask.refined) >= GHOST_MIN_COMPONENT_AREA:
                    review = True
                    review_cause = "ghost"
            except Exception:
                pass
        # Faz 4b: dolgu/zemin uyuşmazlığı (balonsuz beyaz-leke).
        try:
            if self._fill_ring_mismatch(text_mask, inpainted_crop):
                review = True
                review_cause = "ring"
        except Exception:
            pass
        self.last_text_mask = text_mask
        if review and debug_name.startswith("block_"):
            try:
                _bid = int(debug_name.removeprefix("block_"))
                self.review_block_ids.add(_bid)
                self.review_causes[_bid] = review_cause or "unknown"
                if review_cause in ("boundary", "outside"):
                    # Artık-kutu kaydı (ölçüm): global koordinatta.
                    _ox, _oy = int(text_mask.crop_bbox[0]), int(text_mask.crop_bbox[1])
                    if review_cause == "boundary":
                        _crop_boxes = self._boundary_residual_boxes(text_mask, inpainted_crop)
                    else:
                        _crop_boxes = _outside_mask_boxes(
                            text_mask.source, inpainted_crop, text_mask.refined,
                            _bg, text_mask.bubble_interior,
                        )
                    _global_boxes = [
                        [_ox + bx1, _oy + by1, _ox + bx2, _oy + by2]
                        for bx1, by1, bx2, by2 in _crop_boxes
                    ]
                    if _global_boxes:
                        self.review_residual_boxes[_bid] = _global_boxes
            except ValueError:
                pass

        target_canvas = full_source if in_place else np.array(full_source, copy=True)
        destination = target_canvas[y1:y2, x1:x2]
        destination[refined] = inpainted_crop[refined]
        self._save_debug(
            debug_name,
            text_mask,
            inpainted_crop,
            method,
            residual_expansion_passes=residual_expansion_passes,
            review=review,
        )
        # P1 geri-alma: REVIEW'a düşen blokta temizlik geri alınır —
        # YALNIZCA maske-içi pikseller iade edilir (dışarısı hiç
        # dokunulmadı; tüm-kırpıntı iadesi komşu bloğun temizliğini ezerdi).
        # Boş-beyaz/bulaşma/yarım-hasar yerine sapasağlam İngilizce durur.
        # Debug görüntüsü temizlenmiş haliyle saklanır (adli iz korunur).
        if review:
            pristine = np.ascontiguousarray(text_mask.source)
            ph, pw = pristine.shape[:2]
            dh, dw = destination.shape[:2]
            hh, ww = min(ph, dh), min(pw, dw)
            rev_mask = refined[:hh, :ww] if refined.shape != (hh, ww) else refined
            destination[:hh, :ww][rev_mask] = pristine[:hh, :ww][rev_mask]
        return target_canvas



    @staticmethod
    def _can_use_flat_fill(
        image_crop: np.ndarray,
        mask_crop: np.ndarray,
        bubble_interior: np.ndarray | None = None,
        max_std_threshold: float = 14.0,
        overall_std_threshold: float = 12.0,
    ) -> tuple[bool, tuple[int, int, int]]:
        """Maskenin etrafındaki pikselleri analiz ederek düz renkli konuşma balonu kontrolü yapar."""
        import cv2

        if not np.any(mask_crop):
            return False, (255, 255, 255)

        mask = (mask_crop > 0).astype(np.uint8)
        img = np.ascontiguousarray(image_crop)

        # Önce iç-ortanca: maske-içi baskın renk tekdüzeyse dolgu odur.
        # (Büyümüş maskelerin halkası artık sanatta olduğu için halka-medyanı
        # YANLIŞ renk verirdi — kara-leke vakası.) Baskınlık < %50 ise
        # dev glif varsayılır, halka mantığına düşülür.
        interior_px = img[mask > 0].reshape(-1, 3).astype(np.float32)
        if len(interior_px) >= 16:
            med = np.median(interior_px, axis=0)
            dev = np.max(np.abs(interior_px - med), axis=1)
            if float(np.mean(dev < 20.0)) >= 0.5:
                median_color = (
                    int(round(float(med[0]))),
                    int(round(float(med[1]))),
                    int(round(float(med[2]))),
                )
                # Aşırı durum bekçisi: kapkara iç + bembeyaz tekdüze halka =
                # dev glif (logo korumasından kaçmış); halka kazanır.
                # (Koyu balon + parlak ışıma bu bekçiye nadiren takılır:
                # ışıma gradyanı tekdüze değildir.)
                # Aşırı durum bekçisi: kapkara iç + bembeyaz tekdüze halka =
                # dev glif (logo korumasından kaçmış); halka kazanır.
                # (Koyu balon + parlak ışıma bu bekçiye nadiren takılır:
                # ışıma gradyanı tekdüze değildir.)
                ring_probe = _ring_luma_stats(img, mask, bubble_interior)
                if ring_probe is not None:
                    r_med, r_std = ring_probe
                    if (
                        float(np.dot(med, [0.299, 0.587, 0.114])) < 80.0
                        and r_med > 200.0
                        and r_std <= 12.0
                    ):
                        pass  # halka mantığına düş
                    else:
                        return True, median_color
                else:
                    return True, median_color

        # Maske çevresindeki 2-8px halka piksellerini belirle
        d_outer = cv2.dilate(mask, np.ones((9, 9), np.uint8))
        d_inner = cv2.dilate(mask, np.ones((2, 2), np.uint8))
        ring = (d_outer > 0) & (d_inner == 0)

        if bubble_interior is not None and np.any(bubble_interior):
            ring &= (bubble_interior > 0)

        ring_pixels = image_crop[ring]
        if len(ring_pixels) < 16:
            ring_pixels = image_crop[mask == 0]

        if len(ring_pixels) < 8:
            return False, (255, 255, 255)

        std_rgb = np.std(ring_pixels, axis=0)
        max_std = float(np.max(std_rgb))
        overall_std = float(np.std(ring_pixels))

        if overall_std <= overall_std_threshold or max_std <= max_std_threshold:
            median_color = tuple(int(round(c)) for c in np.median(ring_pixels, axis=0))
            return True, median_color

        return False, (255, 255, 255)

    @staticmethod
    def _apply_flat_fill_with_soft_blend(
        source: np.ndarray,
        mask: np.ndarray,
        fill_color: tuple[int, int, int],
    ) -> np.ndarray:
        """Düz renk dolgusunu yumuşak kenar harmanlama (soft-blend) ile uygular."""
        import cv2

        refined = mask > 0
        if not np.any(refined):
            return source.copy()

        # Maske kenarlarına 1-2px Gaussian Blur ile yumuşak geçiş
        mask_float = refined.astype(np.float32)
        blurred_mask = cv2.GaussianBlur(mask_float, (3, 3), 0.8)
        alpha = np.expand_dims(blurred_mask, axis=-1)

        fill_arr = np.full_like(source, fill_color, dtype=np.float32)
        src_float = source.astype(np.float32)

        blended = np.clip(fill_arr * alpha + src_float * (1.0 - alpha), 0, 255).astype(np.uint8)
        # Inside refined mask area, guarantee 100% pure flat fill
        blended[refined] = np.asarray(fill_color, dtype=np.uint8)
        return blended

    @staticmethod
    def _residual_candidates(text_mask: TextMask, result: np.ndarray) -> np.ndarray:
        """Find high-contrast source glyph remnants in the one-pixel mask boundary."""
        import cv2

        mask = text_mask.refined
        if text_mask.dilation_radius <= 0 or not np.any(mask):
            return np.zeros_like(mask)
        ring = (cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0) & (mask == 0)
        allowed = cv2.dilate(text_mask.raw, np.ones((5, 5), np.uint8)) > 0
        gray = cv2.cvtColor(text_mask.source, cv2.COLOR_RGB2GRAY).astype(np.float32)
        result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY).astype(np.float32)
        bg = float(np.dot(np.asarray(text_mask.background_color), [0.299, 0.587, 0.114]))
        source_contrast = np.abs(gray - bg)
        result_contrast = np.abs(result_gray - bg)
        candidate = ring & allowed & (source_contrast >= 24) & (result_contrast >= 18)
        if text_mask.bubble_interior is not None:
            candidate &= text_mask.bubble_interior > 0
        if text_mask.protected is not None:
            candidate &= text_mask.protected == 0
        return candidate.astype(np.uint8) * 255

    @classmethod
    def _residual_expansion(cls, text_mask: TextMask, result: np.ndarray) -> np.ndarray:
        import cv2

        candidates = cls._residual_candidates(text_mask, result)
        if np.count_nonzero(candidates) < 2:
            return text_mask.refined
        tiny = cv2.dilate(candidates, np.ones((3, 3), np.uint8))
        return cv2.bitwise_or(text_mask.refined, tiny)

    @staticmethod
    def _interior_ghost_area(inpainted_crop: np.ndarray, refined_mask: np.ndarray) -> int:
        """Inpaint SONRASI maske içinde kalan zıt piksel alanı.

        Düz-dolguda iç zaten tek renktir (0 döner). LaMa/ortanca yolda
        kalan glif hayaleti (P003 `I`/tırnak artığı, P005 zerreleri) burada
        yakalanır: tekil bileşen eşiği VEYA dağınık toplam eşik.
        Saf numpy/cv2 — model çağrısı yok.
        """
        import cv2

        refined = (np.asarray(refined_mask) > 0)
        if not np.any(refined):
            return 0
        gray = cv2.cvtColor(np.ascontiguousarray(inpainted_crop), cv2.COLOR_RGB2GRAY).astype(np.float32)
        interior = gray[refined]
        if interior.size == 0:
            return 0
        bg = float(np.median(interior))
        dev = (np.abs(gray - bg) >= GHOST_CONTRAST_THRESHOLD) & refined
        total_dev = int(np.count_nonzero(dev))
        if total_dev < GHOST_MIN_COMPONENT_AREA:
            return 0
        # Bileşen-ebat artı toplam-ebat: dağınık zerreler de bayraklanır.
        if total_dev >= GHOST_MIN_TOTAL_AREA:
            return total_dev
        # Bileşen alanı: stub-gürültüsüz yol (labels argümanı geçilmez).
        count, labels = cv2.connectedComponents(dev.astype(np.uint8))
        if count <= 1:
            return 0
        areas = np.bincount(labels.ravel())[1:]
        return int(np.max(areas)) if areas.size else 0

    @staticmethod
    def _expand_mask_in_bubble(mask: TextMask, member_text: str = "") -> TextMask:
        """Maskeyi zemin-bitişik bölgeye taşır (flood-fill).

        Kısmi kutunun glifleri maske dışındadır ve çoğu siyahtır:
        renk-kısıtlı geometrik büyüme onlara ASLA ulaşamaz (siyah piksel
        zemin-rengi değildir). Bunun yerine tohumdan zemin-rengi üzerinden
        taşma yapılır: beyaz balonun tamamı kapsanır, siyah zemin/sivri
        uçlarda durur. Glif çekirdekleri HER HALDE korunur (`| refined`).
        Alan tavanı (12×) komşu balona taşmayı keser.
        Genel: renk-bağıl, metin-bağımsız (member_text yedekte durur).
        """
        import cv2

        del member_text
        refined = (np.asarray(mask.refined) > 0)
        if not np.any(refined):
            return mask
        src = np.ascontiguousarray(mask.source).astype(np.int16)
        # Zemin rengi MESAFE-HALKASINDAN kestirilir: maskeye 7-25px uzaklıktaki
        # piksel medyanı. Yakın-alan (≤7px) glif-ağırlıklı, uzak-alan iki-modlu
        # olur; halka gerçek çevreyi verir. Örneklem <24px ise maske rengine düş.
        # (Maske background_color'ı balonsuz kutularda glif piksellerinden
        # zehirlenebilir.)
        inner = cv2.dilate(
            refined.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        )
        outer = cv2.dilate(
            refined.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
        )
        halo = (outer > 0) & (inner == 0)
        if int(np.count_nonzero(halo)) >= 24:
            bg = np.median(src[halo].reshape(-1, 3).astype(np.float32), axis=0)
        else:
            bg = np.asarray(mask.background_color, dtype=np.float32)
        bg_like = np.max(np.abs(src - bg.reshape(1, 1, 3)), axis=-1) < SECOND_CHANCE_BG_TOLERANCE
        # İŞ 1 SONUCU (geri alındı, Faz 4-kısıt): 7px renksiz köprü
        # (`bg_like | dilate(refined,15x15)`) beyaz dolguyu birleştirir ama
        # UZAK GLİF piksellerini maskeye sokmaz (glifler bg_like dışıdır;
        # kapsama yalnız refined + 7px band). Tam-audit kanıtı (block_0035):
        # grown maske beyazı %97 kaplar, glif pikselleri bant dışında kalır,
        # sınır-artık denetimi REVIEW üretir (0→10 inpaint-REVIEW regresyonu).
        # Kısmi-kutu glifleri maske katmanının ötesindedir (detector-recall
        # işi) — bkz. ROADMAP "FAZ 4-KISIT". Orijinal 3x3 tohum korundu.
        seed_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        seeds = (cv2.dilate(refined.astype(np.uint8), seed_kernel) > 0) & bg_like
        if not np.any(seeds):
            return mask
        count, labels = cv2.connectedComponents(bg_like.astype(np.uint8))
        if count <= 1:
            return mask
        seed_labels = set(np.unique(labels[seeds]))
        seed_labels.discard(0)
        if not seed_labels:
            return mask
        region = np.isin(labels, list(seed_labels))
        refined_area = int(np.count_nonzero(refined))
        if int(np.count_nonzero(region)) > SECOND_CHANCE_MAX_AREA_RATIO * max(1, refined_area):
            return mask
        grown = region | refined
        if mask.bubble_interior is not None and np.any(mask.bubble_interior):
            grown &= (np.asarray(mask.bubble_interior) > 0)
            grown |= refined
        if not np.any(grown):
            return mask
        # Taşma zemini maskeye işlenir: dolgu rengi halka yerine BURADAN
        # doğrulanır (büyümüş maskenin halkası artık sanattadır!).
        flood_bg = tuple(int(round(v)) for v in bg.reshape(-1))
        return replace(
            mask,
            refined=(grown.astype(np.uint8) * 255),
            background_color=flood_bg,
        )

    @staticmethod
    def _fill_ring_mismatch(text_mask: TextMask, inpainted_crop: np.ndarray) -> bool:
        """Dolgu rengi çevre sanatla bağdaşmıyorsa True (P003 beyaz-leke).

        Balon bulunduysa dolgu zaten balon içindedir (meşru). Balon YOKSA
        iç-ortanca ile dış-halka ortancası yakın olmalı; değilse maske
        sanata taşmış demektir → REVIEW (orijinal korunur).
        """
        import cv2

        if text_mask.bubble_found:
            return False
        refined = (np.asarray(text_mask.refined) > 0)
        if int(np.count_nonzero(refined)) < 16:
            return False
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (FILL_RING_WIDTH, FILL_RING_WIDTH))
        ring = (cv2.dilate(refined.astype(np.uint8), kernel) > 0) & (~refined)
        if int(np.count_nonzero(ring)) < 16:
            return False
        inner_lum = _luminance(np.ascontiguousarray(inpainted_crop)[refined])
        ring_lum = _luminance(np.ascontiguousarray(text_mask.source)[ring])
        return abs(inner_lum - ring_lum) > FILL_RING_LUMA_GAP

    @classmethod
    def _has_boundary_residual(cls, text_mask: TextMask, result: np.ndarray) -> bool:
        import cv2

        return len(cls._boundary_residual_boxes(text_mask, result)) > 0

    @classmethod
    def _boundary_residual_boxes(cls, text_mask: TextMask, result: np.ndarray) -> list[tuple[int, int, int, int]]:
        """_has_boundary_residual ile AYNI kapı; geçen bileşenlerin kutuları.

        Kırpıntı-içi (x1, y1, x2, y2) döndürür. Kapı mantığı tek kaynaktır.
        """
        import cv2

        candidates = cls._residual_candidates(text_mask, result)
        total_residual = int(np.count_nonzero(candidates))
        if total_residual < 2:
            return []
        outside_raw = candidates & (text_mask.raw == 0)
        outside_count = int(np.count_nonzero(outside_raw))
        if outside_count < 2:
            return []
        count, _labels, stats, _ = cv2.connectedComponentsWithStats(outside_raw, 8)
        if count <= 1:
            return []
        areas = stats[1:, cv2.CC_STAT_AREA]
        large_components = int(np.sum(areas >= 10))
        total_components = count - 1
        if not (large_components >= 1 and total_components <= 10):
            return []
        boxes: list[tuple[int, int, int, int]] = []
        for idx in range(1, count):
            if int(stats[idx, cv2.CC_STAT_AREA]) < 10:
                continue
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            boxes.append((x, y, x + w, y + h))
        return boxes

    def _save_debug(
        self,
        name: str,
        text_mask: TextMask,
        inpainted: np.ndarray,
        method: str,
        second_pass: bool = False,
        review: bool = False,
        residual_expansion_passes: int | None = None,
    ) -> None:
        import cv2

        expansion_passes = (
            int(residual_expansion_passes)
            if residual_expansion_passes is not None
            else int(bool(second_pass))
        )
        candidates = self._residual_candidates(text_mask, inpainted)
        total_residual = int(np.count_nonzero(candidates))
        if total_residual < 2:
            remaining_residual_pixels = 0
        else:
            outside_raw = candidates & (text_mask.raw == 0)
            outside_count = int(np.count_nonzero(outside_raw))
            if outside_count < 2:
                remaining_residual_pixels = 0
            else:
                count, _, stats, _ = cv2.connectedComponentsWithStats(outside_raw, 8)
                if count <= 1:
                    remaining_residual_pixels = 0
                else:
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    large_components = int(np.sum(areas >= 10))
                    total_components = count - 1
                    if large_components >= 1 and total_components <= 10:
                        remaining_residual_pixels = outside_count
                    else:
                        remaining_residual_pixels = 0
        record = {
            "name": name,
            "crop_bbox": list(text_mask.crop_bbox),
            "method": method,
            "mask_pixels": int(np.count_nonzero(text_mask.refined)),
            "crop_pixels": int(text_mask.refined.size),
            "bubble_found": text_mask.bubble_found,
            "adaptive_dilation": text_mask.dilation_radius,
            "protected_pixels": text_mask.protected_pixels,
            "second_pass": expansion_passes > 0,
            "residual_expansion_passes": expansion_passes,
            "remaining_boundary_residual_pixels": remaining_residual_pixels,
            "review": review,
        }
        self.debug_records.append(record)
        if self.debug_dir is None:
            return
        target = self.debug_dir / name
        target.mkdir(parents=True, exist_ok=True)
        Image.fromarray(text_mask.source, "RGB").save(target / "source.png")
        Image.fromarray(text_mask.raw, "L").save(target / "raw_text_mask.png")
        if text_mask.predicted_segmentation is not None:
            Image.fromarray(text_mask.predicted_segmentation, "L").save(target / "raw_ctd_segmentation.png")
        if text_mask.glyph_refined is not None:
            Image.fromarray(text_mask.glyph_refined, "L").save(target / "upstream_glyph_mask.png")
        Image.fromarray(text_mask.refined, "L").save(target / "refined_text_mask.png")
        if text_mask.bubble_interior is not None:
            Image.fromarray(text_mask.bubble_interior, "L").save(target / "bubble_interior.png")
        if text_mask.protected is not None:
            Image.fromarray(text_mask.protected, "L").save(target / "protected_structures.png")
        text_mask.overlay().save(target / "mask_overlay.png")
        Image.fromarray(inpainted, "RGB").save(target / "inpainted.png")
