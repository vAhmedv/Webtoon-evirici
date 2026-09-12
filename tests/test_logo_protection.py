"""Faz 1 logo/art koruması testleri: tamamı sentetik geometri, gerçek bölüm yok.

Anti-overfit: test metinleri uydurmadır ("QZ", "QK" — hiçbir manhwa'ya ait
değil); eşikler sayfa-göreli oranlardır. Gerçek logo kelimeleri teste gömülmez.
"""

from pathlib import Path

from PIL import Image

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.detection.classification import classify_regions
from core.detection.detection import BBox, Region, RegionStatus, RegionType
from core.models import Page


def _coords(tmp_path: Path, width: int = 800, height: int = 1000) -> GlobalCoordinateSystem:
    page_path = tmp_path / "001.png"
    Image.new("RGB", (width, height), "white").save(page_path)
    return GlobalCoordinateSystem((Page(0, page_path, width, height, 0),))


def _region(
    region_id: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    text: str,
    rtype: RegionType = RegionType.UNKNOWN,
    conf: float = 0.99,
) -> Region:
    return Region(
        id=region_id,
        global_bbox=BBox(x1, y1, x2, y2),
        type=rtype,
        detection_confidence=conf,
        source_window_ids=(1,),
        status=RegionStatus.AUTO,
        text=text,
        ocr_confidence=conf,
        metadata={},
    )


def test_top_zone_giant_sparse_logo_skipped(tmp_path: Path) -> None:
    """Sayfa üstü dev seyrek art-yazı (logo imzası) korunur."""
    coords = _coords(tmp_path)
    region = _region(1, 100, 40, 420, 140, "QZ")
    [classified] = classify_regions([region], coords)
    assert classified.status is RegionStatus.SKIP
    assert classified.review_reason == "logo_art_skip"


def test_giant_sparse_logo_skipped_anywhere(tmp_path: Path) -> None:
    """Dev seyrek glif sayfa ortasında da logodur."""
    coords = _coords(tmp_path)
    region = _region(2, 100, 500, 300, 620, "QK")
    [classified] = classify_regions([region], coords)
    assert classified.status is RegionStatus.SKIP
    assert classified.review_reason == "logo_art_skip"


def test_dialogue_typed_logo_fragment_also_skipped(tmp_path: Path) -> None:
    """Logo parçası DIALOGUE tipiyle gelse bile art korunur (SKIP).

    Sebep eski stilize-kural ya da yeni geometrik kural olabilir; ikisi de
    çeviri/inpaint/render dışı bırakır.
    """
    coords = _coords(tmp_path)
    region = _region(3, 100, 40, 420, 140, "QZ", rtype=RegionType.DIALOGUE, conf=0.7)
    [classified] = classify_regions([region], coords)
    assert classified.status is RegionStatus.SKIP
    assert classified.review_reason in {"logo_art_skip", "stylized_art_logo_skip"}


def test_dense_top_narration_strip_untouched(tmp_path: Path) -> None:
    """Yoğun anlatı şeridi (üstte bile olsa) logoya benzemez."""
    coords = _coords(tmp_path)
    text = "UZUN BİR ANLATI CÜMLESİ BURADA YER ALIYOR VE DEVAM EDİYOR"
    region = _region(4, 50, 30, 650, 110, text)
    region = Region(
        id=region.id,
        global_bbox=region.global_bbox,
        type=region.type,
        detection_confidence=region.detection_confidence,
        source_window_ids=region.source_window_ids,
        status=RegionStatus.REVIEW,
        text=region.text,
        ocr_confidence=region.ocr_confidence,
        metadata={},
    )
    [classified] = classify_regions([region], coords)
    assert classified.review_reason != "logo_art_skip"


def test_normal_dialogue_untouched(tmp_path: Path) -> None:
    """Normal diyalog balonu logoya benzemez."""
    coords = _coords(tmp_path)
    region = _region(
        5, 100, 400, 500, 480, "BURADA NORMAL BİR DİYALOG VAR",
        rtype=RegionType.DIALOGUE, conf=0.9,
    )
    [classified] = classify_regions([region], coords)
    assert classified.review_reason != "logo_art_skip"
    assert classified.status is RegionStatus.AUTO


def test_stat_window_like_box_untouched(tmp_path: Path) -> None:
    """Orta boy glifli oyun arayüzü (stat penceresi imzası) logoya benzemez."""
    coords = _coords(tmp_path)
    region = _region(6, 500, 500, 700, 600, "LV 12")
    [classified] = classify_regions([region], coords)
    assert classified.review_reason != "logo_art_skip"


def test_short_dialogue_exclamation_untouched(tmp_path: Path) -> None:
    """Kısa diyalog ünlemi logo sayılmaz."""
    coords = _coords(tmp_path)
    region = _region(7, 100, 40, 300, 120, "HEY!", rtype=RegionType.DIALOGUE, conf=0.7)
    [classified] = classify_regions([region], coords)
    assert classified.review_reason != "logo_art_skip"
