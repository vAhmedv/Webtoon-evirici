"""CTD ikinci-şans bandı testleri: sentetik YOLO çıktısı, model yok.

Anti-overfit: eşik altı/yakın skorlar genel davranışla test edilir;
hiçbir bölüm/sayfa verisi kullanılmaz.
"""

import numpy as np

from providers.detector.ctd import ComicTextDetector


def _make_detector() -> ComicTextDetector:
    det = ComicTextDetector.__new__(ComicTextDetector)
    det._conf_thresh = 0.4
    det._second_chance_margin = 0.1
    det._nms_thresh = 0.35
    return det


def _blk(cx: float, cy: float, w: float, h: float, obj: float, c0: float) -> list[float]:
    return [cx, cy, w, h, obj, c0, 0.1]


def test_threshold_property_roundtrip_and_clamp() -> None:
    det = _make_detector()
    assert det.confidence_threshold == 0.4
    det.confidence_threshold = 0.55
    assert det.confidence_threshold == 0.55
    det.confidence_threshold = 99.0
    assert det.confidence_threshold == 1.0
    det.confidence_threshold = -5.0
    assert det.confidence_threshold == 0.0
    det.confidence_threshold = 0.4


def test_band_box_flagged_not_dropped() -> None:
    """0.382'lik kutu (DAMMIT vakası) elenmez, işaretlenir."""
    det = _make_detector()
    out = np.array([
        _blk(500, 500, 100, 60, 0.95, 0.95),  # ~0.90 normal
        _blk(200, 200, 80, 50, 0.62, 0.62),  # ~0.384 bant
    ])
    res = det._postprocess_yolo_blocks(out, 1.0, 1.0, 1000, 1000)
    assert len(res) == 2
    band = [r for r in res if r.get("second_chance")]
    normal = [r for r in res if not r.get("second_chance")]
    assert len(band) == 1 and len(normal) == 1
    assert band[0]["confidence"] < 0.4
    assert normal[0]["confidence"] > 0.4


def test_deep_sub_threshold_still_dropped() -> None:
    """Bandın altı (0.25) hâlâ elenir."""
    det = _make_detector()
    out = np.array([_blk(200, 200, 80, 50, 0.5, 0.5)])  # 0.25
    assert det._postprocess_yolo_blocks(out, 1.0, 1.0, 1000, 1000) == []


def test_margin_zero_restores_old_behavior() -> None:
    """Margin 0 = eski katı eşik (geriye uyumluluk)."""
    det = _make_detector()
    det._second_chance_margin = 0.0
    out = np.array([_blk(200, 200, 80, 50, 0.62, 0.62)])
    assert det._postprocess_yolo_blocks(out, 1.0, 1.0, 1000, 1000) == []
