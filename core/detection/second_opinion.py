"""İkinci-görüş detektör süzgeci (A1): CTD'nin ıskaladığını YOLO tamamlar.

Saf fonksiyonlar — model/torch yok, tamamı sentetik testli. İlke:
TEKLİF eden ikinci detektör, KARAR veren alt-hattır (geçerlilik + OCR
uyumu + verifier çöpü eler). Bu modül yalnız BARİZ çöpleri (kopya, uzak,
zerre) eler; gerisi normal hattan akar. Eşikler oranlı (IoU, komşu-boyuna
göre mesafe, pencere-boyuna göre en-küçük-boy) — bölüme-özel sayı YOK.
"""

from __future__ import annotations

from typing import Sequence

from core.detection import BBox, Detection

# CTD'de varsa kopyadır (0.35 altı = CTD görmemiş).
SECOND_OPINION_MAX_IOU = 0.35
# Yakınlık: en-yakın CTD kutusunun büyük-kenarının bu katı kadar.
# (IS, blok kutusunun dibinde durur; sayfa-ötesi zincir kurulamaz.)
SECOND_OPINION_GAP_RATIO = 0.5
# Zerre tabanı: pencere-boyuna oranlı (1024px pencerede ~8px).
SECOND_OPINION_MIN_SIZE_RATIO = 0.008


def bbox_edge_gap(a: BBox, b: BBox) -> float:
    """İki kutunun kenar-kenara en-kısa uzaklığı (örtüşmede 0)."""
    dx = max(b.x1 - a.x2, a.x1 - b.x2, 0)
    dy = max(b.y1 - a.y2, a.y1 - b.y2, 0)
    return float((dx * dx + dy * dy) ** 0.5)


def max_iou_with(box: BBox, others: Sequence[BBox]) -> float:
    """Kutunun listedeki en-yüksek IoU'su (boş listede 0.0)."""
    best = 0.0
    for other in others:
        best = max(best, box.iou(other))
    return best


def select_second_opinion(
    candidates: Sequence[Detection],
    existing: Sequence[Detection],
    window_height: int,
    window_width: int,
) -> list[Detection]:
    """İkinci-detektör kutularından hatta girecekleri seçer.

    Kabul: CTD ile örtüşmüyor (IoU<0.35) + bir CTD kutusuna komşu-boyuna
    göre yakın + zerre değil. Reddedilen sessizce düşer (iz yok).
    """
    existing_boxes = [d.bbox for d in existing]
    min_h = max(1.0, window_height * SECOND_OPINION_MIN_SIZE_RATIO)
    min_w = max(1.0, window_width * SECOND_OPINION_MIN_SIZE_RATIO)
    accepted: list[Detection] = []
    for det in candidates:
        box = det.bbox
        if box.height < min_h or box.width < min_w:
            continue
        if max_iou_with(box, existing_boxes) >= SECOND_OPINION_MAX_IOU:
            continue
        near = False
        for other in existing:
            gap = bbox_edge_gap(box, other.bbox)
            allow = SECOND_OPINION_GAP_RATIO * max(other.bbox.width, other.bbox.height)
            if gap <= allow:
                near = True
                break
        if not near:
            continue
        accepted.append(det)
    return accepted
