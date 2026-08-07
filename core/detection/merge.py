"""Duplicate detection merge modülü.

Aynı içerik birden fazla window'da tespit edilirse tek canonical Region
üretmek için IoU ve mesafe bazlı birleştirme yapar.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bbox import BBox
from .detection import Detection, Region, RegionStatus, RegionType


def _regions_are_compatible(a: Region, b: Region) -> bool:
    """İki Region'un aynı türde olup olmadığını kontrol eder."""
    return a.type == b.type


def _compute_merged_bbox(a: BBox, b: BBox) -> BBox:
    """İki bbox'ın kapsayıcı birleşimini üretir."""
    return BBox(
        x1=min(a.x1, b.x1),
        y1=min(a.y1, b.y1),
        x2=max(a.x2, b.x2),
        y2=max(a.y2, b.y2),
    )


def merge_duplicates(
    detections: list[Detection],
    iou_threshold: float = 0.5,
    center_distance_threshold: int = 200,
    min_confidence: float = 0.5,
) -> list[Region]:
    """Window-local Detection listesini global Region listesine çevirir ve
    aynı içerikli olanları birleştirir.

    Girdi: window-local Detection'lar (her biri farklı window'dan gelebilir).
    Çıktı: global canonical Region listesi.

    Args:
        detections: Tüm window'lardan gelen Detection listesi.
        iou_threshold: Üst üste bmx IoU eşiği (0.0-1.0).
        center_distance_threshold: Merkez nokta piksel mesafe eşiği.
        min_confidence: Kalite kapısı minimum güven eşiği.

    Returns:
        Birleştirilmiş canonical Region listesi.
    """
    if not detections:
        return []

    # Detection'ları önce global koordinata çevir
    global_detections: list[tuple[Detection, BBox]] = []
    for det in detections:
        # Detection bbox'ı zaten global olmalı; burada doğrulama yok,
        # dönüşüm pipeline'ının öncesinde yapılması varsayılır.
        global_detections.append((det, det.bbox))

    # Basit greedy merge
    merged: list[Region] = []
    used: set[int] = set()

    for i, (det_i, bbox_i) in enumerate(global_detections):
        if i in used:
            continue

        region_bbox = bbox_i
        region_confidence = det_i.confidence
        source_windows: set[int] = {det_i.source_window_id}

        for j, (det_j, bbox_j) in enumerate(global_detections):
            if j == i or j in used:
                continue
            if not _regions_are_compatible(det_i, det_j):
                continue

            iou = bbox_i.iou(bbox_j)
            if iou < iou_threshold:
                continue

            # Merkez mesafe kontrolü
            ci = bbox_i.center
            cj = bbox_j.center
            dist = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
            if dist > center_distance_threshold:
                continue

            # Birleştir
            region_bbox = _compute_merged_bbox(region_bbox, bbox_j)
            region_confidence = max(region_confidence, det_j.confidence)
            source_windows.add(det_j.source_window_id)
            used.add(j)

        used.add(i)

        # Safety gate ataması (basit kural seti)
        status = _assign_status(det_i.type, region_confidence, min_confidence)

        region = Region(
            id=len(merged),
            global_bbox=region_bbox,
            type=det_i.type,
            detection_confidence=region_confidence,
            source_window_ids=tuple(sorted(source_windows)),
            status=status,
        )
        merged.append(region)

    return merged


def _assign_status(region_type: RegionType, confidence: float, min_conf: float = 0.5) -> RegionStatus:
    """Kalite kapısı başlangıç durumunu belirler.

    Args:
        region_type: Bölge türü.
        confidence: Tespit güven skoru.
        min_conf: Minimum güven eşiği.

    Returns:
        RegionStatus değeri.
    """
    if region_type in (RegionType.SFX, RegionType.WATERMARK):
        return RegionStatus.SKIP

    if region_type == RegionType.UNKNOWN:
        return RegionStatus.REVIEW

    if confidence < min_conf:
        return RegionStatus.REVIEW

    return RegionStatus.AUTO