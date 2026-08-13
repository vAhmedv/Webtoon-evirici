"""Duplicate detection merge modülü.

Aynı içerik birden fazla window'da tespit edilirse tek canonical Region
üretmek için IoU ve merkez mesafe bazlı birleştirme yapar.

Coordinate contract:
- Girdi `Detection` listesi GLOBAL chapter koordinatlıdır (chapter_analyzer
  local→global dönüşümünü yaptıktan sonra gelir).
- `metadata["polygon"]` aynı GLOBAL koordinattadır; merge seed'e bağlı
  olarak korunur. Hiçbir zaman window-local polygon karıştırılmaz.
"""

from __future__ import annotations

from loguru import logger

from .bbox import BBox
from .detection import Detection, Region, RegionStatus, RegionType


def _regions_are_compatible(a: Region, b: Region) -> bool:
    """İki Region'un aynı türde olup olmadığını kontrol eder."""
    return a.type == b.type


def _detections_are_compatible(a: Detection, b: Detection) -> bool:
    """İki Detection'un aynı türde olup olmadığını kontrol eder."""
    return a.type == b.type


def _compute_merged_bbox(a: BBox, b: BBox) -> BBox:
    """İki bbox'ın kapsayıcı birleşimini üretir."""
    return BBox(
        x1=min(a.x1, b.x1),
        y1=min(a.y1, b.y1),
        x2=max(a.x2, b.x2),
        y2=max(a.y2, b.y2),
    )


def _containment_ratio(inner: BBox, outer: BBox) -> float:
    """İç bbox'ın dış bbox içindeki oranını (0.0-1.0) döndürür."""
    inter = inner.intersection(outer)
    if inter is None:
        return 0.0
    return inter.area / inner.area


def _polygon_of(det: Detection) -> list[list[float]] | None:
    """Detection'dan GLOBAL polygon okur (varsa)."""
    meta = det.metadata
    if not isinstance(meta, dict):
        return None
    poly = meta.get("polygon")
    if isinstance(poly, list) and len(poly) > 0:
        return [[float(x), float(y)] for x, y in poly]
    return None


def _select_group_polygon(members: list[Detection]) -> list[list[float]] | None:
    """Merge grubunun üyelerinden polygon seçer.

    Kural (Phase 3D): highest-confidence detection'ın GLOBAL polygon'unu koru.
    Girdi detection'ları zaten global'e dönüştürülmüştür (chapter_analyzer),
    bu yüzden seçilen polygon da GLOBAL koordinattadır.
    """
    if not members:
        return None
    best_conf = -1.0
    best_polygon: list[list[float]] | None = None
    for det in members:
        poly = _polygon_of(det)
        if poly is None:
            continue
        if det.confidence > best_conf:
            best_conf = det.confidence
            best_polygon = poly
    return best_polygon


def _collect_compact_polygons(members: list[Detection], key: str) -> list[list[list[float]]]:
    """Union compact global polygons from overlapping windows with coordinate deduplication."""
    result: list[list[list[float]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for det in sorted(members, key=lambda item: item.confidence, reverse=True):
        polygons = det.metadata.get(key) if isinstance(det.metadata, dict) else None
        if not isinstance(polygons, list):
            continue
        for polygon in polygons:
            if not isinstance(polygon, list) or len(polygon) < 3:
                continue
            normalized = [[float(p[0]), float(p[1])] for p in polygon]
            signature = tuple((round(p[0]), round(p[1])) for p in normalized)
            if signature not in seen:
                seen.add(signature)
                result.append(normalized)
    return result


def merge_duplicates(
    detections: list[Detection],
    iou_threshold: float = 0.5,
    center_distance_threshold: int = 200,
    min_confidence: float = 0.5,
) -> list[Region]:
    """Global koordinatlı Detection listesini canonical Region listesine çevirir ve
    aynı içerikli olanları birleştirir.

    Girdi: GLOBAL koordinatlı Detection'lar (chapter_analyzer tarafından
    local→global dönüşümü yapılmış olmalı).

    Çıktı: global canonical Region listesi. Region içinde koordinat sistemi
    tek olur: ``global_bbox`` ve ``metadata["polygon"]`` ikisi de GLOBAL'dir.

    Args:
        detections: Tüm window'lardan gelen GLOBAL Detection listesi.
        iou_threshold: Üst üste bbox IoU eşiği (0.0-1.0).
        center_distance_threshold: Merkez nokta piksel mesafe eşiği.
        min_confidence: Kalite kapısı minimum güven eşiği.

    Returns:
        Birleştirilmiş canonical Region listesi.
    """
    if not detections:
        return []

    # Girdi zaten global koordinatlıdır; chapter_analyzer dönüşümü yapar.
    global_detections: list[tuple[Detection, BBox]] = [
        (det, det.bbox) for det in detections
    ]

    # Basit greedy merge
    merged: list[Region] = []
    used: set[int] = set()

    for i, (det_i, bbox_i) in enumerate(global_detections):
        if i in used:
            continue

        region_bbox = bbox_i
        region_confidence = det_i.confidence
        source_windows: set[int] = {det_i.source_window_id}
        # Bu grup (region) için merge'e katılan detection'lar. Polygon seçiminde
        # yalnızca bu üyeler arasından seçilir.
        group_members: list[Detection] = [det_i]

        for j, (det_j, bbox_j) in enumerate(global_detections):
            if j == i or j in used:
                continue
            if not _detections_are_compatible(det_i, det_j):
                continue

            iou = bbox_i.iou(bbox_j)
            if iou >= iou_threshold:
                # Standart IoU merge
                pass
            else:
                # Containment kontrolü: window sınırından kaynaklanan partial/full
                # detection'ları yakalamak için. Küçük bbox'ın ~%95'i büyük bbox'ın
                # içindeyse bunlar aynı bubble'ın partial ve full tespitidir.
                containment = _containment_ratio(bbox_i, bbox_j)
                reverse_containment = _containment_ratio(bbox_j,bbox_i)
                max_containment = max(containment, reverse_containment)
                if max_containment < 0.95:
                    continue

            # Merkez mesafe kontrolü
            ci = bbox_i.center
            cj = bbox_j.center
            dist = ((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2) ** 0.5
            if dist > center_distance_threshold:
                continue

            # Kabul edilen her kaynak üyenin geometrisini koru. Containment
            # eşleşmesinde yalnızca büyük kutuyu seçmek, küçük üyenin birkaç
            # piksel dışarı taşan gerçek metin geometrisini kaybedebiliyordu.
            region_bbox = _compute_merged_bbox(region_bbox, bbox_j)
            region_confidence = max(region_confidence, det_j.confidence)
            source_windows.add(det_j.source_window_id)
            group_members.append(det_j)
            used.add(j)

        used.add(i)

        # Safety gate ataması (basit kural seti)
        status = _assign_status(det_i.type, region_confidence, min_confidence)

        # Merged region için polygon: highest-confidence üyenin GLOBAL polygon'u.
        best_polygon = _select_group_polygon(group_members)

        merged_metadata: dict = {}
        seed_meta = det_i.metadata
        if isinstance(seed_meta, dict):
            merged_metadata.update(seed_meta)
        if best_polygon is not None:
            merged_metadata["polygon"] = best_polygon
        else:
            merged_metadata.pop("polygon", None)

        for geometry_key in ("line_polygons", "segmentation_polygons"):
            polygons = _collect_compact_polygons(group_members, geometry_key)
            if polygons:
                merged_metadata[geometry_key] = polygons
            else:
                merged_metadata.pop(geometry_key, None)
        line_memberships: list[dict] = []
        seen_lines: set[tuple[tuple[int, int], ...]] = set()
        for det in group_members:
            det_lines = _collect_compact_polygons([det], "line_polygons")
            for line_index, polygon in enumerate(det_lines):
                signature = tuple((round(point[0]), round(point[1])) for point in polygon)
                if signature in seen_lines:
                    continue
                seen_lines.add(signature)
                line_memberships.append({
                    "line_id": f"w{det.source_window_id}:l{line_index}:{signature[0][0]}:{signature[0][1]}",
                    "polygon": polygon,
                    "detector_block_id": det.metadata.get("ctd_block_id"),
                })
        if line_memberships:
            merged_metadata["ctd_line_memberships"] = line_memberships
        block_ids = sorted({
            str(det.metadata["ctd_block_id"])
            for det in group_members
            if isinstance(det.metadata, dict) and det.metadata.get("ctd_block_id")
        })
        if block_ids:
            merged_metadata["ctd_block_ids"] = block_ids
        block_boxes = [
            list(det.metadata["ctd_block_bbox"])
            for det in sorted(group_members, key=lambda item: item.confidence, reverse=True)
            if isinstance(det.metadata, dict)
            and isinstance(det.metadata.get("ctd_block_bbox"), list)
            and len(det.metadata["ctd_block_bbox"]) == 4
        ]
        if block_boxes:
            merged_metadata["ctd_block_bbox"] = block_boxes[0]
            merged_metadata["ctd_block_bboxes"] = block_boxes

        region = Region(
            id=len(merged),
            global_bbox=region_bbox,
            type=det_i.type,
            detection_confidence=region_confidence,
            source_window_ids=tuple(sorted(source_windows)),
            status=status,
            metadata=merged_metadata,
        )
        logger.debug(
            f"[merge] region={region.id} bbox={region.global_bbox.to_tuple()} "
            f"conf={region_confidence:.3f} windows={region.source_window_ids} "
            f"polygon={'global' if best_polygon else 'none'}"
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
