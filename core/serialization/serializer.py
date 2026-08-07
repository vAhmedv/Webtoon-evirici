"""Region/Detection serileştirme yardımcıları.

JSON güvenli dict'lere dönüşüm ve geri dönüşüm.
"""

from __future__ import annotations

from typing import Any

from core.detection import BBox, Detection, Region, RegionStatus, RegionType


def bbox_to_dict(bbox: BBox) -> dict[str, int]:
    """BBox'ı dict'e çevirir."""
    return {"x1": bbox.x1, "y1": bbox.y1, "x2": bbox.x2, "y2": bbox.y2}


def dict_to_bbox(data: dict[str, Any]) -> BBox:
    """Dict'ten BBox oluşturur."""
    return BBox(x1=int(data["x1"]), y1=int(data["y1"]), x2=int(data["x2"]), y2=int(data["y2"]))


def detection_to_dict(det: Detection) -> dict[str, Any]:
    """Detection'ı JSON güvenli dict'e çevirir."""
    return {
        "bbox": bbox_to_dict(det.bbox),
        "confidence": float(det.confidence),
        "type": det.type.value,
        "source_window_id": int(det.source_window_id),
        "mask": None,
        "metadata": dict(det.metadata),
    }


def region_to_dict(reg: Region) -> dict[str, Any]:
    """Region'ı JSON güvenli dict'e çevirir."""
    return {
        "id": int(reg.id),
        "global_bbox": bbox_to_dict(reg.global_bbox),
        "type": reg.type.value,
        "detection_confidence": float(reg.detection_confidence),
        "source_window_ids": list(reg.source_window_ids),
        "status": reg.status.value,
        "text": reg.text,
        "ocr_confidence": reg.ocr_confidence,
        "translation": reg.translation,
        "review_reason": reg.review_reason,
    }


def dict_to_detection(data: dict[str, Any]) -> Detection:
    """Dict'ten Detection oluşturur."""
    return Detection(
        bbox=dict_to_bbox(data["bbox"]),
        confidence=float(data["confidence"]),
        type=RegionType(data["type"]),
        source_window_id=int(data["source_window_id"]),
        mask=data.get("mask"),
        metadata=dict(data.get("metadata", {})),
    )


def dict_to_region(data: dict[str, Any]) -> Region:
    """Dict'ten Region oluşturur."""
    return Region(
        id=int(data["id"]),
        global_bbox=dict_to_bbox(data["global_bbox"]),
        type=RegionType(data["type"]),
        detection_confidence=float(data["detection_confidence"]),
        source_window_ids=tuple(int(x) for x in data.get("source_window_ids", [])),
        status=RegionStatus(data.get("status", RegionStatus.AUTO.value)),
        text=data.get("text"),
        ocr_confidence=float(data["ocr_confidence"]) if data.get("ocr_confidence") is not None else None,
        translation=data.get("translation"),
        review_reason=data.get("review_reason"),
    )