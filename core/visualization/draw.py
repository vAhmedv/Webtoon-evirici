"""Detection görselleştirme (debug).

Window veya global/merged region listesi üzerinde bbox ve polygon çizer.
"""

from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageDraw

from core.detection import BBox, Detection, Region, RegionStatus, RegionType
from core.detection.coordinate import global_polygon_to_window

# Renk paleti
_COLORS: dict[RegionType, tuple[int, int, int]] = {
    RegionType.DIALOGUE: (0, 200, 0),
    RegionType.NARRATION: (0, 100, 200),
    RegionType.SFX: (255, 165, 0),
    RegionType.WATERMARK: (200, 200, 200),
    RegionType.UNKNOWN: (128, 128, 128),
}

_STATUS_COLORS: dict[RegionStatus, tuple[int, int, int]] = {
    RegionStatus.AUTO: (0, 200, 0),
    RegionStatus.REVIEW: (255, 165, 0),
    RegionStatus.SKIP: (200, 200, 200),
}


def _type_to_color(rtype: RegionType) -> tuple[int, int, int]:
    return _COLORS.get(rtype, (128, 128, 128))


def _status_to_color(status: RegionStatus) -> tuple[int, int, int]:
    return _STATUS_COLORS.get(status, (128, 128, 128))


def draw_detections(
    image: Image.Image,
    detections: Sequence[Detection],
    *,
    window_y_start: int = 0,
) -> Image.Image:
    """PIL görüntü üzerine window-local Detection bbox'ları çizer.

    Args:
        image: Giriş görüntüsü.
        detections: Window-local Detection listesi.
        window_y_start: Window'ın global Y başlangıcı (koordinat dönüşümü için).

    Returns:
        Üzerine çizim yapılmış RGB görüntü.
    """
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)

    for det in detections:
        bbox = det.bbox
        # Görüntü içi koordinat (window-local)
        draw.rectangle(
            [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
            outline=_type_to_color(det.type),
            width=3,
        )
        label = f"{det.type.value}:{det.confidence:.2f}"
        draw.text((bbox.x1, bbox.y1 - 12), label, fill=_type_to_color(det.type))

        # Opsiyonel polygon çizimi (window-local)
        polygon = det.metadata.get("polygon") if isinstance(det.metadata, dict) else None
        if isinstance(polygon, list) and len(polygon) >= 3:
            draw.polygon(
                [(float(x), float(y)) for x, y in polygon],
                outline=(255, 0, 0),
                width=2,
            )

    return out


def draw_regions(
    image: Image.Image,
    regions: Sequence[Region],
    *,
    window_y_start: int = 0,
) -> Image.Image:
    """PIL görüntü üzerine global Region bbox'ları ve polygon'ları çizer.

    Args:
        image: Giriş görüntüsü.
        regions: Global Region listesi.
        window_y_start: Window'ın global Y başlangıcı.

    Returns:
        Üzerine çizim yapılmış RGB görüntü.
    """
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)

    for reg in regions:
        bbox = reg.global_bbox
        # Global -> window-local dönüşüm
        local_bbox = BBox(
            x1=bbox.x1,
            y1=bbox.y1 - window_y_start,
            x2=bbox.x2,
            y2=bbox.y2 - window_y_start,
        )
        if local_bbox.y2 <= 0 or local_bbox.y1 >= image.height:
            continue

        # Görüntü sınırları içinde kırp
        clipped = local_bbox.clip(0, 0, image.width, image.height)
        if clipped is None:
            continue

        color = _status_to_color(reg.status)
        draw.rectangle(
            [clipped.x1, clipped.y1, clipped.x2, clipped.y2],
            outline=color,
            width=3,
        )
        label = f"R{reg.id}:{reg.type.value}:{reg.status.value}"
        draw.text((clipped.x1, max(0, clipped.y1 - 12)), label, fill=color)

        # Global polygon -> window-local polygon çizimi
        polygon = reg.metadata.get("polygon") if isinstance(reg.metadata, dict) else None
        if isinstance(polygon, list) and len(polygon) >= 3:
            local_polygon = global_polygon_to_window(polygon, window_y_start)
            # Görüntü sınırları içinde kırp
            visible_points = []
            for x, y in local_polygon:
                if 0 <= y <= image.height and 0 <= x <= image.width:
                    visible_points.append((x, y))
            if len(visible_points) >= 3:
                draw.polygon(visible_points, outline=(0, 255, 255), width=2)

    return out