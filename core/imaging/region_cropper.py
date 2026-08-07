"""Region crop modülü.

Canonical GLOBAL Region'dan OCR için optimize edilmiş crop üretir.
Page boundary crossing durumlarını da yönetir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.detection import BBox, Region
from core.models import Page


@dataclass(frozen=True)
class RegionCrop:
    """OCR için üretilen crop.

    Attributes:
        image: Crop edilmiş PIL görüntüsü.
        region_id: Region kimliği.
        global_origin: Crop'un global koordinattaki başlangıcı (x1, y1).
        local_polygon: Crop-local polygon (varsa).
        padding: Uygulanan padding (piksel).
        page_indices: Kullanılan sayfa indeksleri.
    """

    image: Image.Image
    region_id: int
    global_origin: tuple[int, int]
    local_polygon: list[list[float]] | None = None
    padding: int = 0
    page_indices: tuple[int, ...] = ()


class RegionCropper:
    """Canonical Region'dan OCR crop'ları üretir."""

    def __init__(
        self,
        pages: Sequence[Page],
        coords: GlobalCoordinateSystem,
        padding: int = 20,
    ) -> None:
        self._pages = list(pages)
        self._coords = coords
        self._padding = padding

    def crop_region(self, region: Region) -> RegionCrop:
        """Region'dan OCR crop'u üretir.

        Args:
            region: Canonical global Region.

        Returns:
            RegionCrop.
        """
        bbox = region.global_bbox
        x1 = max(0, bbox.x1 - self._padding)
        y1 = max(0, bbox.y1 - self._padding)
        x2 = bbox.x2 + self._padding
        y2 = bbox.y2 + self._padding

        relevant_pages = self._coords.pages_in_range(y1, y2)
        if not relevant_pages:
            raise ValueError(
                f"Region {region.id} ile eşleşen sayfa bulunamadı: global Y {y1}-{y2}"
            )

        crops: list[Image.Image] = []
        y_cursor = 0
        page_indices: list[int] = []

        for page in relevant_pages:
            page_global_start = page.y_offset
            page_global_end = page.y_offset + page.height

            # Sayfa ile crop kesişimi
            local_start = max(0, y1 - page_global_start)
            local_end = min(page.height, y2 - page_global_start)

            if local_end <= local_start:
                continue

            with Image.open(page.path) as img:
                crop = img.crop((x1, local_start, x2, local_end))
                crops.append(crop.copy())
                page_indices.append(page.index)

        if not crops:
            raise ValueError(
                f"Region {region.id} için crop edilebilir bölge bulunamadı"
            )

        # Crop'ları global Y'ye göre birleştir
        total_h = sum(c.height for c in crops)
        max_w = max(c.width for c in crops)
        combined = Image.new("RGB", (max_w, total_h), (255, 255, 255))

        y_offset = 0
        for crop in crops:
            combined.paste(crop, (0, y_offset))
            y_offset += crop.height

        # Global polygon → crop-local polygon dönüşümü
        local_polygon = None
        polygon = region.metadata.get("polygon") if isinstance(region.metadata, dict) else None
        if isinstance(polygon, list) and len(polygon) > 0:
            local_polygon = [
                [float(x) - float(x1), float(y) - float(y1)]
                for x, y in polygon
            ]

        return RegionCrop(
            image=combined,
            region_id=region.id,
            global_origin=(x1, y1),
            local_polygon=local_polygon,
            padding=self._padding,
            page_indices=tuple(page_indices),
        )
