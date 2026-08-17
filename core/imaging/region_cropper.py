"""Region crop modülü.

Canonical GLOBAL Region'dan OCR için optimize edilmiş crop üretir.
Page boundary crossing durumlarını ve In-GPU zero-copy tensör dilimlemeyi yönetir.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.detection import BBox, Region
from core.models import Page


@dataclass
class RegionCrop:
    """OCR için üretilen crop.

    Attributes:
        image: Crop edilmiş PIL görüntüsü (veya None ise tensor'dan lazily üretilir).
        region_id: Region kimliği.
        global_origin: Crop'un global koordinattaki başlangıcı (x1, y1).
        local_polygon: Crop-local polygon (varsa).
        padding: Uygulanan padding (piksel).
        page_indices: Kullanılan sayfa indeksleri.
        tensor: GPU/CPU torch.Tensor [C, H, W] (opsiyonel).
    """

    image: Image.Image | None
    region_id: int
    global_origin: tuple[int, int]
    local_polygon: list[list[float]] | None = None
    padding: int = 0
    page_indices: tuple[int, ...] = ()
    tensor: Any | None = None

    def to_pil(self) -> Image.Image:
        """PIL görüntüsünü döndürür, tensor varsa lazily oluşturur."""
        if self.image is not None:
            return self.image
        if self.tensor is not None:
            import torch
            t = self.tensor.detach().cpu()
            if t.ndim == 3 and t.shape[0] == 3:
                # CHW -> HWC
                arr = t.permute(1, 2, 0).numpy()
            elif t.ndim == 2:
                arr = t.numpy()
            else:
                arr = t.squeeze().numpy()
            self.image = Image.fromarray(arr)
            return self.image
        raise ValueError("RegionCrop içinde görsel veya tensör bulunamadı")

    def to_tensor(self, device: str = "cuda") -> Any:
        """torch.Tensor döndürür."""
        if self.tensor is not None:
            import torch
            target_device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
            return self.tensor.to(target_device)
        if self.image is not None:
            import torch
            import numpy as np
            arr = np.array(self.image.convert("RGB"))
            t = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
            target_device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
            return t.to(target_device)
        raise ValueError("RegionCrop içinde görsel veya tensör bulunamadı")


class RegionCropper:
    """Canonical Region'dan hem CPU hem de In-GPU sıfır kopyalı crop'lar üretir."""

    def __init__(
        self,
        pages: Sequence[Page],
        coords: GlobalCoordinateSystem,
        padding: int = 20,
        device: str = "cuda",
    ) -> None:
        self._pages = list(pages)
        self._coords = coords
        self._padding = padding
        self._device = device
        self._page_by_index: dict[int, Page] = {p.index: p for p in self._pages}
        self._gpu_page_cache: dict[int, Any] = {}

    def get_page_tensor(self, page: Page, device: str | None = None) -> Any:
        """Sayfayı doğrudan GPU VRAM'e torch.Tensor [3, H, W] olarak yükler ve önbellekler."""
        dev = device or self._device
        import torch

        target_dev = dev if (dev.startswith("cuda") and torch.cuda.is_available()) else "cpu"

        if page.index in self._gpu_page_cache:
            return self._gpu_page_cache[page.index]

        import numpy as np
        import cv2

        # Fast direct reading with OpenCV
        img_bgr = cv2.imdecode(np.fromfile(str(page.path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            with Image.open(page.path) as img:
                arr = np.array(img.convert("RGB"))
        else:
            arr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(target_dev)
        self._gpu_page_cache[page.index] = tensor
        return tensor

    def clear_gpu_cache(self) -> None:
        """GPU üzerindeki sayfa tensörlerini serbest bırakır ve VRAM'i temizler."""
        self._gpu_page_cache.clear()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    def crop_region_gpu(
        self,
        region: Region,
        adaptive_padding: bool = False,
        device: str | None = None,
        create_pil: bool = True,
    ) -> RegionCrop:
        """Region'ı doğrudan GPU VRAM üzerinde (In-GPU Zero-Copy) dilimleyerek kırpar."""
        bbox = region.global_bbox
        if adaptive_padding:
            pad_x = max(4, min(16, int(bbox.width * 0.06)))
            pad_y = max(4, min(16, int(bbox.height * 0.06)))
        else:
            pad_x = pad_y = self._padding

        x1 = max(0, bbox.x1 - pad_x)
        y1 = max(0, bbox.y1 - pad_y)
        x2 = bbox.x2 + pad_x
        y2 = bbox.y2 + pad_y

        relevant_pages = self._coords.pages_in_range(y1, y2)
        if not relevant_pages:
            raise ValueError(
                f"Region {region.id} ile eşleşen sayfa bulunamadı: global Y {y1}-{y2}"
            )

        import torch

        crops: list[torch.Tensor] = []
        page_indices: list[int] = []

        for page in relevant_pages:
            page_global_start = page.y_offset
            page_global_end = page.y_offset + page.height

            local_start = max(0, y1 - page_global_start)
            local_end = min(page.height, y2 - page_global_start)

            if local_end <= local_start:
                continue

            page_tensor = self.get_page_tensor(page, device=device)
            # GPU tensor slicing [3, H, W] -> [3, local_start:local_end, cx1:cx2]
            pw = page_tensor.shape[2]
            cx1 = min(x1, pw)
            cx2 = min(x2, pw)
            if cx2 <= cx1:
                continue

            crop_t = page_tensor[:, local_start:local_end, cx1:cx2]
            crops.append(crop_t)
            page_indices.append(page.index)

        if not crops:
            raise ValueError(
                f"Region {region.id} için crop edilebilir bölge bulunamadı"
            )

        if len(crops) == 1:
            combined_tensor = crops[0]
        else:
            combined_tensor = torch.cat(crops, dim=1)

        # Küçük metin için GPU üzerinde doğrudan ölçekleme (height < 36px)
        ch, cw = combined_tensor.shape[1], combined_tensor.shape[2]
        if ch < 36 and ch > 0 and cw > 0:
            scale = 36.0 / ch
            new_w = max(1, int(cw * scale))
            float_t = combined_tensor.unsqueeze(0).float()
            resized = torch.nn.functional.interpolate(float_t, size=(36, new_w), mode="bilinear", align_corners=False)
            combined_tensor = resized.squeeze(0).clamp(0, 255).byte()

        # Global polygon → crop-local polygon
        local_polygon = None
        polygon = region.metadata.get("polygon") if isinstance(region.metadata, dict) else None
        if isinstance(polygon, list) and len(polygon) > 0:
            local_polygon = [
                [float(px) - float(x1), float(py) - float(y1)]
                for px, py in polygon
            ]

        pil_img = None
        if create_pil:
            arr = combined_tensor.detach().cpu().permute(1, 2, 0).numpy()
            pil_img = Image.fromarray(arr)

        return RegionCrop(
            image=pil_img,
            region_id=region.id,
            global_origin=(x1, y1),
            local_polygon=local_polygon,
            padding=self._padding,
            page_indices=tuple(page_indices),
            tensor=combined_tensor,
        )

    def crop_region(self, region: Region, adaptive_padding: bool = False) -> RegionCrop:
        """Region'dan crop üretir (Varsayılan olarak GPU tabanlı, gerekirse CPU fallback)."""
        try:
            return self.crop_region_gpu(region, adaptive_padding=adaptive_padding, create_pil=True)
        except Exception:
            return self._crop_region_cpu(region, adaptive_padding=adaptive_padding)

    def _crop_region_cpu(self, region: Region, adaptive_padding: bool = False) -> RegionCrop:
        """Standart CPU PIL tabanlı kırpma fallback mekanizması."""
        bbox = region.global_bbox
        if adaptive_padding:
            pad_x = max(4, min(16, int(bbox.width * 0.06)))
            pad_y = max(4, min(16, int(bbox.height * 0.06)))
        else:
            pad_x = pad_y = self._padding

        x1 = max(0, bbox.x1 - pad_x)
        y1 = max(0, bbox.y1 - pad_y)
        x2 = bbox.x2 + pad_x
        y2 = bbox.y2 + pad_y

        relevant_pages = self._coords.pages_in_range(y1, y2)
        if not relevant_pages:
            raise ValueError(
                f"Region {region.id} ile eşleşen sayfa bulunamadı: global Y {y1}-{y2}"
            )

        crops: list[Image.Image] = []
        page_indices: list[int] = []

        for page in relevant_pages:
            page_global_start = page.y_offset
            page_global_end = page.y_offset + page.height

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

        total_h = sum(c.height for c in crops)
        max_w = max(c.width for c in crops)
        combined = Image.new("RGB", (max_w, total_h), (255, 255, 255))

        y_offset = 0
        for crop in crops:
            combined.paste(crop, (0, y_offset))
            y_offset += crop.height

        if combined.height < 36 and combined.height > 0:
            scale = 36.0 / combined.height
            new_w = max(1, int(combined.width * scale))
            combined = combined.resize((new_w, 36), Image.Resampling.LANCZOS)

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
