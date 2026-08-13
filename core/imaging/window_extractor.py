"""Window görüntüsü çıkarma modülü.

Sliding window'a denk gelen sayfa bölgelerini crop edip tek bir window
görüntüsünde birleştirir. Görüntüler RAM üzerinde tutulur, kaynak dosyalara
dokunulmaz.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.models import Page, Window

if TYPE_CHECKING:
    pass


@dataclass
class WindowImage:
    """Bir window'a denk gelen çıkarılmış görüntü.

    Attributes:
        window_id: Window kimliği.
        image: PIL görüntüsü (window yüksekliğinde, chapter genişliğinde).
        global_y_start: Global başlangıç Y (inclusive).
        global_y_end: Global bitiş Y (exclusive).
        width: Görüntü genişliği (piksel).
        height: Görüntü yüksekliği (piksel).
        page_indices: Dahil edilen sayfa indeksleri.
    """

    window_id: int
    image: Image.Image
    global_y_start: int
    global_y_end: int
    width: int
    height: int
    page_indices: tuple[int, ...]


def _crop_page_region(
    page: Page,
    local_y_start: int,
    local_y_end: int,
) -> Image.Image:
    """Bir sayfadan belirli yerel Y aralığını crop eder.

    Args:
        page: Sayfa.
        local_y_start: Sayfa içi başlangıç Y (inclusive).
        local_y_end: Sayfa içi bitiş Y (exclusive).

    Returns:
        Crop edilmiş PIL görüntüsü.
    """
    with Image.open(page.path) as img:
        if img.width != page.width or img.height != page.height:
            img = img.resize((page.width, page.height), Image.Resampling.LANCZOS)
        crop = img.crop((0, local_y_start, page.width, local_y_end))
        return crop.copy()


def extract_window_image(
    pages: tuple[Page, ...],
    window: Window,
    coords: GlobalCoordinateSystem,
) -> WindowImage:
    """Window'a denk gelen sayfa bölgelerini çıkarır ve tek görüntüde birleştirir.

    Kullanım:

        pages = tuple(load_chapter(...))
        coords = GlobalCoordinateSystem(pages)
        windows = generate_windows_for_pages(pages, ...)
        for w in windows:
            wi = extract_window_image(pages, w, coords)

    Args:
        pages: Tüm sayfalar (y_offset atanmış olmalı).
        window: Sliding window.
        coords: Global koordinat sistemi.

    Returns:
        WindowImage nesnesi.
    """
    if not pages:
        raise ValueError("Sayfa listesi boş")

    relevant_pages = coords.pages_in_range(window.y_start, window.y_end)

    if not relevant_pages:
        raise ValueError(
            f"Window {window.id} ile eşleşen sayfa bulunamadı: "
            f"global Y {window.y_start}-{window.y_end}"
        )

    crops: list[Image.Image] = []
    for page in relevant_pages:
        # Sayfanın window ile kesiştiği yerel aralık
        local_start = max(0, window.y_start - page.y_offset)
        local_end = min(page.height, window.y_end - page.y_offset)

        if local_end <= local_start:
            continue

        crop = _crop_page_region(page, local_start, local_end)
        crops.append(crop)

    if not crops:
        raise ValueError(
            f"Window {window.id} için crop edilebilir bölge bulunamadı"
        )

    # Crop'ları global Y'ye göre sırala (zaten sıralı olmalı)
    crops.sort(key=lambda img: 0)  # placeholder; pages_in_range sıralı döner

    total_h = sum(img.height for img in crops)
    max_w = max(img.width for img in crops)

    combined = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y_cursor = 0
    for img in crops:
        combined.paste(img, (0, y_cursor))
        y_cursor += img.height

    if combined.height != window.height:
        raise ValueError(
            f"Window görüntüsü yüksekliği uyuşmuyor: "
            f"beklenen {window.height}, bulunan {combined.height}"
        )

    return WindowImage(
        window_id=window.id,
        image=combined,
        global_y_start=window.y_start,
        global_y_end=window.y_end,
        width=max_w,
        height=combined.height,
        page_indices=tuple(p.index for p in relevant_pages),
    )