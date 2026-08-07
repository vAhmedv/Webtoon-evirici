"""Sliding window üretici.

Devasa webtoon görüntüsünü tek parça halinde modele vermek yerine
örtüşen pencerelere böler. Böylece model her pencerede çalışabilir
ve overlap sayesinde sınırda kalan balonlar kaybolmaz.
"""

from __future__ import annotations

from loguru import logger

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.models import Window


def generate_windows(
    total_height: int,
    window_height: int,
    overlap: int,
) -> list[Window]:
    """Belirli bir toplam yükseklik için sliding window listesi üretir.

    Args:
        total_height: Global toplam yükseklik (piksel).
        window_height: Her pencerenin yüksekliği (piksel).
        overlap: Pencereler arası örtüşme (piksel).

    Returns:
        Sıralı Window nesneleri listesi.

    Raises:
        ValueError: window_height <= 0 ise.
        ValueError: overlap < 0 veya overlap >= window_height ise.
    """
    if window_height <= 0:
        raise ValueError(f"window_height pozitif olmalı: {window_height}")
    if overlap < 0:
        raise ValueError(f"overlap negatif olamaz: {overlap}")
    if overlap >= window_height:
        raise ValueError(
            f"overlap ({overlap}) window_height'dan ({window_height}) küçük olmalı"
        )

    if total_height <= 0:
        return []

    step = window_height - overlap
    windows: list[Window] = []

    y_start = 0
    window_id = 0
    while y_start < total_height:
        y_end = min(y_start + window_height, total_height)
        windows.append(Window(id=window_id, y_start=y_start, y_end=y_end))
        window_id += 1

        if y_end >= total_height:
            break

        y_start += step

    logger.debug(
        f"Sliding window üretildi: {len(windows)} pencere, "
        f"window_height={window_height}, overlap={overlap}, "
        f"toplam={total_height}px"
    )
    return windows


def generate_windows_for_pages(
    pages,
    window_height: int,
    overlap: int,
) -> list[Window]:
    """Sayfa listesi için sliding window üretir ve her pencereye
    hangi sayfaların dahil olduğunu atar.

    Args:
        pages: Global koordinatlı Page nesneleri listesi.
        window_height: Pencere yüksekliği (piksel).
        overlap: Overlap (piksel).

    Returns:
        Sıralı Window nesneleri listesi. Her window'un page_indices
        alanı, o pencereye denk gelen sayfa indekslerini içerir.
    """
    coords = GlobalCoordinateSystem(tuple(pages))
    total_height = coords.total_height

    windows = generate_windows(total_height, window_height, overlap)

    result: list[Window] = []
    for w in windows:
        page_indices = tuple(p.index for p in coords.pages_in_range(w.y_start, w.y_end))
        result.append(
            Window(
                id=w.id,
                y_start=w.y_start,
                y_end=w.y_end,
                page_indices=page_indices,
            )
        )

    if result:
        ranges = "; ".join(
            f"[{w.y_start}-{w.y_end}] (sayfalar {w.page_indices if w.page_indices else '-'})"
            for w in result
        )
        logger.info(
            f"Window'lar: {len(result)} adet. Aralıklar: {ranges}"
        )

    return result