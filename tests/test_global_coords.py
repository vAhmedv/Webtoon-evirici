"""Global koordinat sistemi testleri."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.coordinate.global_coords import (
    GlobalCoordinateSystem,
    compute_y_offsets,
)
from core.models import Page


def _make_page(index: int, height: int, y_offset: int = 0, width: int = 800) -> Page:
    """Test Page nesnesi oluşturur."""
    return Page(
        index=index,
        path=Path(f"test/{index:03d}.webp"),
        width=width,
        height=height,
        y_offset=y_offset,
    )


def _make_pages() -> list[Page]:
    """3 sayfalı test listesi: yükseklikler 1000, 840, 1500."""
    pages = [
        _make_page(0, height=1000, y_offset=0),
        _make_page(1, height=840, y_offset=1000),
        _make_page(2, height=1500, y_offset=1840),
    ]
    return pages


def test_compute_y_offsets() -> None:
    """compute_y_offsets kümülatif ofsetler atamalı."""
    pages = [
        _make_page(0, height=1000),
        _make_page(1, height=840),
        _make_page(2, height=1500),
    ]
    result = compute_y_offsets(pages)

    assert result[0].y_offset == 0
    assert result[1].y_offset == 1000
    assert result[2].y_offset == 1840


def test_system_basic_properties() -> None:
    """Sistem temel özellikleri doğru olmalı."""
    system = GlobalCoordinateSystem(tuple(_make_pages()))

    assert system.total_height == 3340
    assert system.width == 800
    assert len(system.pages) == 3


def test_page_to_global() -> None:
    """page_to_global doğru çevirmeli."""
    system = GlobalCoordinateSystem(tuple(_make_pages()))

    # Sayfa 0: y_offset 0
    assert system.page_to_global(0, 0) == 0
    assert system.page_to_global(0, 999) == 999

    # Sayfa 1: y_offset 1000
    assert system.page_to_global(1, 0) == 1000
    assert system.page_to_global(1, 100) == 1100

    # Sayfa 2: y_offset 1840
    assert system.page_to_global(2, 0) == 1840
    assert system.page_to_global(2, 500) == 2340


def test_page_to_global_invalid_page_raises() -> None:
    """Geçersiz sayfa indeksi IndexError fırlatmalı."""
    system = GlobalCoordinateSystem(tuple(_make_pages()))
    with pytest.raises(IndexError):
        system.page_to_global(3, 0)


def test_page_to_global_invalid_local_y_raises() -> None:
    """Geçersiz local_y ValueError fırlatmalı."""
    system = GlobalCoordinateSystem(tuple(_make_pages()))
    with pytest.raises(ValueError):
        system.page_to_global(0, 1000)  # sayfa 0 yüksekliği 1000, geçerli: 0-999


def test_global_to_page() -> None:
    """global_to_page doğru çevirmeli."""
    system = GlobalCoordinateSystem(tuple(_make_pages()))

    assert system.global_to_page(0) == (0, 0)
    assert system.global_to_page(999) == (0, 999)
    assert system.global_to_page(1000) == (1, 0)
    assert system.global_to_page(1100) == (1, 100)
    assert system.global_to_page(1840) == (2, 0)
    assert system.global_to_page(3339) == (2, 1499)


def test_global_to_page_out_of_range_raises() -> None:
    """Sınır dışı global_y ValueError fırlatmalı."""
    system = GlobalCoordinateSystem(tuple(_make_pages()))
    with pytest.raises(ValueError):
        system.global_to_page(-1)
    with pytest.raises(ValueError):
        system.global_to_page(3340)


def test_pages_in_range() -> None:
    """pages_in_range doğru sayfaları döndürmeli."""
    system = GlobalCoordinateSystem(tuple(_make_pages()))

    # Yalnızca sayfa 0
    result = system.pages_in_range(100, 500)
    assert [p.index for p in result] == [0]

    # Sayfa 0 ve 1 sınırda
    result = system.pages_in_range(900, 1100)
    assert [p.index for p in result] == [0, 1]

    # Yalnızca sayfa 1
    result = system.pages_in_range(1000, 1840)
    assert [p.index for p in result] == [1]

    # Tümü
    result = system.pages_in_range(0, 3340)
    assert [p.index for p in result] == [0, 1, 2]

    # Boş aralık
    result = system.pages_in_range(500, 100)
    assert result == []


def test_invalid_y_offsets_raise() -> None:
    """Tutarsız y_offset değerleri ValueError fırlatmalı."""
    pages = [
        _make_page(0, height=1000, y_offset=0),
        _make_page(1, height=840, y_offset=100),  # yanlış: 1000 olmalı
    ]
    with pytest.raises(ValueError):
        GlobalCoordinateSystem(tuple(pages))