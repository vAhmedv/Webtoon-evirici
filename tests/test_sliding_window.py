"""Sliding window testleri."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.coordinate.sliding_window import generate_windows, generate_windows_for_pages
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


def test_generate_windows_basic() -> None:
    """Temel window üretimi doğru olmalı."""
    # Toplam 13000, window 5000, overlap 1000 -> step 4000
    windows = generate_windows(13000, 5000, 1000)

    # Kod, y_end >= total_height olduğunda durur.
    # 12000-13000 penceresi 8000-13000 içinde tamamen gömülmüştür,
    # bu yüzden 3 pencere yeterlidir.
    assert len(windows) == 3

    assert windows[0].id == 0
    assert windows[0].y_start == 0
    assert windows[0].y_end == 5000

    assert windows[1].id == 1
    assert windows[1].y_start == 4000
    assert windows[1].y_end == 9000

    assert windows[2].id == 2
    assert windows[2].y_start == 8000
    assert windows[2].y_end == 13000


def test_generate_windows_small_total() -> None:
    """Toplam yükseklik window'tan küçükse tek window olmalı."""
    windows = generate_windows(3000, 5000, 1000)
    assert len(windows) == 1
    assert windows[0].y_start == 0
    assert windows[0].y_end == 3000


def test_generate_windows_exact_multiple() -> None:
    """Toplam tam window katı ise son window sınırda bitmeli."""
    # 10000 = 2x5000, overlap 1000 -> step 4000
    windows = generate_windows(10000, 5000, 1000)
    assert len(windows) == 3  # 0-5000, 4000-9000, 8000-10000

    assert windows[-1].y_start == 8000
    assert windows[-1].y_end == 10000


def test_generate_windows_zero_total() -> None:
    """Toplam sıfırsa boş liste dönmeli."""
    assert generate_windows(0, 5000, 1000) == []


def test_generate_windows_invalid_params() -> None:
    """Geçersiz parametreler ValueError fırlatmalı."""
    with pytest.raises(ValueError):
        generate_windows(10000, 0, 1000)  # window_height 0

    with pytest.raises(ValueError):
        generate_windows(10000, -100, 1000)  # window_height negatif

    with pytest.raises(ValueError):
        generate_windows(10000, 5000, -100)  # overlap negatif

    with pytest.raises(ValueError):
        generate_windows(10000, 5000, 5000)  # overlap >= window_height


def test_windows_overlap() -> None:
    """Pencereler overlap kadar örtüşmeli."""
    windows = generate_windows(13000, 5000, 1000)

    # Window 0: 0-5000, Window 1: 4000-9000 -> 4000-5000 overlap (1000px)
    assert windows[0].y_end - windows[1].y_start == 1000

    # Window 1: 4000-9000, Window 2: 8000-13000 -> 8000-9000 overlap (1000px)
    assert windows[1].y_end - windows[2].y_start == 1000


def test_generate_windows_for_pages() -> None:
    """generate_windows_for_pages sayfa indekslerini doğru atamalı."""
    # Sayfalar: [0-1000], [1000-1840], [1840-3340] (toplam 3340)
    pages = [
        _make_page(0, height=1000, y_offset=0),
        _make_page(1, height=840, y_offset=1000),
        _make_page(2, height=1500, y_offset=1840),
    ]

    # Toplam 3340 < window 5000 -> tek window, tüm sayfalar
    windows = generate_windows_for_pages(pages, 5000, 1000)
    assert len(windows) == 1
    assert windows[0].page_indices == (0, 1, 2)
    assert windows[0].y_start == 0
    assert windows[0].y_end == 3340


def test_generate_windows_for_pages_multi_window() -> None:
    """Birden çok window'ta sayfa indeksleri doğru atanmalı."""
    # 5 sayfa, her biri 1000px. Toplam 5000. Window 2000, overlap 500 -> step 1500
    pages = [
        _make_page(i, height=1000, y_offset=i * 1000) for i in range(5)
    ]

    windows = generate_windows_for_pages(pages, 2000, 500)

    # Window 0: 0-2000 -> sayfa 0, 1
    assert windows[0].page_indices == (0, 1)

    # Window 1: 1500-3500 -> sayfa 1, 2, 3
    assert windows[1].page_indices == (1, 2, 3)

    # Window 2: 3000-5000 -> sayfa 3, 4
    assert windows[2].page_indices == (3, 4)
    # 3000-5000 penceresi toplamı (5000) kapsadığından durur.
    # 4500-5000 penceresi 3000-5000 içinde gömülmüştir, gereksizdir.


def test_window_contains_y() -> None:
    """contains_y doğru çalışmalı."""
    windows = generate_windows(13000, 5000, 1000)

    # Window 0: 0-5000
    assert windows[0].contains_y(0)
    assert windows[0].contains_y(4999)
    assert not windows[0].contains_y(5000)  # exclusive

    # Window 1: 4000-9000
    assert windows[1].contains_y(4000)
    assert windows[1].contains_y(8999)
    assert not windows[1].contains_y(9000)

    # Overlap bölgesi iki window'ta da olmalı
    assert windows[0].contains_y(4500)
    assert windows[1].contains_y(4500)