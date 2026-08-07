"""WindowImage çıkarma testleri."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.imaging import WindowImage, extract_window_image
from core.io.input_loader import load_chapter
from core.models import Page, Window


def _make_page(tmp_path: Path, index: int, width: int, height: int) -> Page:
    path = tmp_path / f"{index:03d}.webp"
    Image.new("RGB", (width, height), (255, 255, 255)).save(path)
    return Page(index=index, path=path.resolve(), width=width, height=height)


def test_single_page_window(tmp_path: Path) -> None:
    """Tek sayfalı window çıkarımı."""
    page = _make_page(tmp_path, 0, 800, 1000)
    pages = (page,)
    coords = GlobalCoordinateSystem(pages)
    window = Window(id=0, y_start=0, y_end=500, page_indices=(0,))

    wi = extract_window_image(pages, window, coords)

    assert isinstance(wi, WindowImage)
    assert wi.window_id == 0
    assert wi.width == 800
    assert wi.height == 500
    assert wi.page_indices == (0,)


def test_window_spanning_two_pages(tmp_path: Path) -> None:
    """İki sayfayı kesen window."""
    p0 = _make_page(tmp_path, 0, 800, 1000)
    p1 = _make_page(tmp_path, 1, 800, 1000)
    # compute y_offset'ları
    from core.coordinate.global_coords import compute_y_offsets
    pages = tuple(compute_y_offsets([p0, p1]))
    coords = GlobalCoordinateSystem(pages)
    window = Window(id=0, y_start=800, y_end=1800, page_indices=(0, 1))

    wi = extract_window_image(pages, window, coords)

    assert wi.height == 1000
    assert wi.width == 800
    assert set(wi.page_indices) == {0, 1}


def test_window_boundary_start_inside_page(tmp_path: Path) -> None:
    """Sayfa içinde başlayan window."""
    page = _make_page(tmp_path, 0, 800, 1000)
    pages = (page,)
    coords = GlobalCoordinateSystem(pages)
    window = Window(id=0, y_start=200, y_end=700, page_indices=(0,))

    wi = extract_window_image(pages, window, coords)

    assert wi.height == 500
    assert wi.width == 800


def test_window_boundary_end_inside_page(tmp_path: Path) -> None:
    """Sayfa içinde biten window."""
    page = _make_page(tmp_path, 0, 800, 1000)
    pages = (page,)
    coords = GlobalCoordinateSystem(pages)
    window = Window(id=0, y_start=0, y_end=700, page_indices=(0,))

    wi = extract_window_image(pages, window, coords)

    assert wi.height == 700


def test_window_image_does_not_modify_source(tmp_path: Path) -> None:
    """Kaynak dosyalar değişmemeli."""
    page = _make_page(tmp_path, 0, 800, 1000)
    pages = (page,)
    coords = GlobalCoordinateSystem(pages)
    window = Window(id=0, y_start=0, y_end=500, page_indices=(0,))

    before = page.path.stat().st_size
    extract_window_image(pages, window, coords)
    after = page.path.stat().st_size

    assert before == after