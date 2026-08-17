"""Unit tests for gui/workers/async_page_loader.py and GUI integration."""

from __future__ import annotations

import os
from pathlib import Path
from PIL import Image
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPixmap

from application.cancellation import CancellationToken
from core.models import Page
from gui.components.left_sidebar import LeftSidebar
from gui.components.webtoon_canvas import WebtoonCanvas
from gui.main_window import MainWindow
from gui.workers.async_page_loader import AsyncPageLoaderWorker


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _create_test_page(tmp_path: Path, index: int, width: int = 800, height: int = 1200) -> Page:
    img_path = tmp_path / f"page_{index:03d}.png"
    color = "red" if index % 2 == 0 else "blue"
    Image.new("RGB", (width, height), color=color).save(img_path)
    return Page(index=index, path=img_path, width=width, height=height, y_offset=index * height)


def test_worker_signals_emitted_in_order(qapp, tmp_path: Path) -> None:
    """Worker sırasıyla page_header_ready, thumbnail_ready ve page_loaded sinyallerini emit eder."""
    p1 = _create_test_page(tmp_path, 0, 800, 1000)
    p2 = _create_test_page(tmp_path, 1, 800, 1500)

    headers = []
    thumbnails = []
    loaded_pages = []
    failures = []

    worker = AsyncPageLoaderWorker([p1, p2])
    worker.page_header_ready.connect(lambda idx, w, h: headers.append((idx, w, h)))
    worker.thumbnail_ready.connect(lambda idx, pix: thumbnails.append((idx, not pix.isNull())))
    worker.page_loaded.connect(lambda idx, qimg, np_arr: loaded_pages.append((idx, not qimg.isNull(), np_arr.shape)))
    worker.loading_failed.connect(lambda idx, err: failures.append((idx, err)))

    worker.start()
    worker.wait(5000)
    qapp.processEvents()

    assert len(failures) == 0
    assert len(headers) == 2
    assert headers[0] == (0, 800, 1000)
    assert headers[1] == (1, 800, 1500)

    assert len(thumbnails) == 2
    assert thumbnails[0] == (0, True)
    assert thumbnails[1] == (1, True)

    assert len(loaded_pages) == 2
    assert loaded_pages[0][0] == 0
    assert loaded_pages[0][1] is True
    assert loaded_pages[0][2] == (1000, 800, 3)
    assert loaded_pages[1][0] == 1
    assert loaded_pages[1][1] is True
    assert loaded_pages[1][2] == (1500, 800, 3)


def test_worker_cancellation(qapp, tmp_path: Path) -> None:
    """request_cancel çağrıldığında worker işlem yapmayı durdurur."""
    pages = [_create_test_page(tmp_path, i, 800, 1000) for i in range(10)]
    worker = AsyncPageLoaderWorker(pages)

    loaded_count = 0

    def on_loaded(idx, qimg, arr):
        nonlocal loaded_count
        loaded_count += 1
        # İlk sayfadan sonra hemen iptal et
        if loaded_count == 1:
            worker.request_cancel()

    worker.page_loaded.connect(on_loaded, Qt.DirectConnection)
    worker.start()
    worker.wait(5000)
    qapp.processEvents()

    assert worker.is_cancelled() is True
    assert loaded_count < len(pages)


def test_worker_loading_failed_signal(qapp, tmp_path: Path) -> None:
    """Olmayan dosya için loading_failed sinyali tetiklenir."""
    missing_page = Page(index=0, path=tmp_path / "missing.png", width=800, height=1200, y_offset=0)
    worker = AsyncPageLoaderWorker([missing_page])

    failures = []
    worker.loading_failed.connect(lambda idx, err: failures.append((idx, err)))

    worker.start()
    worker.wait(3000)
    qapp.processEvents()

    assert len(failures) == 1
    assert failures[0][0] == 0
    assert "bulunamadı" in failures[0][1]


def test_canvas_update_page_image(qapp, tmp_path: Path) -> None:
    """WebtoonCanvas update_page_image ile iskeletten gerçek görsele güncellenir."""
    canvas = WebtoonCanvas()
    p1 = _create_test_page(tmp_path, 0, 800, 1200)

    # İskelet başlangıcı
    canvas.load_chapter_pages([p1])
    assert len(canvas._page_items) == 1
    assert canvas._original_pixmaps[0].isNull() is True

    # Asenkron görsel aktarımı
    test_img = QImage(800, 1200, QImage.Format_RGB888)
    test_img.fill(0xFF0000)
    canvas.update_page_image(0, test_img)

    assert canvas._original_pixmaps[0].isNull() is False
    assert canvas._page_items[0].original_pixmap.isNull() is False


def test_left_sidebar_async_thumbnail(qapp, tmp_path: Path) -> None:
    """LeftSidebar update_page_thumbnail ile küçük resim güncellenir."""
    sidebar = LeftSidebar()
    p1 = _create_test_page(tmp_path, 0, 800, 1200)

    sidebar.load_pages([p1])
    item = sidebar.page_list.item(0)
    card = sidebar.page_list.itemWidget(item)
    assert card.thumb_label.text() == "#1"

    # Thumbnail güncelle
    thumb_pix = QPixmap(48, 64)
    thumb_pix.fill(0x00FF00)
    sidebar.update_page_thumbnail(0, thumb_pix)

    assert card.thumb_label.pixmap() is not None
    assert card.thumb_label.pixmap().isNull() is False


def test_main_window_open_chapter_async_loading(qapp, tmp_path: Path) -> None:
    """MainWindow.open_chapter arka plan loader worker'ını başlatır ve UI'ı doldurur."""
    window = MainWindow()
    p1 = _create_test_page(tmp_path, 0, 800, 1200)
    p2 = _create_test_page(tmp_path, 1, 800, 1200)

    window.open_chapter(tmp_path)
    assert len(window._pages) == 2
    assert window._page_loader_worker is not None

    window._page_loader_worker.wait(5000)
    qapp.processEvents()

    # Sayfaların kanvasta ve kenar çubuğunda güncellendiğini doğrula
    assert window.canvas._original_pixmaps[0].isNull() is False
    assert window.canvas._original_pixmaps[1].isNull() is False

    window.close()
