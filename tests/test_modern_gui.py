"""Unit tests for the Linear-Style Monochrome Dark GUI with Stepper, Auto-Scroll, and View Mode Diff."""

from __future__ import annotations

import os
from pathlib import Path
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from core.detection.detection import BBox, Region, RegionStatus, RegionType
from core.models import Page
from gui.components.top_bar import TopBar
from gui.components.left_sidebar import LeftSidebar
from gui.components.webtoon_canvas import WebtoonCanvas
from gui.components.right_inspector import RightInspector
from gui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _create_dummy_page(tmp_path: Path, index: int = 0) -> Page:
    img_path = tmp_path / f"page_{index:03d}.png"
    Image.new("RGB", (800, 1200), "white").save(img_path)
    return Page(index=index, path=img_path, width=800, height=1200, y_offset=index * 1200)


def _create_dummy_region(region_id: int = 1, status: RegionStatus = RegionStatus.REVIEW) -> Region:
    return Region(
        id=region_id,
        global_bbox=BBox(100, 100, 300, 200),
        type=RegionType.DIALOGUE,
        detection_confidence=0.95,
        source_window_ids=(0,),
        status=status,
        text="Hello World!",
        ocr_confidence=0.98,
        translation="Merhaba Dünya!",
        review_reason="ambiguous_unknown_review" if status == RegionStatus.REVIEW else None,
    )


def test_top_bar_stepper_and_progress(qapp) -> None:
    top_bar = TopBar()
    assert top_bar is not None
    assert top_bar.title_label.text() == "WEBTOON TRANSLATOR"

    top_bar.set_chapter_info("Chapter 1", 25)
    assert "CHAPTER 1" in top_bar.chapter_badge.text()
    assert "25 PAGES" in top_bar.chapter_badge.text()

    # Stepper Granular Progress Update (Stage 2: OCR with 142/610)
    top_bar.update_step_progress("OCR", current=142, total=610, status="running")
    ocr_step = top_bar.step_widgets["OCR"]
    detect_step = top_bar.step_widgets["DETECT"]

    assert ocr_step._state == "running"
    assert "142 / 610 (%23) • Kalan: 468" in ocr_step.progress_label.text()
    assert detect_step._state == "completed"
    assert detect_step.icon_label.text() == "✓"
    assert top_bar.progress_bar.value() > 15

    # Reset
    top_bar.reset_stages()
    assert ocr_step._state == "idle"
    assert ocr_step.progress_label.text() == "Bekliyor"
    assert top_bar.progress_bar.value() == 0

    top_bar.set_pipeline_running(True)
    assert top_bar.run_btn.isEnabled() is False
    assert top_bar.cancel_btn.isEnabled() is True


def test_left_sidebar_component(qapp, tmp_path: Path) -> None:
    sidebar = LeftSidebar()
    p1 = _create_dummy_page(tmp_path, 0)
    p2 = _create_dummy_page(tmp_path, 1)

    sidebar.load_pages([p1, p2], regions_per_page={0: 5, 1: 3}, reviews_per_page={0: 2, 1: 0})
    assert sidebar.page_list.count() == 2
    assert sidebar.count_badge.text() == "2"

    selected_pages = []
    sidebar.page_selected.connect(selected_pages.append)
    sidebar.select_page(1)
    assert 1 in selected_pages


def test_webtoon_canvas_auto_scroll_and_view_mode(qapp, tmp_path: Path) -> None:
    canvas = WebtoonCanvas()
    p1 = _create_dummy_page(tmp_path, 0)
    p2 = _create_dummy_page(tmp_path, 1)
    r1 = _create_dummy_region(1, RegionStatus.REVIEW)
    r2 = _create_dummy_region(2, RegionStatus.AUTO)

    canvas.load_chapter_pages([p1, p2], [r1, r2])
    assert len(canvas._region_items) == 2
    assert len(canvas._page_y_offsets) == 2

    # Auto-Scroll and Select
    canvas.select_region(1, auto_scroll=True)
    assert canvas._region_items[1]._is_selected is True
    assert canvas._region_items[2]._is_selected is False

    # View Mode Toggle (Original <-> Translated)
    assert canvas._view_mode == "original"
    canvas.toggle_view_mode()
    assert canvas._view_mode == "translated"
    canvas.toggle_view_mode("original")
    assert canvas._view_mode == "original"

    # Before / After Split Mode
    canvas.toggle_split_view()
    assert canvas._view_mode == "split"
    assert len(canvas._page_items) == 2
    assert canvas._page_items[0].view_mode == "split"

    # Split Position Adjustment
    canvas._split_x = 350.0
    for it in canvas._page_items:
        it.set_split_x(350.0)
    assert canvas._page_items[0].split_x == 350.0

    # Hold-to-Peek (Space Key)
    canvas.start_peek_original()
    assert canvas._is_peeking is True
    assert canvas._page_items[0].view_mode == "peek_original"
    canvas.end_peek_original()
    assert canvas._is_peeking is False
    assert canvas._page_items[0].view_mode == "split"

    # Zoom controls
    prev_zoom = canvas._zoom_factor
    canvas.zoom_in()
    assert canvas._zoom_factor > prev_zoom
    prev_zoom = canvas._zoom_factor
    canvas.zoom_out()
    assert canvas._zoom_factor < prev_zoom
    canvas.fit_width()


def test_right_inspector_view_mode_and_actions(qapp) -> None:
    inspector = RightInspector()
    r1 = _create_dummy_region(10, RegionStatus.REVIEW)

    inspector.display_region(r1, current_index=0, total_count=5)
    assert inspector.lbl_id.text() == "#10"
    assert inspector.lbl_type.text() == "DIALOGUE"
    assert inspector.src_text.toPlainText() == "Hello World!"
    assert inspector.tr_text.toPlainText() == "Merhaba Dünya!"
    assert inspector.reason_banner.isHidden() is False

    # View Mode Toggle Signal
    modes = []
    inspector.view_mode_toggled.connect(modes.append)
    inspector.toggle_view_mode()
    assert "translated" in modes

    # Translation editing signal
    edited_translations = []
    inspector.translation_updated.connect(lambda rid, txt: edited_translations.append((rid, txt)))
    inspector.tr_text.setPlainText("Yeni Türkçe Çeviri")
    assert (10, "Yeni Türkçe Çeviri") in edited_translations

    # Confirm action signal
    confirmed = []
    inspector.confirm_requested.connect(confirmed.append)
    inspector.btn_confirm.click()
    assert 10 in confirmed

    # Skip action signal
    skipped = []
    inspector.skip_requested.connect(skipped.append)
    inspector.btn_skip.click()
    assert 10 in skipped


def test_modern_main_window_integration(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    assert window is not None
    assert window.top_bar is not None
    assert window.left_sidebar is not None
    assert window.canvas is not None
    assert window.inspector is not None

    p1 = _create_dummy_page(tmp_path, 0)
    window.open_chapter(tmp_path)
    assert len(window._pages) == 1

    # Test View Mode Synchronization between inspector and canvas
    window.inspector.toggle_view_mode()
    assert window.canvas._view_mode == "translated"

    window.close()
