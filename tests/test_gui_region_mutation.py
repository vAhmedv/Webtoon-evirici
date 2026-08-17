"""Unit test for GUI region translation and status updating (verifying no FrozenInstanceError)."""

from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from core.detection import BBox, Region, RegionStatus, RegionType
from gui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_region_translation_and_status_update_without_freeze_crash(qapp) -> None:
    """Modifying region translation or status in MainWindow does not raise FrozenInstanceError."""
    window = MainWindow()

    sample_region = Region(
        id=101,
        global_bbox=BBox(x1=10, y1=20, x2=100, y2=80),
        type=RegionType.DIALOGUE,
        detection_confidence=0.95,
        source_window_ids=(0,),
        status=RegionStatus.REVIEW,
        text="Hello world",
        translation="Merhaba dunya",
        review_reason="low_confidence",
    )
    window._regions = [sample_region]

    # 1. Update translation
    window._on_translation_updated(101, "Selam Dünya!")
    assert window._regions[0].translation == "Selam Dünya!"
    assert window._regions[0].status == RegionStatus.REVIEW

    # 2. Update status to AUTO
    window._on_status_changed(101, RegionStatus.AUTO)
    assert window._regions[0].status == RegionStatus.AUTO
    assert window._regions[0].translation == "Selam Dünya!"

    # 3. Update status to SKIP
    window._on_status_changed(101, RegionStatus.SKIP)
    assert window._regions[0].status == RegionStatus.SKIP

    window.close()
