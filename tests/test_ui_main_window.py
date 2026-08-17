"""UI smoke testleri for modern GUI."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.workers.analysis_worker import AnalysisWorker
from core.config import Config


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_main_window_instantiates(qapp) -> None:
    """MainWindow örneklenebilir."""
    window = MainWindow()
    assert window is not None
    assert "Webtoon" in window.windowTitle()
    assert window.canvas is not None
    assert window.top_bar is not None
    assert window.telemetry_bar is not None
    window.close()


def test_initial_button_states(qapp) -> None:
    """Başlangıçta buton durumları doğru."""
    window = MainWindow()
    assert window.top_bar.run_btn.isEnabled() is True
    assert window.top_bar.open_btn.isEnabled() is True
    assert window.top_bar.cancel_btn.isEnabled() is False
    window.close()


def test_worker_can_be_created() -> None:
    """AnalysisWorker örneklenebilir."""
    config = Config()
    worker = AnalysisWorker(
        chapter_path="/tmp",
        output_path="/tmp/out",
        detector_name="DummyDetector",
        config=config,
    )
    assert worker is not None
    assert worker.isRunning() is False
