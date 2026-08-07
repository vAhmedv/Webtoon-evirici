"""UI smoke testleri."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.workers.analysis_worker import AnalysisWorker
from core.config import Config


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    app.quit()


def test_main_window_instantiates(qapp) -> None:
    """MainWindow örneklenebilir."""
    window = MainWindow()
    assert window is not None
    assert window.windowTitle() == "Webtoon Çevirici"
    window.close()


def test_initial_button_states(qapp) -> None:
    """Başlangıçta buton durumları doğru."""
    window = MainWindow()
    assert window.analyze_btn.isEnabled() is True
    assert window.cancel_btn.isEnabled() is False
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
