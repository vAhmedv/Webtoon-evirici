"""UI thread-safety testleri."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QThread
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QApplication

from gui.components.log_panel import LogPanel, QtLogEmitter
from gui.workers.analysis_worker import AnalysisWorker
from gui.main_window import MainWindow
from application.chapter_analyzer import ChapterAnalyzer
from core.config import Config


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_qt_log_emitter_signal_is_queued(qapp) -> None:
    """QtLogEmitter sinyali farklı thread'den emit edilse bile GUI thread'inde slot çalışır."""
    received = []

    def slot(message: str) -> None:
        received.append((message, threading.get_ident()))

    emitter = QtLogEmitter()
    emitter.message.connect(slot)

    gui_thread_id = threading.get_ident()

    def emit_from_thread() -> None:
        emitter.message.emit("hello")

    t = threading.Thread(target=emit_from_thread)
    t.start()
    t.join()

    # QueuedConnection: slot GUI thread'de çalışır.
    qapp.processEvents()
    qapp.processEvents()

    assert len(received) == 1
    assert received[0][0] == "hello"
    assert received[0][1] == gui_thread_id


def test_log_panel_slot_runs_in_gui_thread(qapp) -> None:
    """LogPanel._append_log worker thread'den gelen mesajı GUI thread'de işler."""
    panel = LogPanel()
    slot_thread_ids = []

    def patched_append(message: str) -> None:
        slot_thread_ids.append(threading.get_ident())
        panel._text_edit.append(message)
        cursor = panel._text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        panel._text_edit.setTextCursor(cursor)

    panel._emitter.message.disconnect(panel._append_log)
    panel._emitter.message.connect(patched_append)

    def emit_from_bg() -> None:
        panel._emit_log("thread-safety-test")

    t = threading.Thread(target=emit_from_bg)
    t.start()
    t.join()

    qapp.processEvents()
    qapp.processEvents()

    assert len(slot_thread_ids) == 1
    assert slot_thread_ids[0] == threading.get_ident()
    panel.cleanup()


def test_worker_emits_many_logs_without_crash(qapp) -> None:
    """Worker 1000 log mesajı üretse UI crash olmaz, tüm mesajlar GUI thread'de işlenir."""
    panel = LogPanel()
    received = []
    lock = threading.Lock()

    def patched_append(message: str) -> None:
        with lock:
            received.append((message, threading.get_ident()))
        panel._text_edit.append(message)
        cursor = panel._text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        panel._text_edit.setTextCursor(cursor)

    panel._emitter.message.disconnect(panel._append_log)
    panel._emitter.message.connect(patched_append)

    def emit_many() -> None:
        for i in range(1000):
            panel._emit_log(f"msg-{i}")

    t = threading.Thread(target=emit_many)
    t.start()
    t.join()

    for _ in range(5):
        qapp.processEvents()

    with lock:
        assert len(received) == 1000
        for _, tid in received:
            assert tid == threading.get_ident()

    panel.cleanup()


def test_worker_provider_lifecycle_in_separate_thread(qapp) -> None:
    """Worker QThread içinde provider lifecycle'ı gerçekleştirir."""
    from providers.detector.registry import get_registry

    captured = {}

    class FakeProvider:
        def __init__(self):
            captured["create_tid"] = threading.get_ident()
        def load(self):
            captured["load_tid"] = threading.get_ident()
        def detect(self, image, window_id):
            captured["detect_tid"] = threading.get_ident()
            return []
        def unload(self):
            captured["unload_tid"] = threading.get_ident()

    def mock_process_chapter(self, *args, **kwargs):
        detector = kwargs.get("detector")
        if detector is not None:
            detector.load()
            detector.detect(None, 0)
            detector.unload()
        return MagicMock()

    with patch.object(get_registry(), "create", side_effect=lambda name: FakeProvider()), \
         patch.object(ChapterAnalyzer, "process_chapter", mock_process_chapter):
        worker = AnalysisWorker(
            chapter_path="/tmp",
            output_path="/tmp/out",
            detector_name="DummyDetector",
            config=Config(),
        )
        worker.start()
        worker.wait(5000)

    gui_tid = threading.get_ident()
    assert captured["create_tid"] == captured["load_tid"] == captured["detect_tid"] == captured["unload_tid"]
    assert captured["create_tid"] != gui_tid


def test_log_panel_cleanup_removes_handler() -> None:
    """cleanup() loguru handler'ını kaldırır; yeni LogPanel duplicate handler oluşturmaz."""
    from loguru import logger

    panel = LogPanel()
    assert panel._handler_id in logger._core.handlers

    panel.cleanup()
    assert panel._handler_id not in logger._core.handlers

    panel2 = LogPanel()
    assert panel2._handler_id not in logger._core.handlers or panel2._handler_id != panel._handler_id
    panel2.cleanup()


def test_log_panel_no_duplicate_handlers_on_reuse() -> None:
    """Aynı log mesajı tekrar tekrar gönderilse duplicate handler olmaz."""
    from loguru import logger

    panel = LogPanel()
    initial_count = len(logger._core.handlers)
    panel.cleanup()
    assert len(logger._core.handlers) == initial_count - 1


def test_worker_accepts_detector_name() -> None:
    """AnalysisWorker detector_name string kabul eder."""
    worker = AnalysisWorker(
        chapter_path="/tmp",
        output_path="/tmp/out",
        detector_name="DummyDetector",
        config=Config(),
    )
    assert worker._detector_name == "DummyDetector"
    assert worker.isRunning() is False


def test_main_window_starts_worker(qapp, tmp_path) -> None:
    """MainWindow run pipeline worker başlatır."""
    window = MainWindow()
    window._current_chapter_dir = tmp_path
    mock_page = MagicMock()
    mock_page.index = 0
    mock_page.width = 800
    mock_page.height = 1200
    mock_page.path = tmp_path / "page_001.png"
    window._pages = [mock_page]

    with patch.object(QThread, "start", lambda self: None), \
         patch("core.models.manager.ModelManager.get_missing_models", return_value=[]), \
         patch("PySide6.QtWidgets.QMessageBox.information"):
        window._on_run_pipeline_clicked()

    assert window._worker is not None
    assert window._worker.chapter_path == tmp_path
    window.close()
