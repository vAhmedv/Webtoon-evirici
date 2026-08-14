"""Ana pencere."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from application.chapter_analyzer import AnalysisResult, ChapterAnalyzer, ProductionPipelineResult
from application.progress import ProgressEvent
from core.config import Config, load_config
from loguru import logger
from providers.detector.registry import get_registry
from providers.ocr.registry import get_ocr_registry
from ui.widgets.image_viewer import ImageViewer
from ui.widgets.log_panel import LogPanel
from ui.widgets.region_table import RegionTable
from ui.workers.analysis_worker import AnalysisWorker


class MainWindow(QWidget):
    """Ana pencere."""

    def __init__(self, parent: Optional[object] = None) -> None:
        super().__init__(parent)
        logger.debug(f"[THREAD] MainWindow thread id: {threading.get_ident()}")
        self.setWindowTitle("Webtoon Çevirici")
        self.resize(1400, 900)

        self.config = load_config()
        self.analyzer = ChapterAnalyzer(self.config)
        self._worker: Optional[AnalysisWorker] = None
        self._result: Optional[AnalysisResult | ProductionPipelineResult] = None
        self._current_window_index = 0

        self._build_ui()
        self._populate_detectors()

    def closeEvent(self, event) -> None:
        """Pencere kapanırken log handler'ını temizle."""
        if hasattr(self, "log_panel") and self.log_panel is not None:
            self.log_panel.cleanup()
        super().closeEvent(event)

    def _populate_detectors(self) -> None:
        self.detector_combo.clear()
        registry = get_registry()
        providers = registry.list_providers()
        for name in providers:
            self.detector_combo.addItem(name)

        preferred = self.config.detector.provider
        if preferred in providers:
            try:
                provider = registry.create(preferred)
                model_path = getattr(provider, "_model_path", None)
                if model_path and Path(model_path).exists():
                    self.detector_combo.setCurrentText(preferred)
                else:
                    if providers:
                        self.detector_combo.setCurrentIndex(0)
            except Exception:
                if providers:
                    self.detector_combo.setCurrentIndex(0)
        elif providers:
            self.detector_combo.setCurrentIndex(0)
        else:
            self.detector_combo.addItem("DummyDetector")

    def _create_selected_detector(self):
        name = self.detector_combo.currentText()
        if not name:
            return None
        registry = get_registry()
        try:
            return registry.create(name)
        except Exception as e:
            QMessageBox.critical(self, "Detector Error", f"Failed to create detector '{name}':\n{e}")
            return None

    def _populate_ocr(self) -> None:
        self.ocr_combo.clear()
        registry = get_ocr_registry()
        providers = registry.list_providers()
        for name in providers:
            self.ocr_combo.addItem(name)
        if self.ocr_combo.count() == 0:
            self.ocr_combo.addItem("None")
            self.ocr_combo.setEnabled(False)
        else:
            # PaddleOCR-VL-1.6 primary/default olarak seç
            preferred = "PaddleOCR-VL-1.6"
            if preferred in providers:
                self.ocr_combo.setCurrentText(preferred)
            else:
                self.ocr_combo.setCurrentIndex(0)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # --- Top: inputs ---
        inputs_group = QGroupBox("Chapter")
        inputs_layout = QVBoxLayout(inputs_group)

        chapter_layout = QHBoxLayout()
        chapter_layout.addWidget(QLabel("Chapter Folder"))
        self.chapter_edit = QLineEdit()
        self.chapter_edit.setPlaceholderText("Select a chapter folder containing images...")
        self.chapter_edit.setReadOnly(True)
        chapter_layout.addWidget(self.chapter_edit, 1)
        self.chapter_btn = QPushButton("Seç...")
        self.chapter_btn.clicked.connect(self._select_chapter)
        chapter_layout.addWidget(self.chapter_btn)
        inputs_layout.addLayout(chapter_layout)

        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Select output directory...")
        self.output_edit.setReadOnly(True)
        output_layout.addWidget(self.output_edit, 1)
        self.output_btn = QPushButton("Seç...")
        self.output_btn.clicked.connect(self._select_output)
        output_layout.addWidget(self.output_btn)
        inputs_layout.addLayout(output_layout)

        root.addWidget(inputs_group)

        # --- Settings ---
        settings_group = QGroupBox("Pipeline Settings")
        settings_layout = QHBoxLayout(settings_group)

        settings_layout.addWidget(QLabel("Window Height"))
        self.window_height_edit = QLineEdit(str(self.config.window_height))
        self.window_height_edit.setFixedWidth(100)
        settings_layout.addWidget(self.window_height_edit)

        settings_layout.addWidget(QLabel("Overlap"))
        self.overlap_edit = QLineEdit(str(self.config.window_overlap))
        self.overlap_edit.setFixedWidth(100)
        settings_layout.addWidget(self.overlap_edit)

        settings_layout.addWidget(QLabel("Detector"))
        self.detector_combo = QComboBox()
        self._populate_detectors()
        self.detector_combo.setFixedWidth(180)
        settings_layout.addWidget(self.detector_combo)

        settings_layout.addWidget(QLabel("OCR"))
        self.ocr_combo = QComboBox()
        self._populate_ocr()
        self.ocr_combo.setFixedWidth(180)
        settings_layout.addWidget(self.ocr_combo)

        settings_layout.addStretch(1)
        root.addWidget(settings_group)

        # --- Actions ---
        actions_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("Analyze Chapter")
        self.analyze_btn.setMinimumHeight(36)
        self.analyze_btn.clicked.connect(self._start_analysis)
        actions_layout.addWidget(self.analyze_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setMinimumHeight(36)
        self.cancel_btn.clicked.connect(self._cancel_analysis)
        actions_layout.addWidget(self.cancel_btn)

        root.addLayout(actions_layout)

        # --- Progress ---
        progress_layout = QVBoxLayout()
        self.progress_bar = QLabel("")
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.progress_bar.setMinimumHeight(28)
        progress_layout.addWidget(self.progress_bar)

        self.stage_label = QLabel("Ready")
        self.stage_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.stage_label)

        root.addLayout(progress_layout)

        # --- Summary ---
        summary_group = QGroupBox("Summary")
        summary_layout = QHBoxLayout(summary_group)
        self.summary_labels = {}
        for key in ["Pages", "Windows", "Regions", "AUTO", "REVIEW", "SKIP", "Time"]:
            lbl = QLabel(f"{key}: -")
            self.summary_labels[key] = lbl
            summary_layout.addWidget(lbl)
        summary_layout.addStretch(1)
        root.addWidget(summary_group)

        # --- Splitter: preview + regions ---
        splitter = QSplitter(Qt.Horizontal)

        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(QLabel("Window:"))
        self.prev_btn = QPushButton("<")
        self.prev_btn.setFixedWidth(32)
        self.prev_btn.clicked.connect(self._prev_window)
        nav_layout.addWidget(self.prev_btn)
        self.window_nav_edit = QLineEdit("0 / 0")
        self.window_nav_edit.setReadOnly(True)
        self.window_nav_edit.setFixedWidth(60)
        nav_layout.addWidget(self.window_nav_edit)
        self.next_btn = QPushButton(">")
        self.next_btn.setFixedWidth(32)
        self.next_btn.clicked.connect(self._next_window)
        nav_layout.addWidget(self.next_btn)
        nav_layout.addStretch(1)
        preview_layout.addLayout(nav_layout)

        self.image_viewer = ImageViewer()
        preview_layout.addWidget(self.image_viewer)
        splitter.addWidget(preview_group)

        # Regions
        regions_group = QGroupBox("Regions")
        regions_layout = QVBoxLayout(regions_group)
        self.region_table = RegionTable()
        regions_layout.addWidget(self.region_table)
        splitter.addWidget(regions_group)

        splitter.setSizes([900, 500])
        root.addWidget(splitter, 1)

        # --- Log panel ---
        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout(log_group)
        self.log_panel = LogPanel()
        log_layout.addWidget(self.log_panel)
        root.addWidget(self.log_panel, 1)

        self._set_running_state(False)

    # --- Actions ---
    def _select_chapter(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Chapter Folder")
        if not path:
            return
        self.chapter_edit.setText(path)
        p = Path(path)
        if not p.is_dir():
            QMessageBox.critical(self, "Error", "Selected path is not a directory.")
            self.chapter_edit.clear()
            return
        exts = {e.lower() for e in self.config.input_extensions}
        files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts]
        if not files:
            QMessageBox.warning(
                self,
                "No Images",
                f"No supported images found in {path}\nSupported: {', '.join(sorted(exts))}",
            )

    def _select_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_edit.setText(path)

    def _start_analysis(self) -> None:
        chapter_path = self.chapter_edit.text().strip()
        if not chapter_path:
            QMessageBox.warning(self, "Missing Input", "Please select a chapter folder first.")
            return

        output_path = self.output_edit.text().strip()
        if not output_path:
            chapter_dir = Path(chapter_path)
            output_path = str(chapter_dir.parent / "translated" / chapter_dir.name)
            self.output_edit.setText(output_path)

        try:
            if Path(chapter_path).resolve() == Path(output_path).resolve():
                QMessageBox.critical(
                    self,
                    "Unsafe Output Folder",
                    "Output folder must be different from the source chapter folder.",
                )
                return
        except OSError:
            pass

        try:
            window_height = int(self.window_height_edit.text())
            window_overlap = int(self.overlap_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Invalid Settings", "Window height and overlap must be integers.")
            return

        detector_name = self.detector_combo.currentText()
        if not detector_name:
            QMessageBox.warning(self, "Missing Detector", "Please select a detector.")
            self._set_running_state(False)
            return

        registry = get_registry()
        if detector_name not in registry.list_providers():
            QMessageBox.critical(self, "Invalid Detector", f"Detector '{detector_name}' is not available.")
            self._set_running_state(False)
            return

        ocr_name = self.ocr_combo.currentText()
        if not ocr_name or ocr_name == "None":
            ocr_name = None

        self._set_running_state(True)
        self._result = None
        self.region_table.set_regions([])
        self._current_window_index = 0
        self.window_nav_edit.setText("0 / 0")
        self.image_viewer.set_pixmap(QPixmap())
        self._reset_summary()

        self._worker = AnalysisWorker(
            chapter_path=chapter_path,
            output_path=output_path,
            detector_name=detector_name,
            config=self.config,
            ocr_name=ocr_name,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _cancel_analysis(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
            self.cancel_btn.setEnabled(False)

    def _on_progress(self, event: ProgressEvent) -> None:
        self.stage_label.setText(event.stage)
        if event.total > 0:
            pct = int((event.current / event.total) * 100)
            self.progress_bar.setText(f"{event.stage}: {event.current} / {event.total} ({pct}%)")
        else:
            self.progress_bar.setText(event.stage)

    def _on_result(self, result: AnalysisResult | ProductionPipelineResult) -> None:
        self._result = result
        self.region_table.set_regions(result.regions)
        self._update_summary(result)
        self._set_running_state(False)

        if isinstance(result, AnalysisResult) and result.windows:
            self._current_window_index = 0
            self._show_current_window(result)
        elif isinstance(result, ProductionPipelineResult):
            if result.exported_page_paths:
                self.image_viewer.set_window_image(result.exported_page_paths[0])
            QMessageBox.information(
                self,
                "Translation Complete",
                f"Output: {result.output_directory}\n"
                f"Pages: {len(result.exported_page_paths)}\n"
                f"Regions requiring REVIEW: {result.review_required_count}\n"
                f"Skipped regions: {result.skipped_region_count}",
            )

    def _on_error(self, message: str) -> None:
        QMessageBox.critical(self, "Analysis Error", message)
        self._set_running_state(False)

    def _on_cancelled(self) -> None:
        self.stage_label.setText("Cancelled by user")
        self.progress_bar.setText("Cancelled")
        self._set_running_state(False)

    def _prev_window(self) -> None:
        if not isinstance(self._result, AnalysisResult) or not self._result.windows:
            return
        if self._current_window_index > 0:
            self._current_window_index -= 1
            self._show_current_window(self._result)

    def _next_window(self) -> None:
        if not isinstance(self._result, AnalysisResult) or not self._result.windows:
            return
        if self._current_window_index < len(self._result.windows) - 1:
            self._current_window_index += 1
            self._show_current_window(self._result)

    def _show_current_window(self, result: AnalysisResult) -> None:
        window = result.windows[self._current_window_index]
        preview_path = None
        for vp in result.visualization_paths:
            if vp.name == f"window_{window.id:03d}.png":
                preview_path = vp
                break

        if preview_path and preview_path.exists():
            self.image_viewer.set_window_image(preview_path)
        elif result.pages:
            self.image_viewer.set_window_image(result.pages[0].path)

        self.window_nav_edit.setText(f"{self._current_window_index + 1} / {len(result.windows)}")

    # --- Helpers ---
    def _set_running_state(self, running: bool) -> None:
        self.analyze_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.chapter_btn.setEnabled(not running)
        self.output_btn.setEnabled(not running)
        self.window_height_edit.setEnabled(not running)
        self.overlap_edit.setEnabled(not running)

    def _reset_summary(self) -> None:
        for lbl in self.summary_labels.values():
            lbl.setText("-")

    def _update_summary(self, result: AnalysisResult | ProductionPipelineResult) -> None:
        self.summary_labels["Pages"].setText(f"Pages: {len(result.pages)}")
        self.summary_labels["Windows"].setText(f"Windows: {len(result.windows)}")
        self.summary_labels["Regions"].setText(f"Regions: {len(result.regions)}")
        auto_count = sum(1 for region in result.regions if region.status.value == "auto")
        review_count = sum(1 for region in result.regions if region.status.value == "review")
        skip_count = sum(1 for region in result.regions if region.status.value == "skip")
        self.summary_labels["AUTO"].setText(f"AUTO: {auto_count}")
        self.summary_labels["REVIEW"].setText(f"REVIEW: {review_count}")
        self.summary_labels["SKIP"].setText(f"SKIP: {skip_count}")
        self.summary_labels["Time"].setText(f"Time: {result.elapsed_time:.2f}s")
