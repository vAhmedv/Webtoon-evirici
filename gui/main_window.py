"""Modern Monochrome Linear-Style Main Window for Webtoon Translator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from application.chapter_analyzer import (
    ChapterAnalyzer,
    ProductionPipelineResult,
)
from application.progress import ProgressEvent
from core.config import Config, load_config
from core.coordinate.global_coords import GlobalCoordinateSystem
from core.detection.detection import Region, RegionStatus, RegionType
from core.io.input_loader import load_chapter
from core.models import Page
from core.serialization.serializer import dict_to_region
from gui.components.top_bar import TopBar
from gui.components.left_sidebar import LeftSidebar
from gui.components.webtoon_canvas import WebtoonCanvas
from gui.components.right_inspector import RightInspector
from gui.components.telemetry_bar import TelemetryStatusBar
from gui.workers.analysis_worker import AnalysisWorker
from gui.workers.async_page_loader import AsyncPageLoaderWorker


class MainWindow(QMainWindow):
    """Linear/Raycast Monochrome Dark Minimalist Webtoon Translator GUI."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Webtoon Translator — Linear Minimal")
        self.resize(1500, 950)
        self.setMinimumSize(1100, 700)

        self.config: Config = load_config()
        self._current_chapter_dir: Optional[Path] = None
        self._pages: list[Page] = []
        self._regions: list[Region] = []
        self._selected_region_index: int = 0
        self._worker: Optional[AnalysisWorker] = None
        self._page_loader_worker: Optional[AsyncPageLoaderWorker] = None

        self._load_stylesheet()
        self._build_ui()
        self._setup_shortcuts()

    def closeEvent(self, event) -> None:
        """Safely cancel and wait for background workers when window closes."""
        if hasattr(self, "telemetry_bar") and hasattr(self.telemetry_bar, "timer"):
            self.telemetry_bar.timer.stop()
        if self._page_loader_worker and self._page_loader_worker.isRunning():
            self._page_loader_worker.request_cancel()
            self._page_loader_worker.wait(1000)
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self._worker.wait(1500)
        super().closeEvent(event)

    def _load_stylesheet(self) -> None:
        qss_path = Path(__file__).resolve().parent / "styles" / "dark_minimal.qss"
        if qss_path.exists():
            self.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    def _build_ui(self) -> None:
        central_widget = QWidget(self)
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. TopBar (48px) with 6-step Stepper & 2px progress bar
        self.top_bar = TopBar(self)
        self.top_bar.open_chapter_clicked.connect(self._on_open_chapter_clicked)
        self.top_bar.run_pipeline_clicked.connect(self._on_run_pipeline_clicked)
        self.top_bar.cancel_pipeline_clicked.connect(self._on_cancel_pipeline_clicked)
        self.top_bar.settings_clicked.connect(self._on_batch_settings_clicked)
        root_layout.addWidget(self.top_bar)

        # 2. Main 3-Column Splitter (LeftSidebar | WebtoonCanvas | RightInspector)
        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)

        # Left Sidebar (240px)
        self.left_sidebar = LeftSidebar(self)
        self.left_sidebar.page_selected.connect(self._on_page_selected)
        self.splitter.addWidget(self.left_sidebar)

        # Center Canvas (Flexible Viewport)
        self.canvas = WebtoonCanvas(self)
        self.canvas.region_selected.connect(self._on_region_selected_from_canvas)
        self.splitter.addWidget(self.canvas)

        # Right Inspector (380px)
        self.inspector = RightInspector(self)
        self.inspector.translation_updated.connect(self._on_translation_updated)
        self.inspector.status_changed.connect(self._on_status_changed)
        self.inspector.navigate_requested.connect(self._on_navigate_requested)
        self.inspector.confirm_requested.connect(self._on_confirm_requested)
        self.inspector.skip_requested.connect(self._on_skip_requested)
        self.splitter.addWidget(self.inspector)

        # Sync View Mode (Diff / Translated) between Canvas and Inspector
        self.canvas.view_mode_changed.connect(self.inspector.sync_view_mode)
        self.inspector.view_mode_toggled.connect(self.canvas.toggle_view_mode)

        # Set initial splitter stretch factors: 0, 1, 0
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([240, 880, 380])

        root_layout.addWidget(self.splitter, 1)

        # 3. Bottom Telemetry Status Bar (30px)
        self.telemetry_bar = TelemetryStatusBar(self)
        root_layout.addWidget(self.telemetry_bar)

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+O"), self, self._on_open_chapter_clicked)
        QShortcut(QKeySequence("Ctrl+R"), self, self._on_run_pipeline_clicked)
        QShortcut(QKeySequence("Ctrl+Shift+B"), self, self._on_batch_settings_clicked)

    def _on_batch_settings_clicked(self) -> None:
        from gui.dialogs.batch_settings_dialog import BatchSettingsDialog
        dlg = BatchSettingsDialog(self)
        dlg.config_applied.connect(self.telemetry_bar.update_batch_badge)
        dlg.exec()

    def open_chapter(self, chapter_dir: str | Path) -> None:
        path = Path(chapter_dir)
        if not path.exists() or not path.is_dir():
            return

        self._current_chapter_dir = path
        try:
            self._pages = list(load_chapter(path, self.config, allow_non_uniform_widths=True))
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load chapter images:\n{e}")
            return

        self.top_bar.set_chapter_info(path.name, len(self._pages))
        self.left_sidebar.load_pages(self._pages)
        self.canvas.load_chapter_pages(self._pages, regions=[])
        self.top_bar.reset_stages()

        # Stop previous async loader if running
        if self._page_loader_worker and self._page_loader_worker.isRunning():
            self._page_loader_worker.request_cancel()
            self._page_loader_worker.wait(1000)

        # Start asynchronous background page loader
        self._page_loader_worker = AsyncPageLoaderWorker(self._pages, parent=self)
        self._page_loader_worker.thumbnail_ready.connect(self.left_sidebar.update_page_thumbnail)
        self._page_loader_worker.page_loaded.connect(self._on_page_loaded_async)
        self._page_loader_worker.start()

        # Check for cached or existing audit analysis and rendered pages
        analysis_json = path / "analysis" / "regions.json"
        rendered_dir: Optional[Path] = None

        if not analysis_json.exists():
            cand1 = Path("audit_output/real_chapter1_e2e/analysis/regions.json")
            cand2 = Path("audit_output/generalization_test/analysis/regions.json")
            if "Chapter 1" in str(path) and cand1.exists():
                analysis_json = cand1
                rendered_dir = Path("audit_output/real_chapter1_e2e/pages")
            elif "Chapter 2" in str(path) and cand2.exists():
                analysis_json = cand2
                rendered_dir = Path("audit_output/generalization_test/pages")

        if analysis_json.exists():
            self._load_regions_from_json(analysis_json, rendered_dir=rendered_dir)

    def _on_page_loaded_async(self, page_index: int, qimage: Any, np_array: Any) -> None:
        """Asenkron olarak çözülen sayfayı kanvasa aktarır."""
        self.canvas.update_page_image(page_index, qimage)

    def _load_regions_from_json(self, json_path: Path, rendered_dir: Optional[Path] = None) -> None:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            raw_regions = data.get("regions", [])
            self._regions = [dict_to_region(r) for r in raw_regions]
        except Exception:
            return

        # Check rendered page files
        rendered_paths: list[Path] = []
        if rendered_dir and rendered_dir.exists():
            for p in self._pages:
                r_file = rendered_dir / p.path.name
                if not r_file.exists():
                    r_file = rendered_dir / f"{p.index:03d}.png"
                if r_file.exists():
                    rendered_paths.append(r_file)

        # Count regions per page
        coords = GlobalCoordinateSystem(tuple(self._pages))
        regions_per_page: dict[int, int] = {}
        reviews_per_page: dict[int, int] = {}

        for r in self._regions:
            center_y = (r.global_bbox.y1 + r.global_bbox.y2) // 2
            pidx, _ = coords.global_to_page(center_y)
            regions_per_page[pidx] = regions_per_page.get(pidx, 0) + 1
            if r.status == RegionStatus.REVIEW:
                reviews_per_page[pidx] = reviews_per_page.get(pidx, 0) + 1

        self.left_sidebar.load_pages(self._pages, regions_per_page, reviews_per_page)
        self.canvas.load_chapter_pages(self._pages, self._regions, rendered_pages=rendered_paths)

        # Select first review region if available, otherwise first region
        if self._regions:
            review_indices = [idx for idx, r in enumerate(self._regions) if r.status == RegionStatus.REVIEW]
            target_idx = review_indices[0] if review_indices else 0
            self._select_region_by_index(target_idx)

    def _select_region_by_index(self, index: int) -> None:
        if 0 <= index < len(self._regions):
            self._selected_region_index = index
            region = self._regions[index]
            self.canvas.select_region(region.id, auto_scroll=True)
            self.inspector.display_region(region, current_index=index, total_count=len(self._regions))

    def _on_open_chapter_clicked(self) -> None:
        start_dir = r"C:\Users\Ahmed\AppData\Local\Tachidesk\downloads\mangas"
        if not Path(start_dir).exists():
            start_dir = str(Path.home())

        chosen = QFileDialog.getExistingDirectory(self, "Select Chapter Folder", start_dir)
        if chosen:
            self.open_chapter(chosen)

    def _on_run_pipeline_clicked(self) -> None:
        if not self._current_chapter_dir or not self._pages:
            QMessageBox.information(self, "No Chapter", "Please open a chapter folder first.")
            return

        # Check for missing AI models
        from core.models.manager import ModelManager
        from gui.dialogs.model_download_dialog import ModelDownloadDialog

        manager = ModelManager()
        missing = manager.get_missing_models()
        if missing:
            dialog = ModelDownloadDialog(manager, missing, parent=self)
            dialog.exec()
            if not dialog.was_successful:
                return

        out_dir = Path("output") / self._current_chapter_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        self.top_bar.reset_stages()
        self.top_bar.set_pipeline_running(True)

        try:
            self.config = load_config()
        except Exception:
            pass

        self._worker = AnalysisWorker(
            chapter_path=self._current_chapter_dir,
            output_path=out_dir,
            detector_name="ComicTextDetector",
            config=self.config,
            parent=self,
        )
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.result.connect(self._on_worker_result)
        self._worker.error.connect(self._on_worker_error)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.start()

    def _on_cancel_pipeline_clicked(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self.top_bar.cancel_btn.setEnabled(False)

    def _on_worker_progress(self, event: ProgressEvent) -> None:
        stage_name = (event.stage or "").lower()
        current_active = "DETECT"

        if "detect" in stage_name or "window" in stage_name:
            current_active = "DETECT"
        elif "ocr" in stage_name:
            current_active = "OCR"
        elif "repair" in stage_name or "qwen" in stage_name:
            current_active = "REPAIR"
        elif "translat" in stage_name:
            current_active = "TRANSLATE"
        elif "inpaint" in stage_name:
            current_active = "INPAINT"
        elif "render" in stage_name or "export" in stage_name:
            current_active = "RENDER"

        self.top_bar.update_step_progress(
            stage_name=current_active,
            current=event.current,
            total=event.total,
            status="running",
        )
        if hasattr(self, "telemetry_bar"):
            self.telemetry_bar.set_status(f"{event.stage or 'Processing'} ({event.current}/{event.total})", is_busy=True)
            self.telemetry_bar.set_active_stage(current_active)

    def _on_worker_result(self, result: ProductionPipelineResult) -> None:
        for _, name in TopBar.STAGES:
            self.top_bar.step_widgets[name].set_state("completed")
        self.top_bar.set_total_progress(100)
        self.top_bar.set_pipeline_running(False)
        if hasattr(self, "telemetry_bar"):
            self.telemetry_bar.set_status("Pipeline Complete", is_busy=False)
            self.telemetry_bar.reset_badges()

        if hasattr(result, "regions") and result.regions:
            self._regions = list(result.regions)
            rendered_paths = getattr(result, "exported_page_paths", getattr(result, "pages", []))
            self.canvas.load_chapter_pages(self._pages, self._regions, rendered_pages=rendered_paths)
            self._select_region_by_index(0)

        QMessageBox.information(self, "Pipeline Complete", f"Chapter translation finished in {result.elapsed_time:.1f}s.")

    def _on_worker_error(self, err: str) -> None:
        self.top_bar.set_pipeline_running(False)
        if hasattr(self, "telemetry_bar"):
            self.telemetry_bar.set_status("Pipeline Error", is_busy=False)
            self.telemetry_bar.reset_badges()
        QMessageBox.critical(self, "Pipeline Error", f"An error occurred during pipeline execution:\n{err}")

    def _on_worker_cancelled(self) -> None:
        self.top_bar.set_pipeline_running(False)
        self.top_bar.reset_stages()
        if hasattr(self, "telemetry_bar"):
            self.telemetry_bar.set_status("Pipeline Cancelled", is_busy=False)
            self.telemetry_bar.reset_badges()
        QMessageBox.information(self, "Cancelled", "Pipeline execution was cancelled.")

    def _on_page_selected(self, page_index: int) -> None:
        self.canvas.scroll_to_page(page_index)

    def _on_region_selected_from_canvas(self, region_id: int) -> None:
        for idx, r in enumerate(self._regions):
            if r.id == region_id:
                self._select_region_by_index(idx)
                break

    def _on_translation_updated(self, region_id: int, new_text: str) -> None:
        from dataclasses import replace
        for idx, r in enumerate(self._regions):
            if r.id == region_id:
                updated_r = replace(r, translation=new_text)
                self._regions[idx] = updated_r
                if hasattr(self.canvas, "_region_items") and region_id in self.canvas._region_items:
                    self.canvas._region_items[region_id].region = updated_r
                break

    def _on_status_changed(self, region_id: int, new_status: RegionStatus) -> None:
        from dataclasses import replace
        for idx, r in enumerate(self._regions):
            if r.id == region_id:
                updated_r = replace(r, status=new_status)
                self._regions[idx] = updated_r
                if hasattr(self.canvas, "_region_items") and region_id in self.canvas._region_items:
                    item = self.canvas._region_items[region_id]
                    item.region = updated_r
                    item._update_appearance()
                break

    def _on_navigate_requested(self, delta: int) -> None:
        new_idx = self._selected_region_index + delta
        if 0 <= new_idx < len(self._regions):
            self._select_region_by_index(new_idx)

    def _on_confirm_requested(self, region_id: int) -> None:
        self._on_status_changed(region_id, RegionStatus.AUTO)
        # Advance to next REVIEW region if available, otherwise next region
        next_review_indices = [
            idx for idx in range(self._selected_region_index + 1, len(self._regions))
            if self._regions[idx].status == RegionStatus.REVIEW
        ]
        if next_review_indices:
            self._select_region_by_index(next_review_indices[0])
        else:
            self._on_navigate_requested(1)

    def _on_skip_requested(self, region_id: int) -> None:
        self._on_status_changed(region_id, RegionStatus.SKIP)
        self._on_navigate_requested(1)

