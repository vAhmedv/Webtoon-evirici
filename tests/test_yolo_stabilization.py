"""Phase 3C — YOLO Detector Stabilization testleri."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image, ImageDraw
from PySide6.QtWidgets import QApplication

from core.config import Config, load_config
from core.detection import BBox, Detection, Region, RegionStatus, RegionType
from core.detection.coordinate import window_bbox_to_global
from core.detection.merge import merge_duplicates
from core.serialization.serializer import detection_to_dict, dict_to_detection, region_to_dict, dict_to_region
from providers.detector.registry import get_registry
from providers.detector.yolo8_comic import Yolo8ComicTextDetector
from ui.main_window import MainWindow
from ui.workers.analysis_worker import AnalysisWorker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    app.quit()


def _has_yolo_model() -> bool:
    registry = get_registry()
    if "YOLOv8 Comic Text Segmenter" not in registry.list_providers():
        return False
    try:
        provider = registry.create("YOLOv8 Comic Text Segmenter")
        return provider._model_path.exists()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1. Local → Global coordinate tests
# ---------------------------------------------------------------------------

class TestCoordinateFlow:
    """Local bbox → global bbox dönüşümü."""

    def test_window_y_start_zero(self) -> None:
        local = BBox(x1=100, y1=200, x2=500, y2=400)
        global_bbox = window_bbox_to_global(local, 0)
        assert global_bbox == BBox(x1=100, y1=200, x2=500, y2=400)

    def test_window_y_start_4000(self) -> None:
        local = BBox(x1=100, y1=200, x2=500, y2=400)
        global_bbox = window_bbox_to_global(local, 4000)
        assert global_bbox == BBox(x1=100, y1=4200, x2=500, y2=4400)

    def test_window_y_start_20000(self) -> None:
        local = BBox(x1=100, y1=1025, x2=600, y2=1244)
        global_bbox = window_bbox_to_global(local, 20000)
        assert global_bbox == BBox(x1=100, y1=21025, x2=600, y2=21244)

    def test_final_partial_window(self) -> None:
        """Son window tam 5000 değilse de dönüşüm doğru."""
        local = BBox(x1=50, y1=100, x2=300, y2=800)
        global_bbox = window_bbox_to_global(local, 32000)
        assert global_bbox == BBox(x1=50, y1=32100, x2=300, y2=32800)
        assert global_bbox.y2 == 32000 + 800


# ---------------------------------------------------------------------------
# 2. Overlap duplicate merge tests
# ---------------------------------------------------------------------------

class TestOverlapMerge:
    """Overlap sliding windows'da duplicate merge."""

    def test_overlap_duplicate_merge(self) -> None:
        """Aynı text iki pencerede detect edilirse merge edilmeli."""
        # Window A: global 0-5000, text at global 4500-4700
        det_a = Detection(
            bbox=BBox(x1=100, y1=4500, x2=400, y2=4700),
            confidence=0.8,
            type=RegionType.UNKNOWN,
            source_window_id=0,
        )
        # Window B: global 4000-9000, aynı text at global 4500-4700 (slightly different box)
        det_b = Detection(
            bbox=BBox(x1=120, y1=4510, x2=380, y2=4680),
            confidence=0.7,
            type=RegionType.UNKNOWN,
            source_window_id=1,
        )

        regions = merge_duplicates([det_a, det_b], iou_threshold=0.5, min_confidence=0.5)
        assert len(regions) == 1
        reg = regions[0]
        # Merged bbox should cover both
        assert reg.global_bbox.x1 == 100
        assert reg.global_bbox.y1 == 4500
        assert reg.global_bbox.x2 == 400
        assert reg.global_bbox.y2 == 4700
        assert set(reg.source_window_ids) == {0, 1}
        assert reg.detection_confidence == 0.8

    def test_non_overlapping_regions_stay_separate(self) -> None:
        """Farklı text'ler merge edilmez."""
        det_a = Detection(
            bbox=BBox(x1=100, y1=100, x2=300, y2=300),
            confidence=0.8,
            type=RegionType.UNKNOWN,
            source_window_id=0,
        )
        det_b = Detection(
            bbox=BBox(x1=100, y1=5000, x2=300, y2=5200),
            confidence=0.7,
            type=RegionType.UNKNOWN,
            source_window_id=1,
        )
        regions = merge_duplicates([det_a, det_b], iou_threshold=0.5, min_confidence=0.5)
        assert len(regions) == 2


# ---------------------------------------------------------------------------
# 3. Page boundary crossing test
# ---------------------------------------------------------------------------

class TestPageBoundaryCrossing:
    """Suwayomi page boundary crossing."""

    def test_text_spanning_page_boundary(self) -> None:
        """Sayfa sınırında kalan text doğru global koordinata çevrilmeli."""
        # Sayfa 0: 0-5000, Sayfa 1: 5000-9820
        # Sliding window: 4000-9000 (her iki sayfayı içerir)
        # Window local'de text: y=950-1050 (global 4950-5050)
        local = BBox(x1=200, y1=950, x2=600, y2=1050)
        global_bbox = window_bbox_to_global(local, 4000)
        assert global_bbox == BBox(x1=200, y1=4950, x2=600, y2=5050)
        # Sayfa 0 sınırı 5000, bu text her iki sayfada da yer alıyor
        assert global_bbox.y1 < 5000 < global_bbox.y2


# ---------------------------------------------------------------------------
# 4. YOLO provider tests
# ---------------------------------------------------------------------------

class TestYoloProvider:
    """YOLOv8 provider davranışı."""

    def test_provider_name(self) -> None:
        det = Yolo8ComicTextDetector("/nonexistent/path")
        assert det.name == "YOLOv8 Comic Text Segmenter"

    def test_missing_model_raises(self) -> None:
        det = Yolo8ComicTextDetector("/nonexistent/path")
        with pytest.raises(FileNotFoundError):
            det.load()

    @pytest.mark.skipif(not _has_yolo_model(), reason="YOLO model not available")
    def test_yolo_outputs_local_coordinates(self) -> None:
        """YOLO provider window-local bbox üretmeli."""
        det = Yolo8ComicTextDetector()
        det.load()
        img = Image.new("RGB", (1024, 1024), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((100, 200), "Hello", fill=(0, 0, 0))
        draw.rectangle((50, 400, 300, 500), outline=(0, 0, 0), width=3)

        detections = det.detect(img, window_id=0)
        det.unload()

        assert len(detections) > 0
        for d in detections:
            assert d.bbox.x1 >= 0
            assert d.bbox.y1 >= 0
            assert d.bbox.x2 <= img.width
            assert d.bbox.y2 <= img.height
            assert d.source_window_id == 0
            assert d.type == RegionType.UNKNOWN

    @pytest.mark.skipif(not _has_yolo_model(), reason="YOLO model not available")
    def test_yolo_mask_presence(self) -> None:
        """YOLO segmentation mask/polygon korunmalı."""
        det = Yolo8ComicTextDetector()
        det.load()
        img = Image.new("RGB", (1024, 1024), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((100, 200), "Hello", fill=(0, 0, 0))

        detections = det.detect(img, window_id=0)
        det.unload()

        masked = [d for d in detections if d.metadata.get("polygon") is not None]
        assert len(masked) > 0
        for d in masked:
            poly = d.metadata["polygon"]
            assert len(poly) >= 3
            assert all(len(p) == 2 for p in poly)

    @pytest.mark.skipif(not _has_yolo_model(), reason="YOLO model not available")
    def test_yolo_confidence_threshold(self) -> None:
        """Confidence threshold config'den gelmeli."""
        det = Yolo8ComicTextDetector(confidence_threshold=0.9)
        assert det.confidence_threshold == 0.9
        det.load()
        img = Image.new("RGB", (1024, 1024), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((100, 200), "Hello", fill=(0, 0, 0))

        detections_low = det.detect(img, window_id=0)
        det.confidence_threshold = 0.1
        detections_high = det.detect(img, window_id=0)
        det.unload()

        assert len(detections_high) >= len(detections_low)


# ---------------------------------------------------------------------------
# 5. Serialization tests
# ---------------------------------------------------------------------------

class TestSerialization:
    """Mask/polygon serialization."""

    def test_region_with_metadata_roundtrip(self) -> None:
        region = Region(
            id=0,
            global_bbox=BBox(x1=10, y1=20, x2=110, y2=120),
            type=RegionType.UNKNOWN,
            detection_confidence=0.8,
            source_window_ids=(0, 1),
            status=RegionStatus.REVIEW,
            metadata={"polygon": [[10, 20], [110, 20], [110, 120], [10, 120]]},
        )
        data = region_to_dict(region)
        restored = dict_to_region(data)
        assert restored.metadata["polygon"] == region.metadata["polygon"]
        assert restored.global_bbox == region.global_bbox

    def test_detection_with_polygon_in_metadata(self) -> None:
        det = Detection(
            bbox=BBox(x1=10, y1=20, x2=110, y2=120),
            confidence=0.8,
            type=RegionType.UNKNOWN,
            source_window_id=0,
            metadata={"polygon": [[10, 20], [110, 20], [110, 120], [10, 120]]},
        )
        data = detection_to_dict(det)
        assert data["metadata"]["polygon"] == det.metadata["polygon"]
        restored = dict_to_detection(data)
        assert restored.metadata["polygon"] == det.metadata["polygon"]


# ---------------------------------------------------------------------------
# 6. Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    """Provider registry durumu."""

    def test_yolo_status_is_stable(self) -> None:
        registry = get_registry()
        if "YOLOv8 Comic Text Segmenter" in registry.list_providers():
            assert registry.get_status("YOLOv8 Comic Text Segmenter") == "stable/default"

    def test_ctd_status_is_experimental(self) -> None:
        registry = get_registry()
        if "ComicTextDetector" in registry.list_providers():
            assert registry.get_status("ComicTextDetector") == "experimental"

    def test_dummy_status_is_development(self) -> None:
        registry = get_registry()
        assert registry.get_status("DummyDetector") == "development/test"


# ---------------------------------------------------------------------------
# 7. UI default detector tests
# ---------------------------------------------------------------------------

class TestUIDefaultDetector:
    """UI varsayılan detector davranışı."""

    def test_main_window_defaults_to_yolo_when_available(self, qapp: QApplication) -> None:
        window = MainWindow()
        registry = get_registry()
        providers = registry.list_providers()
        if "YOLOv8 Comic Text Segmenter" in providers:
            try:
                provider = registry.create("YOLOv8 Comic Text Segmenter")
                if provider._model_path.exists():
                    assert window.detector_combo.currentText() == "YOLOv8 Comic Text Segmenter"
                else:
                    assert window.detector_combo.currentIndex() >= 0
            except Exception:
                assert window.detector_combo.currentIndex() >= 0
        else:
            assert window.detector_combo.currentIndex() >= 0
        window.close()

    def test_main_window_no_crash_on_missing_yolo(self, qapp: QApplication) -> None:
        """YOLO model yoksa UI crash etmemeli."""
        window = MainWindow()
        assert window.detector_combo.count() > 0
        window.close()


# ---------------------------------------------------------------------------
# 8. Worker thread safety regression
# ---------------------------------------------------------------------------

class TestWorkerThreadSafety:
    """Phase 3B.1 düzeltmelerinin regresyonu olmaması."""

    def test_worker_creates_provider_in_worker_thread(self, qapp: QApplication) -> None:
        import threading
        from unittest.mock import patch, MagicMock
        from providers.detector.registry import get_registry
        from application.chapter_analyzer import ChapterAnalyzer

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

        def mock_analyze(self, *args, **kwargs):
            detector = kwargs.get("detector")
            if detector is not None:
                detector.detect(None, 0)
            return MagicMock()

        with patch.object(get_registry(), "create", side_effect=lambda name: FakeProvider()), \
             patch.object(ChapterAnalyzer, "analyze", mock_analyze):
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


# ---------------------------------------------------------------------------
# 9. Integration: real chapter with YOLO
# ---------------------------------------------------------------------------

class TestYoloRealChapter:
    """Gerçek chapter üzerinde YOLO pipeline doğrulaması."""

    def test_yolo_chapter_global_coordinates(self, tmp_path: Path) -> None:
        """YOLO ile analiz edilen chapter'ın global koordinatları doğru."""
        if not _has_yolo_model():
            pytest.skip("YOLO model not available")

        chapter_dir = tmp_path / "chapter"
        chapter_dir.mkdir()

        # 3 sayfa, her biri 800x3000
        pages = []
        for i in range(3):
            img = Image.new("RGB", (800, 3000), (240, 240, 240))
            draw = ImageDraw.Draw(img)
            draw.text((50, 200), f"Page {i} text", fill=(0, 0, 0))
            draw.text((50, 1200), f"Page {i} more text", fill=(0, 0, 0))
            path = chapter_dir / f"{i:03d}.webp"
            img.save(path, "WEBP", quality=95)
            pages.append(path)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        from dataclasses import replace
        config = replace(load_config(), window_height=2000, window_overlap=500)

        from application.chapter_analyzer import ChapterAnalyzer
        from providers.detector.registry import get_registry
        analyzer = ChapterAnalyzer(config)
        detector = get_registry().create("YOLOv8 Comic Text Segmenter")
        result = analyzer.analyze(
            chapter_path=chapter_dir,
            output_path=output_dir,
            detector=detector,
            progress_callback=lambda e: None,
        )

        assert len(result.pages) == 3
        assert len(result.windows) > 0

        # Global koordinatlar chapter boyunca doğru aralıkta olmalı
        total_height = sum(3000 for _ in range(3))
        for reg in result.regions:
            assert 0 <= reg.global_bbox.y1 < total_height
            assert 0 <= reg.global_bbox.y2 <= total_height
            assert reg.global_bbox.x1 >= 0
            assert reg.global_bbox.x2 <= 800
            assert reg.global_bbox.y2 > reg.global_bbox.y1
