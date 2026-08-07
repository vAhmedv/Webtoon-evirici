"""OCR provider ve RegionCropper testleri."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image, ImageDraw
from PySide6.QtWidgets import QApplication

from core.config import Config, load_config
from core.detection import BBox, Region, RegionStatus, RegionType
from core.imaging.region_cropper import RegionCrop, RegionCropper
from providers.ocr.base import OCRLine, OCRProvider, OCRResult
from providers.ocr.registry import get_ocr_registry
from providers.ocr.rapid_onnx import RapidONNXOCR


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    app.quit()


def _has_ocr_provider(name: str = "RapidOCR-ONNX") -> bool:
    registry = get_ocr_registry()
    return name in registry.list_providers()


# ---------------------------------------------------------------------------
# OCR Provider contract tests
# ---------------------------------------------------------------------------

class TestOCRProviderContract:
    """OCRProvider interface contract."""

    def test_rapid_ocr_provider_name(self) -> None:
        provider = RapidONNXOCR()
        assert provider.name == "RapidOCR-ONNX"

    def test_rapid_ocr_provider_version(self) -> None:
        provider = RapidONNXOCR()
        assert provider.version == "rapidocr-onnx-1.4.4"

    @pytest.mark.skipif(not _has_ocr_provider(), reason="RapidOCR not available")
    def test_rapid_ocr_load_unload(self) -> None:
        provider = RapidONNXOCR()
        assert provider.is_loaded is False
        provider.load()
        assert provider.is_loaded is True
        provider.unload()
        assert provider.is_loaded is False

    @pytest.mark.skipif(not _has_ocr_provider(), reason="RapidOCR not available")
    def test_rapid_ocr_recognize_text(self) -> None:
        provider = RapidONNXOCR()
        provider.load()
        img = Image.new("RGB", (400, 100), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "Hello World", fill=(0, 0, 0))
        result = provider.recognize(img)
        provider.unload()

        assert isinstance(result, OCRResult)
        assert len(result.lines) > 0
        assert "Hello" in result.text or "World" in result.text
        assert result.confidence > 0.0

    @pytest.mark.skipif(not _has_ocr_provider(), reason="RapidOCR not available")
    def test_rapid_ocr_empty_image(self) -> None:
        provider = RapidONNXOCR()
        provider.load()
        img = Image.new("RGB", (400, 100), (255, 255, 255))
        result = provider.recognize(img)
        provider.unload()

        assert isinstance(result, OCRResult)
        assert result.text == ""
        assert result.confidence == 0.0
        assert "empty_ocr_result" in result.warnings


# ---------------------------------------------------------------------------
# RegionCropper tests
# ---------------------------------------------------------------------------

class TestRegionCropper:
    """Region crop tests."""

    def test_single_page_crop(self, tmp_path: Path) -> None:
        """Tek sayfa crop."""
        img = Image.new("RGB", (800, 1000), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((50, 200), "Text", fill=(0, 0, 0))
        page_path = tmp_path / "page_000.webp"
        img.save(page_path, "WEBP", quality=95)

        from core.models import Page
        from core.coordinate.global_coords import GlobalCoordinateSystem

        page = Page(
            index=0,
            path=page_path,
            width=800,
            height=1000,
            y_offset=0,
        )
        coords = GlobalCoordinateSystem([page])
        cropper = RegionCropper([page], coords, padding=10)

        region = Region(
            id=0,
            global_bbox=BBox(x1=50, y1=200, x2=300, y2=240),
            type=RegionType.UNKNOWN,
            detection_confidence=0.8,
            source_window_ids=(0,),
        )
        crop = cropper.crop_region(region)
        assert crop.image.width == 250 + 20  # 250 width + 20 padding
        assert crop.image.height == 40 + 20   # 40 height + 20 padding
        assert crop.global_origin == (40, 190)  # 50-10, 200-10
        assert crop.page_indices == (0,)

    def test_page_boundary_crop(self, tmp_path: Path) -> None:
        """Sayfa sınırı crossing crop."""
        # Page 0: 0-3000, Page 1: 3000-6000
        for i in range(2):
            img = Image.new("RGB", (800, 3000), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text((50, 200), f"Page {i}", fill=(0, 0, 0))
            img.save(tmp_path / f"page_{i:03d}.webp", "WEBP", quality=95)

        from core.models import Page
        from core.coordinate.global_coords import GlobalCoordinateSystem

        pages = [
            Page(index=0, path=tmp_path / "page_000.webp", width=800, height=3000, y_offset=0),
            Page(index=1, path=tmp_path / "page_001.webp", width=800, height=3000, y_offset=3000),
        ]
        coords = GlobalCoordinateSystem(pages)
        cropper = RegionCropper(pages, coords, padding=10)

        region = Region(
            id=0,
            global_bbox=BBox(x1=50, y1=2950, x2=300, y2=3050),
            type=RegionType.UNKNOWN,
            detection_confidence=0.8,
            source_window_ids=(0,),
        )
        crop = cropper.crop_region(region)
        assert crop.image.height == 100 + 20  # 100 height + 20 padding
        assert crop.page_indices == (0, 1)

    def test_polygon_in_crop(self, tmp_path: Path) -> None:
        """Polygon crop-local koordinata çevrilmeli."""
        img = Image.new("RGB", (800, 1000), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((50, 200), "Text", fill=(0, 0, 0))
        page_path = tmp_path / "page_000.webp"
        img.save(page_path, "WEBP", quality=95)

        from core.models import Page
        from core.coordinate.global_coords import GlobalCoordinateSystem

        page = Page(
            index=0,
            path=page_path,
            width=800,
            height=1000,
            y_offset=0,
        )
        coords = GlobalCoordinateSystem([page])
        cropper = RegionCropper([page], coords, padding=10)

        region = Region(
            id=0,
            global_bbox=BBox(x1=50, y1=200, x2=300, y2=240),
            type=RegionType.UNKNOWN,
            detection_confidence=0.8,
            source_window_ids=(0,),
            metadata={"polygon": [[60, 210], [290, 210], [290, 230], [60, 230]]},
        )
        crop = cropper.crop_region(region)
        assert crop.local_polygon is not None
        assert crop.local_polygon == [[20.0, 20.0], [250.0, 20.0], [250.0, 40.0], [20.0, 40.0]]
