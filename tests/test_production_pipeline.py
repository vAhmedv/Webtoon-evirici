"""Unit and end-to-end synthetic tests for the production pipeline."""

from __future__ import annotations

from pathlib import Path
import pytest
from PIL import Image

from application.chapter_analyzer import ChapterAnalyzer, ProductionPipelineResult
from core.detection import BBox, Detection, Region, RegionStatus, RegionType
from core.imaging.inpainter import Inpainter
from core.imaging.renderer import TextRenderer
from core.io.output_exporter import export_chapter_pages
from core.models import Page
from providers.detector.base import DetectorProvider
from providers.ocr.base import OCRProvider, OCRResult
from providers.translation.base import TranslationInput, TranslationItem, TranslationOutput, TranslationOutputItem, TranslationProvider


class DummyDetector(DetectorProvider):
    """Dummy detector provider for testing."""

    def __init__(self) -> None:
        self.loaded = False
        self.unloaded = False

    @property
    def name(self) -> str:
        return "DummyDetector"

    @property
    def version(self) -> str:
        return "1.0"

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.unloaded = True

    def detect(self, image: Image.Image, window_id: int = 1) -> list[Detection]:
        # Return a sample dialogue block detection
        return [
            Detection(
                bbox=BBox(x1=20, y1=20, x2=120, y2=60),
                confidence=0.95,
                type=RegionType.DIALOGUE,
                source_window_id=window_id,
            )
        ]


class DummyOCR(OCRProvider):
    """Dummy OCR provider for testing."""

    def __init__(self, text: str = "HELLO WORLD") -> None:
        self._text = text
        self.loaded = False
        self.unloaded = False

    @property
    def name(self) -> str:
        return "DummyOCR"

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def language(self) -> str:
        return "en"

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.unloaded = True

    def recognize(self, crop: Image.Image, region_bbox: BBox | None = None) -> OCRResult:
        return OCRResult(self._text, 0.98, raw_text=self._text)


class DummyTranslator(TranslationProvider):
    """Dummy translator provider for testing."""

    def __init__(self) -> None:
        self.loaded = False
        self.unloaded = False

    @property
    def name(self) -> str:
        return "DummyTranslator"

    @property
    def version(self) -> str:
        return "1.0"

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.unloaded = True

    def translate(self, input_data: TranslationInput) -> TranslationOutput:
        results = [
            TranslationOutputItem(
                region_id=item.region_id,
                source=item.source,
                translation=f"MERHABA DÜNYA ({item.region_id})",
                raw_model_response=f"MERHABA DÜNYA ({item.region_id})",
            )
            for item in input_data.items
        ]
        return TranslationOutput(
            inputs=input_data,
            results=results,
            raw_response="DUMMY",
            repair_model="dummy",
        )


@pytest.fixture
def synthetic_chapter_dir(tmp_path: Path) -> Path:
    """Creates a temporary synthetic chapter with 2 page images."""
    chap_dir = tmp_path / "synthetic_chapter"
    chap_dir.mkdir(parents=True, exist_ok=True)

    img1 = Image.new("RGB", (200, 400), (255, 255, 255))
    img1.save(chap_dir / "001.png")

    img2 = Image.new("RGB", (200, 400), (255, 255, 255))
    img2.save(chap_dir / "002.png")

    return chap_dir


def test_source_overwrite_protection(synthetic_chapter_dir: Path) -> None:
    """Test 1: Output directory equal to source directory raises ValueError."""
    analyzer = ChapterAnalyzer()
    detector = DummyDetector()

    with pytest.raises(ValueError, match="SOURCE OVERWRITE GUARD"):
        analyzer.process_chapter(
            chapter_path=synthetic_chapter_dir,
            output_path=synthetic_chapter_dir,
            detector=detector,
        )


def test_inpainter_and_renderer_skipped_region() -> None:
    """Test 2: Skipped regions are never inpainted or rendered."""
    canvas = Image.new("RGB", (100, 100), (255, 255, 255))
    reg_skip = Region(
        id=1,
        global_bbox=BBox(x1=10, y1=10, x2=50, y2=50),
        type=RegionType.SFX,
        detection_confidence=0.9,
        source_window_ids=(1,),
        status=RegionStatus.SKIP,
    )

    inpainter = Inpainter()
    clean_canvas = inpainter.inpaint_regions(canvas, [reg_skip])

    renderer = TextRenderer()
    rendered_canvas = renderer.render_regions(clean_canvas, [(reg_skip, "ATLAMA")])

    # Canvas should remain unchanged
    assert canvas.tobytes() == rendered_canvas.tobytes()


def test_export_chapter_pages_count_and_safety(synthetic_chapter_dir: Path, tmp_path: Path) -> None:
    """Test 7 & 8: Export page count and page dimensions preserved."""
    out_dir = tmp_path / "output_test"
    p1 = Page(index=0, path=synthetic_chapter_dir / "001.png", width=200, height=400, y_offset=0)
    p2 = Page(index=1, path=synthetic_chapter_dir / "002.png", width=200, height=400, y_offset=400)
    canvas = Image.new("RGB", (200, 800), (255, 255, 255))

    exported = export_chapter_pages([p1, p2], canvas, out_dir)

    assert len(exported) == 2
    assert exported[0].name == "001.png"
    assert exported[1].name == "002.png"

    with Image.open(exported[0]) as im:
        assert im.size == (200, 400)


def test_end_to_end_synthetic_chapter_smoke_test(synthetic_chapter_dir: Path, tmp_path: Path) -> None:
    """Test 10: Full synthetic chapter end-to-end process_chapter smoke test."""
    out_dir = tmp_path / "output_e2e"
    analyzer = ChapterAnalyzer()

    detector = DummyDetector()
    ocr = DummyOCR("HELLO WORLD")
    translator = DummyTranslator()

    res: ProductionPipelineResult = analyzer.process_chapter(
        chapter_path=synthetic_chapter_dir,
        output_path=out_dir,
        detector=detector,
        primary_ocr=ocr,
        translator=translator,
    )

    assert res.page_count == 2
    assert res.detected_region_count > 0
    assert res.translated_region_count > 0
    assert len(res.exported_page_paths) == 2
    assert detector.unloaded is True
    assert ocr.unloaded is True
    assert translator.unloaded is True

    # Output images exist and are readable
    for page_path in res.exported_page_paths:
        assert page_path.exists()
        with Image.open(page_path) as im:
            assert im.size == (200, 400)
