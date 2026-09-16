"""Unit and end-to-end synthetic tests for the production pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from PIL import Image

from application.chapter_analyzer import ChapterAnalyzer, ProductionPipelineResult
from application.pipeline_common import select_rescued_blocks
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


def test_output_inside_source_is_rejected(synthetic_chapter_dir: Path) -> None:
    """Output directory directly inside source must be rejected."""
    analyzer = ChapterAnalyzer()
    detector = DummyDetector()

    with pytest.raises(ValueError, match="SOURCE OVERWRITE GUARD"):
        analyzer.process_chapter(
            chapter_path=synthetic_chapter_dir,
            output_path=synthetic_chapter_dir / "output",
            detector=detector,
        )


def test_output_deeply_nested_inside_source_is_rejected(synthetic_chapter_dir: Path) -> None:
    """Output directory deeply nested inside source must be rejected."""
    analyzer = ChapterAnalyzer()
    detector = DummyDetector()

    with pytest.raises(ValueError, match="SOURCE OVERWRITE GUARD"):
        analyzer.process_chapter(
            chapter_path=synthetic_chapter_dir,
            output_path=synthetic_chapter_dir / "a" / "b" / "output",
            detector=detector,
        )


def test_source_inside_output_is_rejected(tmp_path: Path) -> None:
    """Source directory inside output must be rejected."""
    analyzer = ChapterAnalyzer()
    detector = DummyDetector()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    chapter_dir = output_dir / "chapter1"
    chapter_dir.mkdir()

    with pytest.raises(ValueError, match="SOURCE OVERWRITE GUARD"):
        analyzer.process_chapter(
            chapter_path=chapter_dir,
            output_path=output_dir,
            detector=detector,
        )


def test_sibling_output_path_is_allowed(synthetic_chapter_dir: Path, tmp_path: Path) -> None:
    """Sibling source/output directories must be allowed."""
    analyzer = ChapterAnalyzer()
    detector = DummyDetector()
    ocr = DummyOCR("HELLO")
    translator = DummyTranslator()
    output_dir = tmp_path / "translated"

    result = analyzer.process_chapter(
        chapter_path=synthetic_chapter_dir,
        output_path=output_dir,
        detector=detector,
        primary_ocr=ocr,
        translator=translator,
    )
    assert result.page_count == 2
    assert len(result.exported_page_paths) == 2


def test_similarly_named_sibling_paths_are_allowed(tmp_path: Path) -> None:
    """Similarly named sibling directories like chapter1/chapter10 must be allowed."""
    chapter1 = tmp_path / "chapter1"
    chapter1.mkdir()
    chapter10 = tmp_path / "chapter10"
    chapter10.mkdir()

    img = Image.new("RGB", (200, 400), (255, 255, 255))
    img.save(chapter1 / "001.png")
    img.save(chapter10 / "001.png")

    analyzer = ChapterAnalyzer()
    detector = DummyDetector()
    ocr = DummyOCR("HELLO")
    translator = DummyTranslator()
    output_dir = tmp_path / "output"

    result = analyzer.process_chapter(
        chapter_path=chapter1,
        output_path=output_dir,
        detector=detector,
        primary_ocr=ocr,
        translator=translator,
    )
    assert result.page_count == 1
    assert len(result.exported_page_paths) == 1


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


class TwoBoxDetector(DummyDetector):
    """Uzak iki DIALOGUE kutusu üretir (iki ayrı blok garanti)."""

    @property
    def name(self) -> str:
        return "TwoBoxDetector"  # tespit önbelleğinden ayrı düşer

    def detect(self, image: Image.Image, window_id: int = 1) -> list[Detection]:
        return [
            Detection(
                bbox=BBox(x1=20, y1=20, x2=120, y2=60),
                confidence=0.95,
                type=RegionType.DIALOGUE,
                source_window_id=window_id,
            ),
            Detection(
                bbox=BBox(x1=20, y1=300, x2=120, y2=340),
                confidence=0.95,
                type=RegionType.DIALOGUE,
                source_window_id=window_id,
            ),
        ]


def test_region_translations_match_own_block(synthetic_chapter_dir: Path, tmp_path: Path) -> None:
    """STUDIO regresyonu bekçisi: her bölge KENDİ bloğunun çevirisini taşır.

    DummyTranslator çeviriye blok id'sini gömer (`MERHABA DÜNYA (<id>)`);
    sızmış döngü değişkeni (`out_map[b.id]` yerine `b_id`) tüm bölgelere
    son bloğun çevirisini yazardı — bu test o sınıfı yakalar.
    """
    out_dir = tmp_path / "output_blockmatch"
    analyzer = ChapterAnalyzer()
    analyzer.process_chapter(
        chapter_path=synthetic_chapter_dir,
        output_path=out_dir,
        detector=TwoBoxDetector(),
        primary_ocr=DummyOCR("HELLO WORLD"),
        translator=DummyTranslator(),
    )
    payload = json.loads((out_dir / "analysis" / "regions.json").read_text(encoding="utf-8"))
    regions = {r["id"]: r for r in payload["regions"]}
    checked = 0
    for block in payload["text_blocks"]:
        expected = f"({block['id']})"
        for member_id in block["member_ids"]:
            region = regions[member_id]
            if region.get("type") in ("sfx", "watermark"):
                continue
            translation = region.get("translation") or ""
            if not translation:
                continue
            checked += 1
            assert expected in translation, (
                f"b{block['id']} r{member_id}: {translation!r} blok id'sini taşımıyor"
            )
    assert checked >= 1, "hiç çevrilmiş bölge bulunamadı"


class GuardFlagTranslator(DummyTranslator):
    """Translator stub emitting a fatal numbering_inconsistent guard flag."""

    warnings = ["numbering_inconsistent"]

    def translate(self, input_data: TranslationInput) -> TranslationOutput:
        results = [
            TranslationOutputItem(
                region_id=item.region_id,
                source=item.source,
                translation=f"MERHABA ({item.region_id})",
                raw_model_response=f"MERHABA ({item.region_id})",
                validation_warnings=list(self.warnings),
                requires_review=True,
            )
            for item in input_data.items
        ]
        return TranslationOutput(
            inputs=input_data,
            results=results,
            raw_response="DUMMY-GUARD",
            repair_model="dummy-guard",
        )


class DroppedFlagTranslator(GuardFlagTranslator):
    """F2: ad-düşürme bayrağı da ölümcül guard sayılır."""

    warnings = ["dropped_number_token"]


class NameGlueFlagTranslator(GuardFlagTranslator):
    """Madde 2: ad-yapışma bayrağı da ölümcül guard sayılır."""

    warnings = ["name_glue"]

def test_translation_guard_blocks_never_render(synthetic_chapter_dir: Path, tmp_path: Path) -> None:
    """P1-B: numbering_inconsistent blok failed sayılır, REVIEW olur, basılmaz."""
    out_dir = tmp_path / "output_guard"
    analyzer = ChapterAnalyzer()

    res: ProductionPipelineResult = analyzer.process_chapter(
        chapter_path=synthetic_chapter_dir,
        output_path=out_dir,
        detector=DummyDetector(),
        primary_ocr=DummyOCR("HELLO WORLD"),
        translator=GuardFlagTranslator(),
    )
    assert res.page_count == 2

    summary = json.loads((out_dir / "analysis" / "summary.json").read_text(encoding="utf-8"))
    assert summary["translated_blocks_count"] == 0
    assert summary["translation_guard_blocks_count"] >= 1
    assert summary["rendered_blocks_count"] == 0

    regions = json.loads((out_dir / "analysis" / "regions.json").read_text(encoding="utf-8"))["regions"]
    guard_regions = [r for r in regions if r.get("review_reason") == "translation_guard_review"]
    assert guard_regions, "guard blok bölgesi REVIEW işaretlenmeli"
    assert all(r.get("status") == "review" for r in guard_regions)


def test_dropped_token_guard_blocks_never_render(synthetic_chapter_dir: Path, tmp_path: Path) -> None:
    """F2: dropped_number_token bayraklı blok failed sayılır, REVIEW olur, basılmaz."""
    out_dir = tmp_path / "output_dropped"
    analyzer = ChapterAnalyzer()

    res: ProductionPipelineResult = analyzer.process_chapter(
        chapter_path=synthetic_chapter_dir,
        output_path=out_dir,
        detector=DummyDetector(),
        primary_ocr=DummyOCR("HELLO WORLD"),
        translator=DroppedFlagTranslator(),
    )
    assert res.page_count == 2

    summary = json.loads((out_dir / "analysis" / "summary.json").read_text(encoding="utf-8"))
    assert summary["translated_blocks_count"] == 0
    assert summary["translation_guard_blocks_count"] >= 1
    assert summary["rendered_blocks_count"] == 0

    regions = json.loads((out_dir / "analysis" / "regions.json").read_text(encoding="utf-8"))["regions"]
    guard_regions = [r for r in regions if r.get("review_reason") == "translation_guard_review"]
    assert guard_regions, "dropped blok bölgesi REVIEW işaretlenmeli"
    assert all(r.get("status") == "review" for r in guard_regions)


def test_name_glue_guard_blocks_never_render(synthetic_chapter_dir: Path, tmp_path: Path) -> None:
    """Madde 2: name_glue bayraklı blok failed sayılır, REVIEW olur, basılmaz."""
    out_dir = tmp_path / "output_nameglue"
    analyzer = ChapterAnalyzer()

    res: ProductionPipelineResult = analyzer.process_chapter(
        chapter_path=synthetic_chapter_dir,
        output_path=out_dir,
        detector=DummyDetector(),
        primary_ocr=DummyOCR("HELLO WORLD"),
        translator=NameGlueFlagTranslator(),
    )
    assert res.page_count == 2

    summary = json.loads((out_dir / "analysis" / "summary.json").read_text(encoding="utf-8"))
    assert summary["translated_blocks_count"] == 0
    assert summary["translation_guard_blocks_count"] >= 1
    assert summary["rendered_blocks_count"] == 0

    regions = json.loads((out_dir / "analysis" / "regions.json").read_text(encoding="utf-8"))["regions"]
    guard_regions = [r for r in regions if r.get("review_reason") == "translation_guard_review"]
    assert guard_regions, "name_glue blok bölgesi REVIEW işaretlenmeli"
    assert all(r.get("status") == "review" for r in guard_regions)


def test_guard_review_region_records_warning_codes(synthetic_chapter_dir: Path, tmp_path: Path) -> None:
    """Teşhis kaydı: guard-REVIEW bölge hangi uyarının tuttuğunu metadata'da taşır."""
    out_dir = tmp_path / "output_guardwarn"
    analyzer = ChapterAnalyzer()

    analyzer.process_chapter(
        chapter_path=synthetic_chapter_dir,
        output_path=out_dir,
        detector=DummyDetector(),
        primary_ocr=DummyOCR("HELLO WORLD"),
        translator=DroppedFlagTranslator(),
    )

    regions = json.loads((out_dir / "analysis" / "regions.json").read_text(encoding="utf-8"))["regions"]
    guard_regions = [r for r in regions if r.get("review_reason") == "translation_guard_review"]
    assert guard_regions, "guard blok bölgesi REVIEW işaretlenmeli"
    assert all(
        (r.get("metadata") or {}).get("translation_guard_warnings") == ["dropped_number_token"]
        for r in guard_regions
    ), "uyarı kodu metadata'da görünmeli"


def test_select_rescued_blocks_needs_full_coverage_and_isolation() -> None:
    """Kurtarma: tam kapsama + komşuyla çakışmasızlık birlikte gerekir."""
    boxes = {7: [[20, 20, 40, 30]], 8: [[20, 20, 40, 30]], 9: [[20, 20, 40, 30]]}
    plan = {7: [[10, 10, 100, 50]], 8: [[10, 10, 30, 50]], 9: [[10, 10, 100, 50]]}
    block_boxes = {
        7: BBox(x1=0, y1=0, x2=200, y2=100),
        8: BBox(x1=0, y1=0, x2=200, y2=100),
        9: BBox(x1=0, y1=0, x2=200, y2=100),
    }
    far = [BBox(x1=500, y1=500, x2=600, y2=600)]
    near = [BBox(x1=10, y1=10, x2=190, y2=90)]
    # 7: kapsanır + ıssız → kurtarılır; 8: taşar → yok; 9: komşuyla çakışır → yok.
    assert select_rescued_blocks({7, 8}, boxes, plan, block_boxes, far) == {7}
    assert select_rescued_blocks({9}, boxes, plan, block_boxes, near) == set()
    assert select_rescued_blocks({7}, {}, plan, block_boxes, far) == set()
    assert select_rescued_blocks({7}, boxes, {}, block_boxes, far) == set()


class EchoTranslator(DummyTranslator):
    """S2: çeviriyi aynen iade eder (tek-kelimelik yankı)."""

    def translate(self, input_data: TranslationInput) -> TranslationOutput:
        results = [
            TranslationOutputItem(
                region_id=item.region_id,
                source=item.source,
                translation=item.source,
                raw_model_response=item.source,
            )
            for item in input_data.items
        ]
        return TranslationOutput(
            inputs=input_data,
            results=results,
            raw_response="DUMMY-ECHO",
            repair_model="dummy-echo",
        )


def test_single_word_echo_blocks_keep_english_pixels(synthetic_chapter_dir: Path, tmp_path: Path) -> None:
    """S2: tek-kelimelik yankı bloğu inpaint/render görmez, SKIP olur."""
    out_dir = tmp_path / "output_echo"
    analyzer = ChapterAnalyzer()

    res: ProductionPipelineResult = analyzer.process_chapter(
        chapter_path=synthetic_chapter_dir,
        output_path=out_dir,
        detector=DummyDetector(),
        primary_ocr=DummyOCR("GOT"),
        translator=EchoTranslator(),
    )
    assert res.page_count == 2

    summary = json.loads((out_dir / "analysis" / "summary.json").read_text(encoding="utf-8"))
    assert summary["translated_blocks_count"] == 0
    assert summary["rendered_blocks_count"] == 0
    assert summary["echo_preserved_blocks_count"] >= 1

    regions = json.loads((out_dir / "analysis" / "regions.json").read_text(encoding="utf-8"))["regions"]
    echo_regions = [r for r in regions if r.get("review_reason") == "echo_single_preserved"]
    assert echo_regions, "yankı blok bölgesi SKIP işaretlenmeli"
    assert all(r.get("status") == "skip" for r in echo_regions)
