"""Bölüm analiz hizmeti.

Core pipeline'ı UI'dan bağımsız olarak orchestrate eder.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Sequence

from loguru import logger

from application.cancellation import CancellationToken, CancelledError
from application.progress import ProgressEvent
from core.config import Config, load_config
from core.coordinate.global_coords import GlobalCoordinateSystem
from core.coordinate.sliding_window import generate_windows_for_pages
from core.detection import Region, RegionStatus, RegionType
from core.detection.merge import merge_duplicates
from core.imaging.window_extractor import extract_window_image, WindowImage
from core.io.input_loader import load_chapter
from core.models import Page, Window
from core.serialization.serializer import region_to_dict
from core.visualization.draw import draw_detections, draw_regions
from providers.detector.base import DetectorProvider


ProgressCallback = Callable[[ProgressEvent], None]


class AnalysisResult:
    """Bölüm analizi sonucu.

    Attributes:
        pages: Yüklenen sayfalar.
        windows: Üretilen window'lar.
        regions: Canonical region listesi.
        auto_count: AUTO durumundaki region sayısı.
        review_count: REVIEW durumundaki region sayısı.
        skip_count: SKIP durumundaki region sayısı.
        elapsed_time: Analiz süresi (saniye).
        visualization_paths: Her window için görselleştirme yolları.
        warnings: Oluşan uyarılar.
    """

    def __init__(
        self,
        pages: list[Page],
        windows: list[Window],
        regions: list[Region],
        elapsed_time: float,
        visualization_paths: list[Path] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.pages = pages
        self.windows = windows
        self.regions = regions
        self.auto_count = sum(1 for r in regions if r.status == RegionStatus.AUTO)
        self.review_count = sum(1 for r in regions if r.status == RegionStatus.REVIEW)
        self.skip_count = sum(1 for r in regions if r.status == RegionStatus.SKIP)
        self.elapsed_time = elapsed_time
        self.visualization_paths = visualization_paths or []
        self.warnings = warnings or []


class ChapterAnalyzer:
    """Bölüm analiz hizmeti.

    Pipeline'ı UI'dan bağımsız olarak çalıştırır.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config if config is not None else load_config()

    def analyze(
        self,
        chapter_path: str | Path,
        output_path: str | Path,
        detector: DetectorProvider,
        window_height: int | None = None,
        window_overlap: int | None = None,
        min_confidence: float | None = None,
        progress_callback: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> AnalysisResult:
        """Bölüm analizini çalıştırır.

        Args:
            chapter_path: Bölüm klasörü yolu.
            output_path: Çıktı klasörü yolu.
            detector: Kullanılacak detector sağlayıcı.
            window_height: Pencere yüksekliği (config.yaml override).
            window_overlap: Pencere örtüşmesi (config.yaml override).
            min_confidence: Minimum güven eşiği (config.yaml override).
            progress_callback: İlerleme bildirimi callback'i.
            cancellation_token: İptal belirteci.

        Returns:
            AnalysisResult.

        Raises:
            FileNotFoundError: chapter_path yoksa.
            ValueError: chapter_path boşsa veya görüntü yoksa.
            RuntimeError: detector hatası olursa.
            CancelledError: Kullanıcı iptal ederse.
        """
        start_time = time.time()
        chapter_path = Path(chapter_path)
        output_path = Path(output_path)
        warnings: list[str] = []

        cfg = self.config
        if window_height is not None:
            cfg = _replace(cfg, window_height=window_height)
        if window_overlap is not None:
            cfg = _replace(cfg, window_overlap=window_overlap)
        conf = min_confidence if min_confidence is not None else cfg.min_confidence

        def _progress(stage: str, current: int = 0, total: int = 0, message: str = "") -> None:
            if progress_callback is None:
                return
            pct = 0.0
            if total > 0:
                pct = max(pct, current / total)
            progress_callback(ProgressEvent(stage=stage, current=current, total=total, message=message, percent=pct))

        # 1. Load chapter
        _progress("Loading chapter", message=str(chapter_path))
        pages = load_chapter(chapter_path, cfg)
        _progress("Loading chapter", current=1, total=1, message=f"{len(pages)} pages loaded")

        if cancellation_token and cancellation_token.is_cancelled:
            raise CancelledError()

        # 2. Global coordinate system
        _progress("Creating coordinate system")
        coords = GlobalCoordinateSystem(tuple(pages))

        # 3. Generate windows
        _progress("Creating windows")
        windows = generate_windows_for_pages(
            pages,
            window_height=cfg.window_height,
            overlap=cfg.window_overlap,
        )
        _progress("Creating windows", current=len(windows), total=len(windows))

        if cancellation_token and cancellation_token.is_cancelled:
            raise CancelledError()

        # 4. Load detector
        _progress("Loading detector")
        detector.load()

        # 5. Detect
        all_detections: list = []
        visualization_dir = output_path / "analysis" / "windows"
        visualization_dir.mkdir(parents=True, exist_ok=True)
        window_visualization_paths: list[Path] = []

        for idx, window in enumerate(windows, start=1):
            if cancellation_token and cancellation_token.is_cancelled:
                raise CancelledError()

            _progress("Detecting", current=idx, total=len(windows), message=f"Window {idx}/{len(windows)}")

            window_image = extract_window_image(tuple(pages), window, coords)
            detections = detector.detect(window_image.image, window.id)
            all_detections.extend(detections)

            vis = draw_detections(window_image.image, detections, window_y_start=window.y_start)
            vis_path = visualization_dir / f"window_{window.id:03d}.png"
            vis.save(vis_path)
            window_visualization_paths.append(vis_path)

        detector.unload()

        if cancellation_token and cancellation_token.is_cancelled:
            raise CancelledError()

        # 6. Merge duplicates
        _progress("Merging regions")
        regions = merge_duplicates(all_detections, min_confidence=conf)

        # 7. Apply safety status (already done in merge, but ensure)
        regions = [
            _replace_status(reg, RegionStatus.REVIEW) if reg.status == RegionStatus.AUTO and reg.detection_confidence < conf else reg
            for reg in regions
        ]

        if cancellation_token and cancellation_token.is_cancelled:
            raise CancelledError()

        # 8. Save outputs
        _progress("Saving results")
        analysis_dir = output_path / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        regions_json = analysis_dir / "regions.json"
        with open(regions_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pages": len(pages),
                    "windows": len(windows),
                    "regions": [region_to_dict(r) for r in regions],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        summary = {
            "pages": len(pages),
            "windows": len(windows),
            "regions": len(regions),
            "auto": sum(1 for r in regions if r.status == RegionStatus.AUTO),
            "review": sum(1 for r in regions if r.status == RegionStatus.REVIEW),
            "skip": sum(1 for r in regions if r.status == RegionStatus.SKIP),
            "elapsed_time": 0.0,
            "warnings": warnings,
        }
        summary_path = analysis_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start_time
        summary["elapsed_time"] = elapsed
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 9. Global visualization
        _progress("Rendering preview")
        preview_path = self._render_global_preview(pages, regions, output_path)

        _progress("Completed", current=1, total=1, message="Analysis complete")
        logger.info(
            f"Analysis complete: {len(pages)} pages, {len(windows)} windows, "
            f"{len(regions)} regions, AUTO={summary['auto']}, "
            f"REVIEW={summary['review']}, SKIP={summary['skip']}, "
            f"time={elapsed:.2f}s"
        )

        return AnalysisResult(
            pages=pages,
            windows=windows,
            regions=regions,
            elapsed_time=elapsed,
            visualization_paths=window_visualization_paths + [preview_path],
            warnings=warnings,
        )

    def _render_global_preview(
        self,
        pages: list[Page],
        regions: list[Region],
        output_path: Path,
    ) -> Path:
        """Global preview görseli oluşturur."""
        from PIL import Image

        if not pages:
            preview_path = output_path / "analysis" / "preview.png"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (1, 1), (255, 255, 255)).save(preview_path)
            return preview_path

        first = pages[0]
        with Image.open(first.path) as sample:
            width, _ = sample.size

        total_height = sum(p.height for p in pages)

        full = Image.new("RGB", (width, total_height), (255, 255, 255))
        y_cursor = 0
        for page in pages:
            with Image.open(page.path) as img:
                full.paste(img, (0, y_cursor))
                y_cursor += page.height

        full = draw_regions(full, regions, window_y_start=0)
        preview_path = output_path / "analysis" / "preview.png"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        full.save(preview_path, "PNG")
        return preview_path


def _replace(config: Config, **kwargs) -> Config:
    """Config ile yeni bir Config oluşturur (override edilebilir alanlar için)."""
    from dataclasses import replace

    allowed = {
        "window_height",
        "window_overlap",
        "input_extensions",
        "output_format",
        "log_level",
        "log_file",
        "min_confidence",
    }
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return replace(config, **filtered)


def _replace_status(region: Region, new_status: RegionStatus) -> Region:
    """Region durumunu değiştirir (yeni Region döndürür)."""
    return Region(
        id=region.id,
        global_bbox=region.global_bbox,
        type=region.type,
        detection_confidence=region.detection_confidence,
        source_window_ids=region.source_window_ids,
        status=new_status,
        text=region.text,
        ocr_confidence=region.ocr_confidence,
        translation=region.translation,
        review_reason=region.review_reason,
    )
