"""Bölüm analiz hizmeti.

Core pipeline'ı UI'dan bağımsız olarak orchestrate eder.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Callable, Sequence

from loguru import logger

from application.cancellation import CancellationToken, CancelledError
from application.progress import ProgressEvent
from core.config import Config, load_config
from core.coordinate.global_coords import GlobalCoordinateSystem
from core.coordinate.sliding_window import generate_windows_for_pages
from core.detection import Detection, DetectionCache, Region, RegionStatus, RegionType
from core.detection.cache import CACHE_PATH
from core.detection.coordinate import (
    window_bbox_to_global,
    global_bbox_to_window,
    window_polygon_to_global,
    global_polygon_to_window,
)
from core.detection.merge import merge_duplicates
from core.imaging.window_extractor import extract_window_image, WindowImage
from core.imaging.region_cropper import RegionCropper
from core.io.input_loader import load_chapter
from core.models import Page, Window
from core.serialization.serializer import region_to_dict
from core.visualization.draw import draw_detections, draw_regions
from providers.detector.base import DetectorProvider
from providers.ocr.base import OCRProvider


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
        ocr_elapsed_time: OCR süresi (saniye).
    """

    def __init__(
        self,
        pages: list[Page],
        windows: list[Window],
        regions: list[Region],
        elapsed_time: float,
        visualization_paths: list[Path] | None = None,
        warnings: list[str] | None = None,
        ocr_elapsed_time: float = 0.0,
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
        self.ocr_elapsed_time = ocr_elapsed_time


class ChapterAnalyzer:
    """Bölüm analiz hizmeti.

    Pipeline'ı UI'dan bağımsız olarak çalıştırır.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config if config is not None else load_config()
        self._cache = DetectionCache(
            cache_path=CACHE_PATH,
            max_entries=self.config.detection.max_cache_entries,
            enabled=self.config.detection.enabled,
        )

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
        ocr_provider: OCRProvider | None = None,
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
            ocr_provider: Opsiyonel OCR sağlayıcı.

        Returns:
            AnalysisResult.
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
        if hasattr(detector, "confidence_threshold"):
            detector.confidence_threshold = conf
        detector.load()

        # Cache: get model identity for cache key
        model_id, model_mtime = _get_model_identity(detector)
        self._cache.load()

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

            # Cache lookup / store
            image_bytes = _image_to_bytes(window_image.image)
            page_hash = DetectionCache.compute_hash(image_bytes)
            cached = self._cache.get(page_hash, model_id, model_mtime)

            if cached is not None:
                # Cache HIT: use cached global detections directly
                global_detections = cached
                # Convert back to window-local for visualization
                detections = [_global_detection_to_window(d, window.y_start) for d in cached]
                logger.debug(f"Window {window.id}: cache HIT")
            else:
                # Cache MISS: run YOLO, convert to global, cache
                detections = detector.detect(window_image.image, window.id)

                # Provider local WindowImage koordinatları üretir.
                # Merge duplicate öncesinde global chapter koordinatına çevir.
                global_detections: list[Detection] = []
                for det in detections:
                    global_bbox = window_bbox_to_global(det.bbox, window.y_start)
                    metadata = dict(det.metadata)
                    polygon = metadata.get("polygon")
                    if isinstance(polygon, list) and len(polygon) > 0:
                        metadata["polygon"] = window_polygon_to_global(polygon, window.y_start)
                    global_det = Detection(
                        bbox=global_bbox,
                        confidence=det.confidence,
                        type=det.type,
                        source_window_id=det.source_window_id,
                        mask=det.mask,
                        metadata=metadata,
                    )
                    global_detections.append(global_det)

                # Store in cache (store global detections)
                self._cache.put(page_hash, model_id, model_mtime, global_detections)

            all_detections.extend(global_detections)

            vis = draw_detections(window_image.image, detections, window_y_start=window.y_start)
            vis_path = visualization_dir / f"window_{window.id:03d}.png"
            vis.save(vis_path)
            window_visualization_paths.append(vis_path)

        # Save cache after all detection
        self._cache.save()
        logger.info(f"Detection cache saved: {len(self._cache._entries)} entries")

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

        # 7.5 OCR (opsiyonel)
        ocr_start = 0.0
        ocr_elapsed = 0.0
        if ocr_provider is not None:
            _progress("Loading OCR")
            try:
                ocr_provider.load()
            except Exception as e:
                logger.error(f"OCR provider yüklenemedi: {e}")
                warnings.append(f"OCR load failed: {e}")
                ocr_provider = None

            if ocr_provider is not None:
                cropper = RegionCropper(pages, coords, padding=20)
                ocr_start = time.time()
                ocr_regions: list[Region] = []
                for idx, region in enumerate(regions, start=1):
                    if cancellation_token and cancellation_token.is_cancelled:
                        raise CancelledError()
                    _progress("OCR", current=idx, total=len(regions), message=f"OCR {idx}/{len(regions)}")
                    try:
                        crop = cropper.crop_region(region)
                        result = ocr_provider.recognize(crop.image, region_bbox=region.global_bbox)
                        if result.text:
                            ocr_regions.append(
                                _replace_region(
                                    region,
                                    text=result.text,
                                    ocr_confidence=result.confidence,
                                    metadata={**region.metadata, "ocr_warnings": result.warnings},
                                )
                            )
                        else:
                            warnings.extend(result.warnings)
                            ocr_regions.append(region)
                    except Exception as e:
                        logger.error(f"OCR failed for region {region.id}: {e}")
                        warnings.append(f"OCR region {region.id}: {e}")
                        ocr_regions.append(region)
                regions = ocr_regions
                ocr_elapsed = time.time() - ocr_start
                try:
                    ocr_provider.unload()
                except Exception:
                    pass

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
            ocr_elapsed_time=ocr_elapsed,
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
    from dataclasses import replace

    return replace(
        region,
        status=new_status,
    )


def _replace_region(region: Region, **kwargs) -> Region:
    """Region alanlarını değiştirir (yeni Region döndürür)."""
    from dataclasses import replace

    allowed = {
        "id",
        "global_bbox",
        "type",
        "detection_confidence",
        "source_window_ids",
        "status",
        "text",
        "ocr_confidence",
        "translation",
        "review_reason",
        "metadata",
    }
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return replace(region, **filtered)


def _get_model_identity(detector: DetectorProvider) -> tuple[str, str | float]:
    """Detector'dan model_id ve model_mtime çıkarır.

    Args:
        detector: DetectorProvider instance.

    Returns:
        (model_id, model_mtime) tuple.
        model_id: model dosya yolu stringi veya provider name.
        model_mtime: model dosyasının mtime (float) veya "unknown" string.
    """
    model_path = getattr(detector, "_model_path", None)
    if model_path is not None and Path(model_path).exists():
        model_id = str(Path(model_path).resolve())
        try:
            model_mtime = os.path.getmtime(str(model_path))
        except OSError:
            model_mtime = "unknown"
    else:
        model_id = getattr(detector, "name", "unknown")
        model_mtime = "unknown"
    return model_id, model_mtime


def _image_to_bytes(image) -> bytes:
    """PIL Image'ı deterministic byte string'e çevirir (PNG encode)."""
    if hasattr(image, "tobytes"):
        return image.tobytes()
    buf = io.BytesIO()
    if hasattr(image, "save"):
        image.save(buf, format="PNG")
    elif isinstance(image, bytes):
        return image
    else:
        buf.write(bytes(image))
    return buf.getvalue()


def _global_detection_to_window(det: Detection, window_y_start: int) -> Detection:
    """Global Detection'ı window-local koordinata çevirir (visualization için)."""
    local_bbox = global_bbox_to_window(det.bbox, window_y_start)
    metadata = dict(det.metadata)
    polygon = metadata.get("polygon")
    if isinstance(polygon, list) and len(polygon) > 0:
        metadata["polygon"] = global_polygon_to_window(polygon, window_y_start)
    return Detection(
        bbox=local_bbox,
        confidence=det.confidence,
        type=det.type,
        source_window_id=det.source_window_id,
        mask=det.mask,
        metadata=metadata,
    )
