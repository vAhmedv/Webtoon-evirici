"""Pipeline ortak yardımcıları (chapter_analyzer'ın ince dilimi).

`ChapterAnalyzer` büyüdükçe (1400+ satır) saf/devletsiz yardımcılar buraya
taşınır; orkestrasyon `application/chapter_analyzer.py` içinde kalır.
Bu modül pipeline state'ine dokunmaz — yalnızca veri dönüştürür.

Geriye-uyumluluk: `chapter_analyzer` bu isimleri yeniden export eder, mevcut
import'lar (`from application.chapter_analyzer import _image_to_bytes` vb.)
kırılmaz.
"""

from __future__ import annotations

import io
import os
from dataclasses import replace
from pathlib import Path

from core.config import Config
from core.detection import Detection, Region, RegionStatus
from core.detection.coordinate import global_bbox_to_window
from providers.detector.base import DetectorProvider

# ÖLÜMCÜL çeviri guard'ları: bu bayraklardan birini taşıyan blok sessizce
# basılmaz — failed sayılır, bölgesi REVIEW olur (İngilizce korunur).
# numbering_inconsistent (P1-B kanıtlı satır-kayma) + dropped_* (F2 ad-düşürme)
# + name_glue (ad-yapışma) + kinship_ambiguous (akıcı-ama-yanlış akrabalık).
_FATAL_TRANSLATION_WARNINGS = frozenset({
    "numbering_inconsistent",
    "dropped_number_token",
    "dropped_content_token",
    "name_glue",
    "kinship_ambiguous",
})


def _norm_echo_text(text: str | None) -> str:
    return " ".join((text or "").split()).casefold()


def _is_single_word_echo(source_text: str | None, translated_text: str | None) -> bool:
    """S2: tek-kelimelik yankı (TR==kaynak) basılmaz — İngilizce korunur.

    Çevrilen tek kelimeler (DAMMIT→Kahretsin!) etkilenmez; çok-kelimeli
    yankılar (ZFO GOBLINS?!) mevcut davranışta kalır (S0 Katman-1 serbest).
    """
    src = _norm_echo_text(source_text)
    return bool(src) and src == _norm_echo_text(translated_text) and len(src.split()) == 1


def _replace(config: Config, **kwargs) -> Config:
    """Config ile yeni bir Config oluşturur (override edilebilir alanlar için)."""
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
    return replace(
        region,
        status=new_status,
    )


def _replace_region(region: Region, **kwargs) -> Region:
    """Region alanlarını değiştirir (yeni Region döndürür)."""
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
    """Detector'dan model_id ve model_mtime çıkarır."""
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
    cache_schema = getattr(detector, "cache_schema_version", None)
    if cache_schema:
        model_id = f"{model_id}|{cache_schema}"
    return model_id, model_mtime


def _image_to_bytes(image) -> bytes:
    """Fast deterministic byte extraction without PNG encoding overhead."""
    if isinstance(image, bytes):
        return image
    if hasattr(image, "tobytes"):
        return image.tobytes()
    if isinstance(image, (bytearray, memoryview)):
        return bytes(image)
    buf = io.BytesIO()
    if hasattr(image, "save"):
        image.save(buf, format="PNG")
    else:
        buf.write(bytes(image))
    return buf.getvalue()


def _global_detection_to_window(det: Detection, window_y_start: int) -> Detection:
    """Global Detection'ı window-local koordinata çevirir (visualization için)."""
    local_bbox = global_bbox_to_window(det.bbox, window_y_start)
    metadata = _offset_geometry_metadata(det.metadata, -window_y_start)
    return Detection(
        bbox=local_bbox,
        confidence=det.confidence,
        type=det.type,
        source_window_id=det.source_window_id,
        mask=det.mask,
        metadata=metadata,
    )


def _offset_geometry_metadata(metadata: object, y_offset: int) -> dict:
    """Translate every compact CTD geometry field without expanding it to a pixel mask."""
    result = dict(metadata) if isinstance(metadata, dict) else {}
    for key in ("polygon",):
        polygon = result.get(key)
        if isinstance(polygon, list) and polygon:
            result[key] = [[float(p[0]), float(p[1]) + y_offset] for p in polygon]
    for key in ("line_polygons", "segmentation_polygons"):
        polygons = result.get(key)
        if isinstance(polygons, list):
            result[key] = [
                [[float(p[0]), float(p[1]) + y_offset] for p in polygon]
                for polygon in polygons
                if isinstance(polygon, list) and len(polygon) >= 3
            ]
    block_bbox = result.get("ctd_block_bbox")
    if isinstance(block_bbox, list) and len(block_bbox) == 4:
        result["ctd_block_bbox"] = [
            float(block_bbox[0]), float(block_bbox[1]) + y_offset,
            float(block_bbox[2]), float(block_bbox[3]) + y_offset,
        ]
    return result
