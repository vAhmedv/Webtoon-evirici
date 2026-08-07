"""RapidOCR ONNX Runtime provider (candidate/default).

``rapidocr_onnxruntime`` paketine (ONNX Runtime backend) dayanır. PaddlePaddle /
torch ortamını bozmaz; model dosyalarını ilk çalıştırmada otomatik indirir.
PP-OCR v3/v4 tabanlı ONNX modeller kullanır; RapidOCR 1.4.4.

Not: Bu provider yalnızca verilmiş Region crop'u üzerinde OCR yapar; chapter
için ayrı bir detectör çalıştırmaz.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from core.detection import BBox
from providers.ocr.base import OCRLine, OCRProvider, OCRResult


class RapidONNXOCR(OCRProvider):
    """RapidOCR ONNX Runtime provider."""

    _NAME = "RapidOCR-ONNX"
    _VERSION = "rapidocr-onnx-1.4.4"

    def __init__(self) -> None:
        self._loaded = False
        self._engine = None
        self._device = "cpu"

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def version(self) -> str:
        return self._VERSION

    @property
    def device(self) -> str:
        return self._device

    @property
    def language(self) -> str:
        # RapidOCR 1.4.4 varsayılan rec modeli çok dilli (EN destekli).
        return "multi"

    @property
    def status(self) -> str:
        return "candidate/default"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise RuntimeError(
                "rapidocr_onnxruntime package gerekli.\n"
                "Kurulum: pip install rapidocr-onnxruntime"
            ) from e
        self._engine = RapidOCR()
        try:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self._device = "cpu"
        self._loaded = True
        logger.info(f"RapidOCR-ONNX model loaded successfully (device={self._device})")

    def unload(self) -> None:
        if not self._loaded:
            return
        self._engine = None
        self._loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("RapidOCR-ONNX model unloaded")

    def recognize(
        self,
        image,
        region_bbox: BBox | None = None,
    ) -> OCRResult:
        if not self._loaded or self._engine is None:
            raise RuntimeError("RapidOCR not loaded; call load() first")

        import numpy as np
        from PIL import Image

        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # RapidOCR returns: (results, elapsed)
        # results: [[[polygon], text, confidence], ...] — reading order'da.
        raw_results, _ = self._engine(img_array)

        lines: list[OCRLine] = []
        all_text_parts: list[str] = []
        total_conf: float = 0.0

        if raw_results is None or len(raw_results) == 0:
            return OCRResult(
                text="",
                confidence=0.0,
                raw_text="",
                lines=[],
                warnings=["empty_ocr_result"],
                metadata={
                    "provider": self.name,
                    "version": self.version,
                    "region_bbox": _bbox_to_dict(region_bbox) if region_bbox else None,
                },
            )

        for item in raw_results:
            polygon_coords, text, conf = item
            polygon = [[float(x), float(y)] for x, y in polygon_coords]
            line_bbox = BBox(
                x1=int(min(p[0] for p in polygon)),
                y1=int(min(p[1] for p in polygon)),
                x2=int(max(p[0] for p in polygon)),
                y2=int(max(p[1] for p in polygon)),
            )
            line = OCRLine(
                text=str(text),
                confidence=float(conf),
                bbox=line_bbox,
                polygon=polygon,
            )
            lines.append(line)
            all_text_parts.append(str(text))
            total_conf += float(conf)

        # Okuma sırası RapidOCR tarafından zaten sağlanır (sorted_boxes).
        # raw_text: ham, satır sonları korunur. text: canonical, boşlukla birleştir.
        raw_text = "\n".join(all_text_parts)
        normalized_text = " ".join(all_text_parts).strip()

        avg_conf = total_conf / len(lines) if lines else 0.0

        warnings: list[str] = []
        if avg_conf < 0.5:
            warnings.append("low_ocr_confidence")
        if not normalized_text:
            warnings.append("empty_ocr_text")

        return OCRResult(
            text=normalized_text,
            confidence=avg_conf,
            lines=lines,
            raw_text=raw_text,
            warnings=warnings,
            metadata={
                "provider": self.name,
                "version": self.version,
                "language": self.language,
                "device": self._device,
                "region_bbox": _bbox_to_dict(region_bbox) if region_bbox else None,
            },
        )


def _bbox_to_dict(bbox: BBox) -> dict[str, int]:
    return {"x1": bbox.x1, "y1": bbox.y1, "x2": bbox.x2, "y2": bbox.y2}
