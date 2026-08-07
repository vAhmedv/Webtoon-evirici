"""OCR sağlayıcı paketi.

Provider abstraction: detector OCR yapmaz; OCR yalnızca canonical global
Region crop'ları üzerinde çalışır (spec: "OCR yalnızca canonical Region
crop'ları üzerinde çalışmalı").
"""

from providers.ocr.base import OCRLine, OCRProvider, OCRResult
from providers.ocr.registry import OCRRegistry, get_ocr_registry

try:
    from providers.ocr.rapid_onnx import RapidONNXOCR
except Exception:  # pragma: no cover
    RapidONNXOCR = None  # type: ignore

try:
    from providers.ocr.paddleocr import PaddleOCRProvider
except Exception:  # pragma: no cover
    PaddleOCRProvider = None  # type: ignore

__all__ = [
    "OCRLine",
    "OCRProvider",
    "OCRResult",
    "OCRRegistry",
    "get_ocr_registry",
    "RapidONNXOCR",
    "PaddleOCRProvider",
]
