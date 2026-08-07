"""OCR sağlayıcı kayıt defteri.

Kayıtlı sağlayıcılar:
- RapidOCR-ONNX: ONNX Runtime backend, PP-OCR aile modeli. Çalışır, gerçek
  lokal OCR. PyTorch ortamını bozmaz. candidate/default.
- PaddleOCR: PaddleOCR paketini (`paddleocr`) kullanan candidate. PP-OCRv6 /
  en_PP-OCRv5 ONNX backend'leri için. `paddleocr` kurulu değilse kayıtlı
  değildir; rapor "Candidate unavailable due to runtime compatibility".
  (PyTorch/CUDA 12.8 ortamının bozulmaması için otomatik kurma yapılmaz.)
"""

from __future__ import annotations

from typing import Callable

from providers.ocr.base import OCRProvider


class OCRRegistry:
    """OCR provider kayıt defteri."""

    def __init__(self) -> None:
        self._providers: dict[str, Callable[[], OCRProvider]] = {}
        self._status: dict[str, str] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        try:
            from providers.ocr.rapid_onnx import RapidONNXOCR
            self.register("RapidOCR-ONNX", RapidONNXOCR, status="candidate/default")
        except Exception:
            pass
        try:
            from providers.ocr.paddleocr import PaddleOCRProvider
            self.register("PaddleOCR-PP-OCRv6", PaddleOCRProvider, status="candidate")
        except Exception:
            pass

    def register(self, name: str, factory: Callable[[], OCRProvider], status: str = "stable") -> None:
        self._providers[name] = factory
        self._status[name] = status

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def get_status(self, name: str) -> str:
        return self._status.get(name, "unknown")

    def create(self, name: str) -> OCRProvider:
        if name not in self._providers:
            raise KeyError(f"Unknown OCR provider: {name}")
        return self._providers[name]()


_registry = OCRRegistry()


def get_ocr_registry() -> OCRRegistry:
    return _registry
