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

    # Eski config.yaml'larda registry anahtarı yerine model adı yazılmıştı
    # (örn. "PaddleOCR-PP-OCRv6_medium_rec"). Sessiz yanlış davranış yerine
    # kanonik isme çevir.
    LEGACY_ALIASES: dict[str, str] = {
        "PaddleOCR-PP-OCRv6_medium_rec": "PaddleOCR-PP-OCRv6",
        "PaddleOCR-PP-OCRv6-server": "PaddleOCR-PP-OCRv6",
        "en_PP-OCRv5_mobile_rec": "PaddleOCR English v5",
        "PaddleOCR-VL-1.5": "PaddleOCR-VL-1.6",
        "PaddleOCR-VL": "PaddleOCR-VL-1.6",
    }

    def __init__(self) -> None:
        self._providers: dict[str, Callable[[], OCRProvider]] = {}
        self._status: dict[str, str] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        # PaddleOCR-VL-1.6: primary/default OCR (Transformers native)
        try:
            from providers.ocr.paddleocr_vl import PaddleOCRVLOcrProvider
            self.register("PaddleOCR-VL-1.6", PaddleOCRVLOcrProvider, status="candidate/default")
        except Exception:
            pass
        try:
            from providers.ocr.rapid_onnx import RapidONNXOCR
            self.register("RapidOCR-ONNX", RapidONNXOCR, status="candidate")
        except Exception:
            pass
        try:
            from providers.ocr.paddleocr import PaddleOCRProvider
            self.register("PaddleOCR-PP-OCRv6", PaddleOCRProvider, status="candidate")
            self.register(
                "PaddleOCR English v5",
                lambda: PaddleOCRProvider(model_name="en_PP-OCRv5_mobile_rec"),
                status="candidate",
            )
        except Exception:
            pass

    def register(self, name: str, factory: Callable[[], OCRProvider], status: str = "stable") -> None:
        self._providers[name] = factory
        self._status[name] = status

    def resolve_name(self, name: str | None) -> str | None:
        """Legacy/alternatif OCR ismini kanonik registry anahtarına çevir."""
        if not name:
            return name
        if name in self._providers:
            return name
        return self.LEGACY_ALIASES.get(name, name)

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def get_status(self, name: str) -> str:
        return self._status.get(name, "unknown")

    def create(self, name: str) -> OCRProvider:
        canonical = self.resolve_name(name)
        if canonical not in self._providers:
            available = ", ".join(sorted(self._providers)) or "<none>"
            raise KeyError(
                f"Unknown OCR provider: {name!r} (resolved: {canonical!r}). "
                f"Available: {available}"
            )
        return self._providers[canonical]()


_registry = OCRRegistry()


def get_ocr_registry() -> OCRRegistry:
    return _registry


def resolve_ocr_provider_name(name: str | None) -> str | None:
    """Modül seviyesi kolaylaştırıcı: legacy ismi kanoniğe çevir."""
    return _registry.resolve_name(name)
