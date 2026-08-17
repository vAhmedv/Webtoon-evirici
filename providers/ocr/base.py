"""OCR sağlayıcı temel sınıfı.

OCR provider'ın sorumluluğu yalnızca: verilmiş bir Region crop'u üzerindeki
İngilizce metni okumaktır. Chapter-level text detection yapmaz; o işi
YOLO section segmenter yapar. OCR kendi crop içinde text-line segmentation
kullanabilir.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from core.detection import BBox


class OCRLine:
    """Tek bir OCR satırı sonucu.

    Attributes:
        text: Satır metni (normalize edilmemiş, ham OCR çıktısı).
        confidence: Satır güven skoru (0.0 - 1.0).
        bbox: Satırın crop-local bbox'u (opsiyonel).
        polygon: Satırın crop-local polygon'u (opsiyonel).
    """

    def __init__(
        self,
        text: str,
        confidence: float,
        bbox: BBox | None = None,
        polygon: list[list[float]] | None = None,
    ) -> None:
        self.text = text
        self.confidence = confidence
        self.bbox = bbox
        self.polygon = polygon


class OCRResult:
    """OCR sonucu (canonical).

    Attributes:
        text: Normalize edilmiş kanonical metin (satırlar whitespace ile
            birleştirilmiş, baş/son boşluk kırılmış). Satır sonları korunmaz;
            ham satırlar ``lines`` içinde saklanır.
        confidence: Aggregate güven (0.0 - 1.0). Boş sonuçta 0.0.
        raw_text: Provider'ın ürettiği ham metin (satır sonları korunmuş).
            Mümkünse doldurulur; provider ham çıktıyı vermiyorsa ``text``'e eşittir.
        lines: Metin satırları (top→bottom, left→right okuma sırası).
        warnings: Kalite uyarıları (ör. "empty_ocr_result", "low_ocr_confidence").
        metadata: Provider/model/bbox vs. ek meta veri.
    """

    def __init__(
        self,
        text: str,
        confidence: float,
        lines: Sequence[OCRLine] | None = None,
        raw_text: str | None = None,
        warnings: list[str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.text = text
        self.confidence = confidence
        self.raw_text = raw_text if raw_text is not None else text
        self.lines = list(lines) if lines else []
        self.warnings = list(warnings) if warnings else []
        self.metadata = dict(metadata) if metadata else {}


class OCRProvider(ABC):
    """OCR sağlayıcı arayüzü.

    Uygulamanın geri kalanı hangi OCR motoru kullanılıyorsa bilmez.
    """

    @abstractmethod
    def load(self) -> None:
        """Modeli yükler (GPU/CPU hazırlığı)."""

    @abstractmethod
    def unload(self) -> None:
        """Yükü serbest bırakır (VRAM temizlik dahil)."""

    @abstractmethod
    def recognize(
        self,
        image,
        region_bbox: BBox | None = None,
    ) -> OCRResult:
        """Region crop'u üzerinde OCR yapar.

        Args:
            image: Giriş görüntüsü (PIL Image veya numpy array).
            region_bbox: Opsiyel global bbox (debug/metadata için).

        Returns:
            OCRResult.
        """

    def recognize_batch(
        self,
        images: Sequence[Any],
        region_bboxes: Sequence[BBox | None] | None = None,
    ) -> Sequence[OCRResult]:
        """Toplu görsel OCR tanıma.

        Args:
            images: Giriş görüntüleri dizisi.
            region_bboxes: Opsiyonel global bbox dizisi.

        Returns:
            OCRResult dizisi.
        """
        if not images:
            return []
        bboxes = region_bboxes if region_bboxes is not None else [None] * len(images)
        return [self.recognize(img, bbox) for img, bbox in zip(images, bboxes)]

    @property
    @abstractmethod
    def name(self) -> str:
        """Sağlayıcı insan okunabilir adı."""

    @property
    def version(self) -> str:
        """Model/ motor sürümü."""
        return "unknown"

    @property
    def device(self) -> str:
        """Çalışma cihazı (cpu/cuda)."""
        return "cpu"

    @property
    def language(self) -> str:
        """Tanıma dili/etki alanı (ör. 'en', 'multi')."""
        return "multi"

    @property
    def status(self) -> str:
        """Kayıt statüsü (candidate/default, candidate, stable)."""
        return "candidate"

    @property
    def is_loaded(self) -> bool:
        """Yük durumu."""
        return False
