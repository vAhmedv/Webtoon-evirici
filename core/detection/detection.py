"""Detection ve Region veri modelleri."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .bbox import BBox


class RegionType(str, Enum):
    """Bölge türü."""

    DIALOGUE = "dialogue"
    NARRATION = "narration"
    SFX = "sfx"
    WATERMARK = "watermark"
    UNKNOWN = "unknown"


class RegionStatus(str, Enum):
    """Kalite kapısı durumu."""

    AUTO = "auto"
    REVIEW = "review"
    SKIP = "skip"


@dataclass(frozen=True)
class Detection:
    """Ham detector çıktısı.

    Provider'dan gelen ham sonuçları temsil eder. Canonical Region'a
    dönüştürülmeden önce kullanılır.

    Attributes:
        bbox: Window-local BBox.
        confidence: Güven skoru (0.0 - 1.0).
        type: Tahmini bölge türü.
        source_window_id: İlk üretim window'ının kimliği.
        mask: Opsiyonel maske (ileride kullanılacak).
        metadata: Provider'a özel ek veri.
    """

    bbox: BBox
    confidence: float
    type: RegionType
    source_window_id: int
    mask: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Region:
    """Canonical global bölge.

    Aynı içerik birden fazla window'da tespit edilirse merge edilir.
    Pipeline'ın geri kalanı Detection yerine Region ile çalışır.

    Attributes:
        id: Sabit kimlik (otomatik atanabilir).
        global_bbox: Global koordinatlı BBox.
        type: Bölge türü.
        detection_confidence: Tespit güven skoru.
        source_window_ids: Katkıda bulunan window kimlikleri.
        status: Kalite kapısı durutu.
        text: OCR sonrası metin (ileride doldurulacak).
        ocr_confidence: OCR güven skoru (ileride).
        translation: Çeviri (ileride).
        review_reason: REVIEW durumunda açıklama.
    """

    id: int
    global_bbox: BBox
    type: RegionType
    detection_confidence: float
    source_window_ids: tuple[int, ...]
    status: RegionStatus = RegionStatus.AUTO
    text: str | None = None
    ocr_confidence: float | None = None
    translation: str | None = None
    review_reason: str | None = None