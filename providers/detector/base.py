"""Detector sağlayıcı temel sınıfı."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from core.detection import Detection, RegionType


class DetectorProvider(ABC):
    """Detector sağlayıcı arayüzü.

    Uygulamanın geri kalanı hangi detector kullanılıyorsa bilmez.
    """

    @abstractmethod
    def load(self) -> None:
        """Modeli yükler (GPU/CPU hazırlığı)."""

    @abstractmethod
    def unload(self) -> None:
        """Yükü serbest bırakır."""

    @abstractmethod
    def detect(self, image, window_id: int) -> Sequence[Detection]:
        """Görüntü üzerinde tespit yapar.

        Args:
            image: Giriş görüntüsü (PIL Image veya uygun format).
            window_id: Görüntünün ait olduğu window kimliği.

        Returns:
            Detection listesi (window-local koordinatlı).
        """

    def detect_batch(self, items: Sequence[tuple[Any, int]]) -> Sequence[Sequence[Detection]]:
        """Toplu pencere tespiti (Varsayılan olarak tek tek detect çağırır).

        Args:
            items: (image, window_id) tuple listesi.

        Returns:
            Her pencere için Detection listesi.
        """
        return [self.detect(img, wid) for img, wid in items]

    @property
    @abstractmethod
    def name(self) -> str:
        """Sağlayıcı insan okunabilir adı."""

    @property
    def is_loaded(self) -> bool:
        """Yük durumu."""
        return False

    @property
    def confidence_threshold(self) -> float:
        """Tespit skor eşiği (varsayılan 0.5; sağlayıcılar geçersiz kılabilir)."""
        return float(getattr(self, "_confidence_threshold", 0.5))

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        self._confidence_threshold = max(0.0, min(1.0, float(value)))