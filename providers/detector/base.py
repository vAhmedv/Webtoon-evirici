"""Detector sağlayıcı temel sınıfı."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

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

    @property
    @abstractmethod
    def name(self) -> str:
        """Sağlayıcı insan okunabilir adı."""

    @property
    def is_loaded(self) -> bool:
        """Yük durumu."""
        return False