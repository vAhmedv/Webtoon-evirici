"""Sayfa veri modeli.

Suwayomi'den gelen tek bir WEBP/PNG/JPG görüntüsünü temsil eder.
Global koordinat sistemindeki konumu y_offset ile belirlenir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Page:
    """Tek bir bölüm sayfası.

    Attributes:
        index: Sayfanın doğal sıradaki indeksi (0'dan başlar).
        path: Görüntü dosyasının yolu.
        width: Genişlik (piksel).
        height: Yükseklik (piksel).
        y_offset: Global koordinat sisteminde sayfanın başlangıç Y konumu.
    """

    index: int
    path: Path
    width: int
    height: int
    y_offset: int = 0

    @property
    def name(self) -> str:
        """Dosya adı (ör. 031.webp)."""
        return self.path.name

    @property
    def extension(self) -> str:
        """Dosya uzantısı (ör. .webp)."""
        return self.path.suffix.lower()

    @property
    def y_end(self) -> int:
        """Global koordinatta sayfanın bitiş Y konumu (exclusive)."""
        return self.y_offset + self.height