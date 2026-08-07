"""Bounding box veri modeli ve yardımcıları.

Kural: x1/y1 inclusive, x2/y2 exclusive.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """2D bounding box.

    Koordinatlar global piksel koordinatlarıdır.

    Attributes:
        x1: Sol kenar X (inclusive).
        y1: Üst kenar Y (inclusive).
        x2: Sağ kenar X (exclusive).
        y2: Alt kenar Y (exclusive).
    """

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x2 <= self.x1:
            raise ValueError(f"x2 ({self.x2}) x1'den ({self.x1}) küçük veya eşit olamaz")
        if self.y2 <= self.y1:
            raise ValueError(f"y2 ({self.y2}) y1'den ({self.y1}) küçük veya eşit olamaz")

    @property
    def width(self) -> int:
        """Genişlik (piksel)."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Yükseklik (piksel)."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Alan (piksel kare)."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Merkez koordinatları."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def intersection(self, other: BBox) -> BBox | None:
        """İki bbox'ın kesişimini döndürür.

        Args:
            other: Diğer bbox.

        Returns:
            Kesişim BBox'ı veya None (kesişim yoksa).
        """
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def iou(self, other: BBox) -> float:
        """Intersection over Union (IoU) skoru.

        Args:
            other: Diğer bbox.

        Returns:
            0.0 ile 1.0 arasında IoU değeri.
        """
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        union_area = self.area + other.area - inter.area
        if union_area <= 0:
            return 0.0
        return inter.area / union_area

    def clip(self, x1: int, y1: int, x2: int, y2: int) -> BBox | None:
        """Bbox'ı verilen sınıra göre kırpar.

        Args:
            x1: Kırpma sınırı sol üst X.
            y1: Kırpma sınırı sol üst Y.
            x2: Kırpma sınırı sağ alt X.
            y2: Kırpma sınırı sağ alt Y.

        Returns:
            Kırpılmış BBox veya None (eğer tamamen dışarıda kalırsa).
        """
        nx1 = max(self.x1, x1)
        ny1 = max(self.y1, y1)
        nx2 = min(self.x2, x2)
        ny2 = min(self.y2, y2)
        if nx2 <= nx1 or ny2 <= ny1:
            return None
        return BBox(x1=nx1, y1=ny1, x2=nx2, y2=ny2)

    def to_tuple(self) -> tuple[int, int, int, int]:
        """(x1, y1, x2, y2) tuple olarak döndürür."""
        return (self.x1, self.y1, self.x2, self.y2)

    @classmethod
    def from_tuple(cls, coords: tuple[int, int, int, int]) -> BBox:
        """(x1, y1, x2, y2) tuple'dan BBox oluşturur."""
        x1, y1, x2, y2 = coords
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)