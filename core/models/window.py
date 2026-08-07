"""Window veri modeli.

Sliding window ile oluşturulan tek bir analiz penceresini temsil eder.
Global koordinat sisteminde y_start ile y_end arasında yer alır.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Window:
    """Sliding window penceresi.

    Attributes:
        id: Pencere kimliği (0'dan başlar, sıralı).
        y_start: Global koordinatta başlangıç Y (inclusive).
        y_end: Global koordinatta bitiş Y (exclusive).
        page_indices: Bu pencereye dahil olan sayfa indeksleri (sıralı).
        crop_offsets: Her sayfa için window'un global Y'sinden o sayfaya
            göre yerel Y'ye ofset haritası. Anahtar: sayfa indeksi,
            değer: o sayfada window'un başladığı yerel Y.
    """

    id: int
    y_start: int
    y_end: int
    page_indices: tuple[int, ...] = field(default_factory=tuple)

    @property
    def height(self) -> int:
        """Pencere yüksekliği (piksel)."""
        return self.y_end - self.y_start

    def contains_y(self, global_y: int) -> bool:
        """Belirtilen global Y bu pencerenin içinde mi?

        Args:
            global_y: Global koordinattaki Y değeri.

        Returns:
            True ise global_y bu pencerenin [y_start, y_end) aralığındadır.
        """
        return self.y_start <= global_y < self.y_end