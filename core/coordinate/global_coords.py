"""Global koordinat sistemi.

Suwayomi'nin parçaladığı WEBP'leri tek uzun webtoon gibi düşünmek için
her sayfaya kümülatif y_offset atanır. Bu modül iki yönlü dönüşüm sağlar:
- Sayfa + yerel Y -> Global Y
- Global Y -> Sayfa + yerel Y
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from core.models import Page


@dataclass(frozen=True)
class GlobalCoordinateSystem:
    """Global koordinat sistemini temsil eder.

    Attributes:
        pages: Sıralı Page nesneleri listesi (y_offset'ları atanmış olmalı).
    """

    pages: tuple[Page, ...]

    def __post_init__(self) -> None:
        # y_offset'ların tutarlı olduğunu doğrula
        expected_offset = 0
        for page in self.pages:
            if page.y_offset != expected_offset:
                raise ValueError(
                    f"Sayfa {page.index} ({page.name}) y_offset'u tutarsız: "
                    f"beklenen {expected_offset}, bulunan {page.y_offset}"
                )
            expected_offset += page.height

    @property
    def total_height(self) -> int:
        """Tüm bölümün toplam yüksekliği (global piksel)."""
        if not self.pages:
            return 0
        return self.pages[-1].y_end

    @property
    def width(self) -> int:
        """Bölüm genişliği (sayfalar aynı genişlikte olmalı)."""
        if not self.pages:
            return 0
        return self.pages[0].width

    def page_to_global(self, page_index: int, local_y: int) -> int:
        """Sayfa içi yerel Y koordinatını global Y'ye çevirir.

        Args:
            page_index: Sayfa indeksi (0'dan başlar).
            local_y: Sayfadaki yerel Y koordinatı.

        Returns:
            Global Y koordinatı.

        Raises:
            IndexError: Sayfa indeksi geçersizse.
            ValueError: local_y sayfa sınırları dışındaysa.
        """
        if not 0 <= page_index < len(self.pages):
            raise IndexError(f"Sayfa indeksi geçersiz: {page_index}")

        page = self.pages[page_index]
        if not 0 <= local_y < page.height:
            raise ValueError(
                f"local_y ({local_y}) sayfa {page.name} yüksekliği ({page.height}) dışında"
            )

        return page.y_offset + local_y

    def global_to_page(self, global_y: int) -> tuple[int, int]:
        """Global Y koordinatını (sayfa indeksi, yerel Y) çiftine çevirir.

        Args:
            global_y: Global koordinattaki Y.

        Returns:
            (page_index, local_y) ikilisi.

        Raises:
            ValueError: global_y bölüm sınırları dışındaysa.
        """
        if global_y < 0 or global_y >= self.total_height:
            raise ValueError(
                f"global_y ({global_y}) bölüm sınırları dışında (0-{self.total_height})"
            )

        # İkili arama ile sayfayı bul (verimli, çok sayıda sayfa için)
        low, high = 0, len(self.pages) - 1
        while low <= high:
            mid = (low + high) // 2
            page = self.pages[mid]
            if page.y_offset <= global_y < page.y_end:
                return page.index, global_y - page.y_offset
            if global_y < page.y_offset:
                high = mid - 1
            else:
                low = mid + 1

        # Doğrulanamaz durum (sınır kontrolü nedeniyle ulaşılmamalı)
        raise ValueError(f"global_y ({global_y}) için sayfa bulunamadı")

    def pages_in_range(self, y_start: int, y_end: int) -> list[Page]:
        """Belirtilen global Y aralığına denk gelen sayfaları döndürür.

        Args:
            y_start: Global başlangıç Y (inclusive).
            y_end: Global bitiş Y (exclusive).

        Returns:
            Aralıkla kesişen, sıralı Page listesi.
        """
        if y_start >= y_end:
            return []

        clipped_start = max(0, y_start)
        clipped_end = min(self.total_height, y_end)

        result: list[Page] = []
        for page in self.pages:
            if page.y_offset < clipped_end and page.y_end > clipped_start:
                result.append(page)
        return result


def compute_y_offsets(pages: list[Page]) -> list[Page]:
    """Sayfa listesine kümülatif y_offset atar ve yeni Page listesi döndürür.

    Args:
        pages: Page nesneleri (y_offset değerleri dikkate alınmaz).

    Returns:
        y_offset'u doğru atanmış yeni Page listesi.
    """
    offset = 0
    result: list[Page] = []
    for page in pages:
        new_page = Page(
            index=page.index,
            path=page.path,
            width=page.width,
            height=page.height,
            y_offset=offset,
        )
        result.append(new_page)
        offset += page.height
    logger.debug(f"y_offset hesaplandı: {len(result)} sayfa, toplam {offset}px")
    return result