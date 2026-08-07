"""Bölüm girdi yükleyici.

Bir bölüm klasöründeki tüm görüntü dosyalarını bulur, doğal sıralar
(001, 002, 010 gibi), boyutlarını okur ve tutarlı olduğunu doğrular.
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from core.config import Config
from core.models import Page


def natural_sort_key(path: Path) -> list[int | str]:
    """Doğal sıralama anahtarı üretir.

    Örnek: 1.webp, 2.webp, 10.webp -> 1, 2, 10 (doğru sıra)
    Sayısal olmayan dosya adları da desteklenir.

    Args:
        path: Dosya yolu.

    Returns:
        Sıralama anahtarı: sayılar int, metinler str olarak listelenir.
    """

    def convert(text: str) -> int | str:
        return int(text) if text.isdigit() else text.lower()

    # Dosya adından uzantıyı ayırır ve sayısal/alfabetik parçalara böler
    stem = path.stem
    return [convert(c) for c in re.split(r"([0-9]+)", stem)]


def list_image_files(folder: str | Path, extensions: list[str]) -> list[Path]:
    """Klasördeki tüm görüntü dosyalarını doğal sırada döndürür.

    Args:
        folder: Bölüm klasörü.
        extensions: Kabul edilen uzantılar (ör. [".webp", ".png"]).

    Returns:
        Doğal sırada dosya yolları listesi.

    Raises:
        FileNotFoundError: Klasör bulunamazsa.
        ValueError: Klasörde görüntü dosyası yoksa.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Klasör bulunamadı: {folder_path}")

    ext_set = {ext.lower() for ext in extensions}
    files = [
        f
        for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in ext_set
    ]

    if not files:
        raise ValueError(
            f"'{folder_path}' içinde görüntü dosyası bulunamadı. "
            f"Kabul edilen uzantılar: {sorted(ext_set)}"
        )

    files.sort(key=natural_sort_key)
    return files


def load_chapter(
    folder: str | Path,
    config: Config | None = None,
) -> list[Page]:
    """Bir bölüm klasörünü yükler ve Page nesneleri listesi döndürür.

    Yükleme sırasında:
    - Görüntü dosyaları doğal sırada sıralanır.
    - Her dosyanın genişlik/yüksekliği okunur.
    - Tüm sayfaların aynı genişlikte olduğu doğrulanır.
    - Her sayfaya kümülatif y_offset atanır (global koordinat sistemi).

    Args:
        folder: Bölüm klasörünün yolu.
        config: Yapılandırma. Varsayılan olarak config.yaml'dan yüklenir.

    Returns:
        Sıralı Page nesneleri listesi.

    Raises:
        FileNotFoundError: Klasör bulunamazsa.
        ValueError: Görüntü yoksa veya genişlikler tutarsızsa.
    """
    from PIL import Image

    cfg = config if config is not None else Config()

    image_paths = list_image_files(folder, cfg.input_extensions)
    logger.info(f"Toplam {len(image_paths)} görüntü bulundu: {folder}")

    pages: list[Page] = []
    y_offset = 0

    for index, path in enumerate(image_paths):
        try:
            with Image.open(path) as img:
                width, height = img.size
        except Exception as e:
            raise ValueError(f"Görüntü okunamadı: {path} — {e}") from e

        page = Page(
            index=index,
            path=path.resolve(),
            width=width,
            height=height,
            y_offset=y_offset,
        )
        pages.append(page)
        y_offset += height
        logger.debug(
            f"Sayfa {index}: {page.name} ({width}x{height}) "
            f"global y {page.y_offset}-{page.y_end}"
        )

    # Genişlik tutarlılığı kontrolü
    first_width = pages[0].width
    inconsistent = [p for p in pages if p.width != first_width]
    if inconsistent:
        names = ", ".join(p.name for p in inconsistent[:5])
        raise ValueError(
            f"Tüm sayfalar aynı genişlikte olmalı. "
            f"İlk sayfa genişliği {first_width}px, "
            f"uyumsuz sayfalar: {names}"
        )

    logger.info(
        f"Bölüm yüklendi: {len(pages)} sayfa, "
        f"genişlik {first_width}px, toplam yükseklik {y_offset}px"
    )
    return pages