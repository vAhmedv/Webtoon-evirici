"""Bölüm görüntüsü yükleme ve doğrulama modülü.

Bir bölüm klasöründeki görüntü dosyalarını listeler, sıralar, boyut ve
format doğrulaması yapar. Görüntüler diske yazılmaz, salt-okunur işlenir.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from loguru import logger

from core.config import Config
from core.models import Page


# Desteklenen varsayılan görüntü uzantıları
DEFAULT_EXTENSIONS = (
    ".webp",
    ".png",
    ".jpg",
    ".jpeg",
)


def list_image_files(
    folder: str | Path,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
) -> list[Path]:
    """Klasördeki görüntü dosyalarını arar ve sıralar."""
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Bölüm klasörü bulunamadı: {folder}")

    normalized_exts = tuple(ext.lower() for ext in extensions)

    files: list[Path] = []
    for entry in folder_path.iterdir():
        if entry.is_file() and entry.suffix.lower() in normalized_exts:
            files.append(entry)

    if not files:
        raise ValueError(
            f"Klasörde desteklenen görüntü bulunamadı ({folder}). "
            f"Aranan uzantılar: {extensions}"
        )

    files.sort(key=_natural_sort_key)
    return files


def _natural_sort_key(path: Path) -> list:
    """Doğal sıralama (natural sort) anahtarı üretir."""
    import re

    parts = re.split(r"(\d+)", path.name)
    return [int(text) if text.isdigit() else text.lower() for text in parts]


natural_sort_key = _natural_sort_key


def load_chapter(
    folder: str | Path,
    config: Config | None = None,
    allow_non_uniform_widths: bool = False,
) -> list[Page]:
    """Bölüm klasörünü yükler ve sıralı Page listesi döndürür."""
    from core.imaging.fast_loader import get_image_dimensions

    cfg = config if config is not None else Config()

    image_paths = list_image_files(folder, cfg.input_extensions)
    logger.info(f"Toplam {len(image_paths)} görüntü bulundu: {folder}")

    raw_dims: list[tuple[int, int]] = []
    for path in image_paths:
        try:
            raw_dims.append(get_image_dimensions(path))
        except Exception as e:
            raise ValueError(f"Görüntü okunamadı: {path} — {e}") from e

    first_width = raw_dims[0][0]
    inconsistent = [p for p, (w, _) in zip(image_paths, raw_dims) if w != first_width]

    if inconsistent and not allow_non_uniform_widths:
        names = ", ".join(p.name for p in inconsistent[:5])
        raise ValueError(
            f"Tüm sayfalar aynı genişlikte olmalı. "
            f"İlk sayfa genişliği {first_width}px, "
            f"uyumsuz sayfalar: {names}"
        )

    width_counts = Counter(w for w, _ in raw_dims)
    target_width = width_counts.most_common(1)[0][0]

    pages: list[Page] = []
    y_offset = 0

    for index, (path, (w, h)) in enumerate(zip(image_paths, raw_dims)):
        if w != target_width:
            scaled_h = max(1, int(round(h * target_width / w)))
            w, h = target_width, scaled_h

        page = Page(
            index=index,
            path=path.resolve(),
            width=w,
            height=h,
            y_offset=y_offset,
        )
        pages.append(page)
        y_offset += h

    logger.info(
        f"Bölüm yüklendi: {len(pages)} sayfa, "
        f"genişlik {target_width}px, toplam yükseklik {y_offset}px"
    )
    return pages