"""Sentetik test bölümü üretici.

Phase 1 kabul testleri için farklı yüksekliklerde 5 WEBP sayfası oluşturur.
Sayfalar doğal sıralamayı test edecek şekilde karışık isimlerle oluşturulur.

Kullanım:
    python scripts/generate_test_chapter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Proje kökünü sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw

from core.config import load_config
from core.logging_setup import setup_logging

DEFAULT_OUTPUT = PROJECT_ROOT / "test_data" / "chapter_test"

# (dosya_adı, yükseklik, panel rengi, metin)
PAGES = [
    ("003.webp", 1000, (200, 220, 240), "Page 3"),
    ("001.webp", 1500, (220, 200, 240), "Page 1"),
    ("005.webp", 1200, (240, 220, 200), "Page 5"),
    ("002.webp", 840, (200, 240, 220), "Page 2"),
    ("004.webp", 1100, (240, 200, 200), "Page 4"),
]

WIDTH = 800


def generate_test_chapter(output_dir: str | Path = DEFAULT_OUTPUT) -> Path:
    """Test bölümünü üretir.

    Args:
        output_dir: Çıktı klasörü.

    Returns:
        Oluşturulan klasörün yolu.
    """
    config = load_config()
    setup_logging(config.log_level, config.log_file)

    from loguru import logger

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Test bölümü oluşturuluyor: {out}")

    for filename, height, bg_color, text in PAGES:
        img = Image.new("RGB", (WIDTH, height), bg_color)
        draw = ImageDraw.Draw(img)

        # Üst kısımda sayfa adı
        draw.text((20, 20), f"{text} ({WIDTH}x{height})", fill=(0, 0, 0))

        # Sayfayı taklit eden basit birkaç panel çiz
        panel_margin = 40
        panel_padding = 30
        panel_height = (height - 2 * panel_margin - panel_padding) // 2

        # Üst panel
        y1 = panel_margin
        y2 = y1 + panel_height
        draw.rectangle([panel_margin, y1, WIDTH - panel_margin, y2], outline=(0, 0, 0), width=3)
        draw.text((panel_margin + 20, y1 + 20), "Panel A", fill=(0, 0, 0))

        # Alt panel
        y3 = y2 + panel_padding
        y4 = y3 + panel_height
        draw.rectangle([panel_margin, y3, WIDTH - panel_margin, y4], outline=(0, 0, 0), width=3)
        draw.text((panel_margin + 20, y3 + 20), "Panel B", fill=(0, 0, 0))

        save_path = out / filename
        img.save(save_path, "WEBP", quality=95)
        logger.info(f"Oluşturuldu: {save_path.name} ({WIDTH}x{height})")

    total = sum(h for _, h, _, _ in PAGES)
    logger.info(f"Test bölümü tamamlandı: {len(PAGES)} sayfa, toplam yükseklik {total}px")
    logger.info(f"Klasör: {out}")

    return out


if __name__ == "__main__":
    generate_test_chapter()