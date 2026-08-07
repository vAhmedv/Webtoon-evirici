"""Phase 1 pipeline scripti.

Bölüm klasörünü yükler, global koordinat sistemi oluşturur, sliding window
üretir ve her window için önizleme görseli kaydeder.

Kullanım:
    python scripts/process_chapter.py --input <bölüm_klasörü> --output <çıktı_klasörü>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Proje kökünü sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw

from core.config import load_config
from core.coordinate.global_coords import GlobalCoordinateSystem
from core.coordinate.sliding_window import generate_windows_for_pages
from core.io.input_loader import load_chapter
from core.logging_setup import setup_logging

# Önizleme görseli maksimum genişliği
PREVIEW_WIDTH = 400


def _draw_preview(
    window,
    pages,
    coords: GlobalCoordinateSystem,
    output_path: Path,
) -> None:
    """Tek bir window için önizleme görseli oluşturur.

    Window'a denk gelen sayfa bölgelerini yükler, küçültür ve
    window/sayfa sınırlarını çizer.

    Args:
        window: Window nesnesi.
        pages: Tüm sayfalar.
        coords: Global koordinat sistemi.
        output_path: Kaydedilecek yol.
    """
    from loguru import logger

    # Window'a denk gelen sayfaları bul
    relevant_pages = coords.pages_in_range(window.y_start, window.y_end)

    # Global Y aralığını sayfa bazlı crop'lara çevir
    crops: list[tuple[Image.Image, int]] = []  # (crop, global_y_offset)
    for page in relevant_pages:
        # Window'un bu sayfayla kesiştiği aralık
        local_start = max(0, window.y_start - page.y_offset)
        local_end = min(page.height, window.y_end - page.y_offset)

        if local_end <= local_start:
            continue

        with Image.open(page.path) as img:
            crop = img.crop((0, local_start, page.width, local_end))
            crops.append((crop, page.y_offset + local_start))

    if not crops:
        logger.warning(f"Window {window.id} için görüntü bulunamadı")
        return

    # Crop'ları global Y'ye göre sırala ve dikey birleştir
    crops.sort(key=lambda c: c[1])
    images = [c[0] for c in crops]

    total_h = sum(img.height for img in images)
    max_w = max(img.width for img in images)

    combined = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y_cursor = 0
    for img in images:
        combined.paste(img, (0, y_cursor))
        y_cursor += img.height

    # Küçült
    scale = PREVIEW_WIDTH / combined.width
    preview_h = int(combined.height * scale)
    preview = combined.resize((PREVIEW_WIDTH, preview_h), Image.LANCZOS)

    draw = ImageDraw.Draw(preview)

    # Window sınırı (kırmızı)
    draw.rectangle([0, 0, PREVIEW_WIDTH - 1, preview_h - 1], outline=(255, 0, 0), width=3)

    # Sayfa sınırları (mavi) — global Y'yi önizleme koordinatına çevir
    for page in relevant_pages:
        # Sayfanın window içindeki global aralığı
        page_start = max(page.y_offset, window.y_start)
        page_end = min(page.y_end, window.y_end)
        if page_end <= page_start:
            continue

        # Önizleme koordinatına çevir
        preview_y = int((page_start - window.y_start) * scale)
        draw.line([(0, preview_y), (PREVIEW_WIDTH, preview_y)], fill=(0, 0, 255), width=2)

    # Window bilgisi (üstte)
    draw.text(
        (10, 10),
        f"Window {window.id}: y {window.y_start}-{window.y_end} "
        f"(sayfalar {window.page_indices})",
        fill=(255, 0, 0),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path, "PNG")
    logger.info(f"Önizleme kaydedildi: {output_path}")


def process_chapter(input_dir: str | Path, output_dir: str | Path) -> None:
    """Phase 1 pipeline'ını çalıştırır.

    Args:
        input_dir: Bölüm klasörü.
        output_dir: Çıktı klasörü.
    """
    config = load_config()
    setup_logging(config.log_level, config.log_file)

    from loguru import logger

    start_time = time.time()

    logger.info("=== Phase 1 Pipeline Başlıyor ===")
    logger.info(f"Girdi: {input_dir}")
    logger.info(f"Çıktı: {output_dir}")
    logger.info(
        f"Config: window_height={config.window_height}, "
        f"window_overlap={config.window_overlap}"
    )

    # 1. Bölümü yükle
    pages = load_chapter(input_dir, config)
    logger.info(f"Sayfa sırası: {[p.name for p in pages]}")

    # 2. Global koordinat sistemi
    coords = GlobalCoordinateSystem(tuple(pages))
    logger.info(f"Toplam yükseklik: {coords.total_height}px")
    logger.info(f"Genişlik: {coords.width}px")

    # 3. Sliding window üret
    windows = generate_windows_for_pages(
        pages,
        window_height=config.window_height,
        overlap=config.window_overlap,
    )
    logger.info(f"Toplam window: {len(windows)}")

    # 4. Önizleme görselleri
    preview_dir = Path(output_dir) / "windows"
    for window in windows:
        _draw_preview(window, pages, coords, preview_dir / f"window_{window.id:03d}.png")

    # 5. Özet
    elapsed = time.time() - start_time
    logger.info("=== Phase 1 Pipeline Tamamlandı ===")
    logger.info(f"Girdi görüntüleri: {len(pages)}")
    logger.info(f"Window sayısı: {len(windows)}")
    logger.info(f"Toplam yükseklik: {coords.total_height}px")
    logger.info(f"Önizleme klasörü: {preview_dir}")
    logger.info(f"Toplam süre: {elapsed:.2f} saniye")


def main() -> None:
    """Komut satırı giriş noktası."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Bölüm yükle + global koordinat + sliding window"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Bölüm klasörü (WEBP/PNG/JPG dosyaları içeren)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Çıktı klasörü (window önizlemeleri buraya yazılır)",
    )
    args = parser.parse_args()

    process_chapter(args.input, args.output)


if __name__ == "__main__":
    main()