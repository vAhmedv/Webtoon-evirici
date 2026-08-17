"""Unit tests for core/imaging/fast_loader.py."""

from __future__ import annotations

import io
import time
from pathlib import Path
from PIL import Image
import pytest

from core.imaging.fast_loader import (
    get_image_dimensions,
    _get_png_dimensions,
    _get_jpeg_dimensions,
    _get_webp_dimensions,
)


@pytest.fixture
def sample_images(tmp_path: Path) -> dict[str, Path]:
    """Create test image files across multiple formats."""
    w, h = 800, 14000
    images = {}

    # 1. PNG
    png_path = tmp_path / "test.png"
    Image.new("RGB", (w, h), color="red").save(png_path, format="PNG")
    images["png"] = png_path

    # 2. JPEG Baseline
    jpg_base_path = tmp_path / "test_base.jpg"
    Image.new("RGB", (w, h), color="green").save(jpg_base_path, format="JPEG", progressive=False)
    images["jpeg_base"] = jpg_base_path

    # 3. JPEG Progressive
    jpg_prog_path = tmp_path / "test_prog.jpg"
    Image.new("RGB", (w, h), color="blue").save(jpg_prog_path, format="JPEG", progressive=True)
    images["jpeg_prog"] = jpg_prog_path

    # 4. WebP Lossy
    webp_lossy_path = tmp_path / "test_lossy.webp"
    Image.new("RGB", (w, h), color="yellow").save(webp_lossy_path, format="WEBP", lossless=False)
    images["webp_lossy"] = webp_lossy_path

    # 5. WebP Lossless
    webp_lossless_path = tmp_path / "test_lossless.webp"
    Image.new("RGB", (w, h), color="purple").save(webp_lossless_path, format="WEBP", lossless=True)
    images["webp_lossless"] = webp_lossless_path

    return images


def test_png_dimensions(sample_images: dict[str, Path]) -> None:
    """PNG dosya başlığından boyutları doğru okur."""
    w, h = get_image_dimensions(sample_images["png"])
    assert w == 800
    assert h == 14000


def test_jpeg_baseline_dimensions(sample_images: dict[str, Path]) -> None:
    """Baseline JPEG dosya başlığından boyutları doğru okur."""
    w, h = get_image_dimensions(sample_images["jpeg_base"])
    assert w == 800
    assert h == 14000


def test_jpeg_progressive_dimensions(sample_images: dict[str, Path]) -> None:
    """Progressive JPEG dosya başlığından boyutları doğru okur."""
    w, h = get_image_dimensions(sample_images["jpeg_prog"])
    assert w == 800
    assert h == 14000


def test_webp_lossy_dimensions(sample_images: dict[str, Path]) -> None:
    """WebP VP8 Lossy dosya başlığından boyutları doğru okur."""
    w, h = get_image_dimensions(sample_images["webp_lossy"])
    assert w == 800
    assert h == 14000


def test_webp_lossless_dimensions(sample_images: dict[str, Path]) -> None:
    """WebP VP8L Lossless dosya başlığından boyutları doğru okur."""
    w, h = get_image_dimensions(sample_images["webp_lossless"])
    assert w == 800
    assert h == 14000


def test_webp_extended_vp8x_dimensions(tmp_path: Path) -> None:
    """WebP VP8X Extended formatından boyutları doğru okur."""
    vp8x_path = tmp_path / "test_vp8x.webp"
    # PIL adds VP8X chunk when ICC profile or EXIF is attached
    img = Image.new("RGBA", (1200, 16000), color=(255, 0, 0, 128))
    img.save(vp8x_path, format="WEBP")
    w, h = get_image_dimensions(vp8x_path)
    assert w == 1200
    assert h == 16000


def test_fallback_mechanism(tmp_path: Path) -> None:
    """Binary header okuyucu tanımazsa PIL fallback mekanizması çalışır."""
    bmp_path = tmp_path / "test.bmp"
    Image.new("RGB", (640, 480), color="white").save(bmp_path, format="BMP")
    w, h = get_image_dimensions(bmp_path)
    assert w == 640
    assert h == 480


def test_nonexistent_file_raises_error(tmp_path: Path) -> None:
    """Mevcut olmayan dosya için FileNotFoundError fırlatılır."""
    with pytest.raises(FileNotFoundError):
        get_image_dimensions(tmp_path / "non_existent.png")


def test_corrupt_file_raises_value_error(tmp_path: Path) -> None:
    """Bozuk ve okunamayan dosya için ValueError fırlatılır."""
    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"\x00\x01\x02\x03not_an_image")
    with pytest.raises(ValueError):
        get_image_dimensions(corrupt_path)


def test_header_reading_speed(sample_images: dict[str, Path]) -> None:
    """Her formatta başlık okuma ortalama süresi <5 ms olmalıdır."""
    for name, path in sample_images.items():
        # Warm-up
        get_image_dimensions(path)

        # Benchmark 10 iterations
        t0 = time.perf_counter()
        for _ in range(10):
            w, h = get_image_dimensions(path)
            assert w == 800
            assert h == 14000
        avg_ms = ((time.perf_counter() - t0) / 10.0) * 1000
        assert avg_ms < 5.0, f"{name} header read took {avg_ms:.2f} ms (> 5 ms)!"
