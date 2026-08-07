"""Bölüm girdi yükleyici testleri."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from core.config import Config
from core.io.input_loader import list_image_files, load_chapter, natural_sort_key


def _make_image(path: Path, width: int = 800, height: int = 1000, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    """Test görüntüsü oluşturur."""
    img = Image.new("RGB", (width, height), color)
    img.save(path)


def test_natural_sort_key_sorts_numerically() -> None:
    """Doğal sıralama sayıları doğru sıralamalı."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        files = [
            root / "10.webp",
            root / "2.webp",
            root / "1.webp",
        ]
        # Doğal sıralama: 1, 2, 10
        sorted_files = sorted(files, key=natural_sort_key)
        assert [f.name for f in sorted_files] == ["1.webp", "2.webp", "10.webp"]


def test_natural_sort_key_mixed_names() -> None:
    """Karışık dosya adları desteklenmeli."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        files = [
            root / "page_10.webp",
            root / "page_2.webp",
            root / "cover.webp",
        ]
        sorted_files = sorted(files, key=natural_sort_key)
        assert [f.name for f in sorted_files] == ["cover.webp", "page_2.webp", "page_10.webp"]


def test_list_image_files_returns_sorted(tmp_path: Path) -> None:
    """list_image_files doğal sırada dosya döndürmeli."""
    (tmp_path / "10.webp").write_bytes(b"x")
    (tmp_path / "2.webp").write_bytes(b"x")
    (tmp_path / "1.webp").write_bytes(b"x")
    (tmp_path / "not_image.txt").write_text("merhaba")

    result = list_image_files(tmp_path, [".webp"])
    assert [f.name for f in result] == ["1.webp", "2.webp", "10.webp"]


def test_list_image_files_missing_folder_raises(tmp_path: Path) -> None:
    """Var olmayan klasör FileNotFoundError fırlatmalı."""
    with pytest.raises(FileNotFoundError):
        list_image_files(tmp_path / "yok", [".webp"])


def test_list_image_files_empty_raises(tmp_path: Path) -> None:
    """Görüntü içermeyen klasör ValueError fırlatmalı."""
    (tmp_path / "not_image.txt").write_text("merhaba")
    with pytest.raises(ValueError):
        list_image_files(tmp_path, [".webp"])


def test_load_chapter_orders_and_offsets(tmp_path: Path) -> None:
    """load_chapter doğal sıralamalı ve y_offset atamalı."""
    # Farklı yüksekliklerde 3 sayfa oluştur
    _make_image(tmp_path / "002.webp", width=800, height=1000)
    _make_image(tmp_path / "001.webp", width=800, height=1500)
    _make_image(tmp_path / "003.webp", width=800, height=840)

    pages = load_chapter(tmp_path)

    assert len(pages) == 3
    # Doğal sıralama: 001, 002, 003
    assert [p.name for p in pages] == ["001.webp", "002.webp", "003.webp"]

    # Boyutlar
    assert pages[0].height == 1500
    assert pages[1].height == 1000
    assert pages[2].height == 840

    # y_offset'lar kümülatif olmalı
    assert pages[0].y_offset == 0
    assert pages[1].y_offset == 1500
    assert pages[2].y_offset == 2500

    # y_end'ler
    assert pages[0].y_end == 1500
    assert pages[1].y_end == 2500
    assert pages[2].y_end == 3340

    # Genişlikler aynı
    assert pages[0].width == pages[1].width == pages[2].width == 800


def test_load_chapter_inconsistent_width_raises(tmp_path: Path) -> None:
    """Farklı genişlikteki sayfalar ValueError fırlatmalı."""
    _make_image(tmp_path / "001.webp", width=800, height=1000)
    _make_image(tmp_path / "002.webp", width=900, height=1000)

    with pytest.raises(ValueError, match="genişlik"):
        load_chapter(tmp_path)


def test_load_chapter_missing_folder_raises(tmp_path: Path) -> None:
    """Var olmayan klasör FileNotFoundError fırlatmalı."""
    from core.config import Config

    cfg = Config()
    with pytest.raises(FileNotFoundError):
        load_chapter(tmp_path / "yok", cfg)


def test_load_chapter_with_config_extensions(tmp_path: Path) -> None:
    """Config uzantılarına göre dosyalar yüklenmeli."""
    _make_image(tmp_path / "001.png", width=800, height=1000)
    _make_image(tmp_path / "002.webp", width=800, height=1000)
    _make_image(tmp_path / "003.jpg", width=800, height=1000)
    (tmp_path / "004.txt").write_text("yok sayılmalı")

    cfg = Config(input_extensions=[".png", ".webp", ".jpg", ".jpeg"])
    pages = load_chapter(tmp_path, cfg)

    assert len(pages) == 3
    assert [p.name for p in pages] == ["001.png", "002.webp", "003.jpg"]