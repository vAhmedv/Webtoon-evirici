"""config.yaml yükleyici testleri."""

from __future__ import annotations

from core.config import Config, DetectorConfig, OCRConfig, load_config


def test_load_config_returns_config() -> None:
    """config.yaml dosyası doğru yüklenmeli."""
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.window_height == 1024
    assert cfg.window_overlap == 256
    assert ".webp" in cfg.input_extensions
    assert cfg.output_format == "webp"
    assert cfg.translator.enabled is True
    assert cfg.translator.provider == "hy_mt2_gguf"
    assert cfg.translator.server_url == "http://127.0.0.1:8085"
    assert cfg.translator.model_path == r"C:\AI\Models\HY-MT2-7B-Q8_0.gguf"
    assert cfg.translator.llama_executable == r"C:\AI\llama-cpp-cuda\llama-server.exe"
    assert cfg.translator.fallback_provider is None


def test_load_config_defaults_when_file_missing(tmp_path) -> None:
    """Var olmayan dosya için varsayılan Config dönmeli."""
    cfg = Config()
    assert cfg.window_height == 1024
    assert cfg.window_overlap == 256
    assert cfg.output_format == "webp"
    assert cfg.min_confidence == 0.5


def test_load_config_missing_file_raises(tmp_path) -> None:
    """Var olmayan dosya için FileNotFoundError fırlatılmalı."""
    import pytest

    from core.config import load_config

    missing = tmp_path / "yok.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)


def test_nested_configs_have_defaults() -> None:
    """Alt yapılandırmalar varsayılan değerlerle başlamalı."""
    cfg = Config()
    assert isinstance(cfg.detector, DetectorConfig)
    assert cfg.detector.enabled is False
    assert cfg.detector.model is None
    assert cfg.ocr.enabled is False
    assert cfg.ocr.provider is None
    assert cfg.translator.enabled is False
    assert cfg.inpainter.enabled is False


def test_ocr_default_never_uses_translation_provider() -> None:
    """OCR varsayılanı bir translation backend'ine bağlanmamalı."""
    assert OCRConfig().provider is None
    assert OCRConfig().provider != "hy_mt2_gguf"
