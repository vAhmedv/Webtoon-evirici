"""Yapılandırma yükleyici.

config.yaml dosyasını okur ve tip güvenli bir Config nesnesi döndürür.
Hard-code edilmiş parametre yoktur; her şey config.yaml'dan gelir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Proje kök dizini (bu dosyanın iki üstü: core/config.py -> proje kökü)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DetectionConfig:
    """Detection cache ayarları."""

    enabled: bool = True
    min_confidence: float = 0.5
    duplicate_iou_threshold: float = 0.5
    max_cache_entries: int = 512


@dataclass(frozen=True)
class DetectorConfig:
    """Detector ayarları."""

    enabled: bool = False
    model: str | None = None


@dataclass(frozen=True)
class OCRConfig:
    """OCR ayarları."""

    enabled: bool = False
    engine: str | None = None
    provider: str | None = None
    min_confidence: float = 0.5
    crop_padding: int = 20
    upscale_small_regions: bool = False
    upscale_factor: float = 2.0


@dataclass(frozen=True)
class TranslatorConfig:
    """Çeviri ayarları."""

    enabled: bool = False
    provider: str | None = None
    qwen_model: str | None = None
    model_path: str | None = None
    llama_executable: str | None = None
    server_url: str | None = None
    fallback_provider: str | None = None


@dataclass(frozen=True)
class InpainterConfig:
    """Inpainting ayarları."""

    enabled: bool = False
    model: str | None = None


@dataclass(frozen=True)
class Config:
    """Uygulama geneli yapılandırma nesnesi."""

    window_height: int = 1024
    window_overlap: int = 256
    input_extensions: list[str] = field(
        default_factory=lambda: [".webp", ".png", ".jpg", ".jpeg"]
    )
    output_format: str = "webp"
    log_level: str = "INFO"
    log_file: str = "logs/latest.log"
    min_confidence: float = 0.5
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    inpainter: InpainterConfig = field(default_factory=InpainterConfig)

    @property
    def log_file_path(self) -> Path:
        """Log dosyasının mutlak yolu."""
        return PROJECT_ROOT / self.log_file


def load_config(path: str | Path | None = None) -> Config:
    """config.yaml dosyasını okur ve Config nesnesi döndürür."""
    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml bulunamadı: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    det_raw = raw.get("detector", {})
    detection_raw = raw.get("detection", {})
    ocr_raw = raw.get("ocr", {})
    trans_raw = raw.get("translator", {})
    inp_raw = raw.get("inpainter", {})

    return Config(
        window_height=raw.get("window_height", 1024),
        window_overlap=raw.get("window_overlap", 256),
        input_extensions=raw.get("input_extensions", [".webp", ".png", ".jpg", ".jpeg"]),
        output_format=raw.get("output_format", "webp"),
        log_level=raw.get("log_level", "INFO"),
        log_file=raw.get("log_file", "logs/latest.log"),
        min_confidence=raw.get("min_confidence", 0.5),
        detector=DetectorConfig(**det_raw),
        detection=DetectionConfig(**detection_raw),
        ocr=OCRConfig(**ocr_raw),
        translator=TranslatorConfig(**trans_raw),
        inpainter=InpainterConfig(**inp_raw),
    )
