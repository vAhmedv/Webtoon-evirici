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
class DetectorConfig:
    """Detector ayarları (Phase 3'te kullanılacak)."""

    enabled: bool = False
    model: str | None = None


@dataclass(frozen=True)
class OCRConfig:
    """OCR ayarları (Phase 4'te kullanılacak)."""

    enabled: bool = False
    engine: str | None = None


@dataclass(frozen=True)
class TranslatorConfig:
    """Çeviri ayarları (Phase 5'te kullanılacak)."""

    enabled: bool = False
    provider: str | None = None
    qwen_model: str | None = None


@dataclass(frozen=True)
class InpainterConfig:
    """Inpainting ayarları (Phase 6'da kullanılacak)."""

    enabled: bool = False
    model: str | None = None


@dataclass(frozen=True)
class Config:
    """Uygulama geneli yapılandırma nesnesi."""

    window_height: int = 5000
    window_overlap: int = 1000
    input_extensions: list[str] = field(
        default_factory=lambda: [".webp", ".png", ".jpg", ".jpeg"]
    )
    output_format: str = "webp"
    log_level: str = "INFO"
    log_file: str = "logs/latest.log"
    min_confidence: float = 0.5
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    inpainter: InpainterConfig = field(default_factory=InpainterConfig)

    @property
    def log_file_path(self) -> Path:
        """Log dosyasının mutlak yolu."""
        return PROJECT_ROOT / self.log_file


def load_config(path: str | Path | None = None) -> Config:
    """config.yaml dosyasını okur ve Config nesnesi döndürür.

    Args:
        path: config.yaml yolu. Varsayılan olarak proje kökündeki config.yaml.

    Returns:
        Config: Yapılandırma nesnesi.

    Raises:
        FileNotFoundError: config.yaml bulunamazsa.
        yaml.YAMLError: YAML ayrıştırılamazsa.
    """
    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml bulunamadı: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return Config(
        window_height=int(raw.get("window_height", 5000)),
        window_overlap=int(raw.get("window_overlap", 1000)),
        input_extensions=[str(x) for x in raw.get("input_extensions", [".webp", ".png", ".jpg", ".jpeg"])],
        output_format=str(raw.get("output_format", "webp")),
        log_level=str(raw.get("log_level", "INFO")),
        log_file=str(raw.get("log_file", "logs/latest.log")),
        min_confidence=float(raw.get("min_confidence", 0.5)),
        detector=DetectorConfig(**raw.get("detector", {})),
        ocr=OCRConfig(**raw.get("ocr", {})),
        translator=TranslatorConfig(**raw.get("translator", {})),
        inpainter=InpainterConfig(**raw.get("inpainter", {})),
    )