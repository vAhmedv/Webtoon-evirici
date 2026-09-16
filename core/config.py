"""Yapılandırma yükleyici.

config.yaml dosyasını okur ve tip güvenli bir Config nesnesi döndürür.
Hard-code edilmiş parametre yoktur; her şey config.yaml'dan gelir.
"""

from __future__ import annotations

import os
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
    provider: str = "ComicTextDetector"
    # Opsiyonel skor eşiği (None = sağlayıcı varsayılanı korunur: CTD 0.4).
    # Ayarlanırsa analyzer/provider threshold'u buradan yazar.
    threshold: float | None = None


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
    primary_model: str = "PP-OCRv6_medium_rec"
    verifier_provider: str = "PaddleOCR-VL-1.6"
    qwen_model_path: str | None = None
    qwen_mmproj_path: str | None = None
    qwen_server_path: str | None = None
    qwen_server_url: str = "http://127.0.0.1:8086"
    qwen_server_port: int = 8086


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
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-flash-latest"


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

    # Legacy OCR provider adları (registry anahtarı yerine model adı yazılmış
    # eski config'ler) kanonik isme çevrilir. Bkz. OCRRegistry.LEGACY_ALIASES.
    _legacy_ocr_aliases = {
        "PaddleOCR-PP-OCRv6_medium_rec": "PaddleOCR-PP-OCRv6",
        "PaddleOCR-PP-OCRv6-server": "PaddleOCR-PP-OCRv6",
        "en_PP-OCRv5_mobile_rec": "PaddleOCR English v5",
        "PaddleOCR-VL-1.5": "PaddleOCR-VL-1.6",
        "PaddleOCR-VL": "PaddleOCR-VL-1.6",
    }
    if isinstance(ocr_raw, dict) and ocr_raw.get("provider") in _legacy_ocr_aliases:
        ocr_raw["provider"] = _legacy_ocr_aliases[ocr_raw["provider"]]
    if isinstance(ocr_raw, dict) and ocr_raw.get("verifier_provider") in _legacy_ocr_aliases:
        ocr_raw["verifier_provider"] = _legacy_ocr_aliases[ocr_raw["verifier_provider"]]

    # Secret hijyeni: API key config.yaml'a yazılmaz; yaml null ise env'den
    # (GEMINI_API_KEY / GOOGLE_API_KEY) çözülür. Bkz. ROADMAP Faz 0.1.
    _gemini_key = (
        trans_raw.get("gemini_api_key")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    trans_raw = dict(trans_raw)
    trans_raw["gemini_api_key"] = _gemini_key

    # Auto-upgrade deprecated Gemini model names to the active 2026 LTS alias
    deprecated_models = {"gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash-lite"}
    if trans_raw.get("gemini_model") in deprecated_models or not trans_raw.get("gemini_model"):
        trans_raw["gemini_model"] = "gemini-flash-latest"

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


def validate_config(cfg: Config) -> list[str]:
    """Config tutarlılığını kontrol eder, sorun listesi döndürür (boşsa OK).

    Ağır provider import'u yapmaz; yalnızca bilinen registry anahtarlarına
    karşı isim kontrolü + sayısal aralık kontrolü yapar.
    """
    problems: list[str] = []
    if cfg.window_height <= 0:
        problems.append(f"window_height must be >0 (got {cfg.window_height})")
    if not 0 <= cfg.window_overlap < cfg.window_height:
        problems.append(
            f"window_overlap must be in [0, window_height) "
            f"(got {cfg.window_overlap}/{cfg.window_height})"
        )
    if not 0.0 <= cfg.min_confidence <= 1.0:
        problems.append(f"min_confidence must be in [0,1] (got {cfg.min_confidence})")

    known_detectors = {"ComicTextDetector", "DummyDetector"}
    if cfg.detector.provider not in known_detectors:
        problems.append(
            f"Unknown detector.provider {cfg.detector.provider!r}. "
            f"Known: {sorted(known_detectors)}"
        )
    known_ocr = {
        "PaddleOCR-PP-OCRv6",
        "PaddleOCR English v5",
        "PaddleOCR-VL-1.6",
        "RapidOCR-ONNX",
    }
    if cfg.ocr.provider is not None and cfg.ocr.provider not in known_ocr:
        problems.append(
            f"Unknown ocr.provider {cfg.ocr.provider!r}. Known: {sorted(known_ocr)}"
        )
    if cfg.ocr.verifier_provider not in known_ocr:
        problems.append(
            f"Unknown ocr.verifier_provider {cfg.ocr.verifier_provider!r}. "
            f"Known: {sorted(known_ocr)}"
        )
    # Üretim port çakışma kontrolü (Hy-MT2 vs Qwen-repair aynı portta olmamalı)
    try:
        from urllib.parse import urlparse as _urlparse

        def _port(url: str | None, default: int) -> int:
            try:
                return _urlparse(url or "").port or default
            except Exception:
                return default

        hy_port = _port(cfg.translator.server_url, 8085)
        qwen_port = _port(cfg.ocr.qwen_server_url, cfg.ocr.qwen_server_port)
        if hy_port and qwen_port and hy_port == qwen_port:
            problems.append(
                f"translator.server_url and ocr.qwen_server_url share port {hy_port}; "
                f"parallel llama-servers would collide"
            )
        if qwen_port != cfg.ocr.qwen_server_port:
            problems.append(
                f"ocr.qwen_server_url port ({qwen_port}) != ocr.qwen_server_port "
                f"({cfg.ocr.qwen_server_port})"
            )
    except Exception:
        pass
    return problems


def update_gemini_api_key(
    api_key: str,
    model_name: str = "gemini-flash-latest",
    path: str | Path | None = None,
) -> None:
    """Updates the Gemini API key and model.

    Secret hijyeni (ROADMAP Faz 0.1): key config.yaml'a ASLA yazılmaz —
    yalnızca process env (GEMINI_API_KEY) + model adı yaml'a yazılır.
    Kalıcı kullanıcı env değişkeni işletim sisteminden ayarlanmalıdır.
    """

    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    raw: dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    if "translator" not in raw or not isinstance(raw["translator"], dict):
        raw["translator"] = {}

    raw["translator"]["gemini_api_key"] = None
    raw["translator"]["gemini_model"] = model_name

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, default_flow_style=False, allow_unicode=True)

    if api_key and api_key.strip():
        os.environ["GEMINI_API_KEY"] = api_key.strip()
    elif "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
