"""Model Manager & Registry: Catalogs required AI models, verifies presence, and downloads missing models with resumable streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Callable, Optional, Sequence, Any
import urllib.request
import urllib.error


@dataclass(frozen=True)
class ModelSpec:
    """Specification of an AI model required by the translation pipeline."""

    id: str
    name: str
    filename: str
    relative_path: str
    url: str
    size_bytes: int
    category: str  # "detector", "ocr", "translation", "inpaint"
    description: str
    sha256: Optional[str] = None
    fallback_paths: list[str] = field(default_factory=list)

    def get_target_path(self, base_dir: Path) -> Path:
        return base_dir / self.relative_path


# Required Production Pipeline Models (EN -> TR)
PIPELINE_MODELS: list[ModelSpec] = [
    ModelSpec(
        id="ctd_onnx",
        name="ComicTextDetector ONNX",
        filename="comictextdetector.pt.onnx",
        relative_path="detectors/ctd/comictextdetector.pt.onnx",
        url="https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx",
        size_bytes=125_829_120,  # ~120 MB
        category="detector",
        description="Manga ve Webtoon diyalog baloncuklarını ve SFX metinlerini tespit eden derin öğrenme modeli.",
        sha256="1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f",
        fallback_paths=[
            r"models/detectors/ctd/comictextdetector.pt.onnx",
        ],
    ),
    ModelSpec(
        id="lama_large",
        name="LaMa Large Inpainting Checkpoint",
        filename="lama_large_512px.ckpt",
        relative_path="inpainting/lama_large_512px.ckpt",
        url="https://github.com/advimman/lama/raw/main/lama_large_512px.ckpt",
        size_bytes=208_666_624,  # ~199 MB
        category="inpaint",
        description="Baloncukların arkasındaki çizimleri ve dokuları temizleyip yeniden oluşturan inpainting modeli.",
        sha256=None,
        fallback_paths=[
            r"C:\AI\Models\LaMa\lama_large_512px.ckpt",
            r"models/inpainting/lama_large_512px.ckpt",
        ],
    ),
    ModelSpec(
        id="hy_mt2_gguf",
        name="Hy-MT2 7B Q8_0 GGUF Translator",
        filename="HY-MT2-7B-Q8_0.gguf",
        relative_path="translation/HY-MT2-7B-Q8_0.gguf",
        url="https://huggingface.co/Tencent/HY-MT2-7B-GGUF/resolve/main/HY-MT2-7B-Q8_0.gguf",
        size_bytes=7_696_000_000,  # ~7.6 GB
        category="translation",
        description="İngilizce -> Türkçe diyalog çevirisini yüksek bağlam tutarlılığıyla yapan Tencent Hy-MT2 modeli.",
        sha256=None,
        fallback_paths=[
            r"C:\AI\Models\HY-MT2-7B-Q8_0.gguf",
            r"models/translation/HY-MT2-7B-Q8_0.gguf",
        ],
    ),
]


class ModelManager:
    """Manages model directory configuration, existence checks, and resumable downloads."""

    CONFIG_FILE = Path.home() / ".webtoon_translator_config.json"

    def __init__(self, custom_base_dir: Optional[Path | str] = None) -> None:
        if custom_base_dir:
            self._base_dir = Path(custom_base_dir)
        else:
            self._base_dir = self._load_saved_model_dir()

    def get_model_dir(self) -> Path:
        return self._base_dir

    def set_model_dir(self, new_dir: Path | str) -> None:
        self._base_dir = Path(new_dir)
        self._save_model_dir(self._base_dir)

    def _load_saved_model_dir(self) -> Path:
        default_dir = Path(__file__).resolve().parent.parent.parent / "models"
        if self.CONFIG_FILE.exists():
            try:
                data = json.loads(self.CONFIG_FILE.read_text(encoding="utf-8"))
                if "model_dir" in data and Path(data["model_dir"]).exists():
                    return Path(data["model_dir"])
            except Exception:
                pass
        return default_dir

    def _save_model_dir(self, dir_path: Path) -> None:
        try:
            data = {}
            if self.CONFIG_FILE.exists():
                try:
                    data = json.loads(self.CONFIG_FILE.read_text(encoding="utf-8"))
                except Exception:
                    pass
            data["model_dir"] = str(dir_path.resolve())
            self.CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def get_all_models(self) -> list[ModelSpec]:
        return list(PIPELINE_MODELS)

    def find_model_path(self, spec: ModelSpec) -> Optional[Path]:
        """Returns the valid existing path for a model spec if present."""
        # 1. Check in configured base_dir
        target = spec.get_target_path(self._base_dir)
        if target.is_file() and target.stat().st_size > 1024:
            return target

        # 2. Check in fallback legacy paths
        for fb in spec.fallback_paths:
            fb_path = Path(fb)
            if fb_path.is_file() and fb_path.stat().st_size > 1024:
                return fb_path

        return None

    def get_missing_models(self) -> list[ModelSpec]:
        """Identifies any model specs that are not yet available on the system."""
        missing: list[ModelSpec] = []
        for spec in PIPELINE_MODELS:
            if self.find_model_path(spec) is None:
                missing.append(spec)
        return missing

    def download_model(
        self,
        spec: ModelSpec,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Path:
        """Downloads a model file with HTTP Range resume capability.
        
        Args:
            spec: The model to download.
            progress_callback: (downloaded_bytes, total_bytes, speed_bytes_per_sec, eta_seconds)
            cancel_check: Callable returning True if cancellation requested.
            
        Returns:
            Path to downloaded model.
        """
        dest_path = spec.get_target_path(self._base_dir)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = dest_path.with_suffix(dest_path.suffix + ".part")

        existing_size = 0
        if part_path.exists():
            existing_size = part_path.stat().st_size

        req = urllib.request.Request(spec.url)
        req.add_header("User-Agent", "WebtoonTranslator/2.0")
        if existing_size > 0:
            req.add_header("Range", f"bytes={existing_size}-")

        try:
            response = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            # If server doesn't support Range, start fresh
            if e.code in (416, 400):
                if part_path.exists():
                    part_path.unlink()
                existing_size = 0
                req = urllib.request.Request(spec.url)
                req.add_header("User-Agent", "WebtoonTranslator/2.0")
                response = urllib.request.urlopen(req, timeout=30)
            else:
                raise

        content_range = response.headers.get("Content-Range")
        content_length = response.headers.get("Content-Length")

        if content_range:
            total_size = int(content_range.split("/")[-1])
        elif content_length:
            total_size = existing_size + int(content_length)
        else:
            total_size = spec.size_bytes

        mode = "ab" if existing_size > 0 and response.status == 206 else "wb"
        if mode == "wb":
            existing_size = 0

        downloaded = existing_size
        chunk_size = 1024 * 512  # 512 KB chunks

        start_time = time.time()
        last_time = start_time
        last_downloaded = downloaded

        with open(part_path, mode) as f:
            while True:
                if cancel_check and cancel_check():
                    raise InterruptedError("Download cancelled by user")

                chunk = response.read(chunk_size)
                if not chunk:
                    break

                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                elapsed = now - last_time
                if elapsed >= 0.25:
                    speed = (downloaded - last_downloaded) / elapsed
                    remaining = total_size - downloaded
                    eta = remaining / speed if speed > 0 else 0
                    if progress_callback:
                        progress_callback(downloaded, total_size, speed, eta)
                    last_time = now
                    last_downloaded = downloaded

        # Final progress update
        if progress_callback:
            progress_callback(downloaded, total_size, 0.0, 0.0)

        # Atomic rename once completed
        if dest_path.exists():
            dest_path.unlink()
        part_path.rename(dest_path)
        return dest_path
