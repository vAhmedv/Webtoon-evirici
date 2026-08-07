"""Detector model indirme scripti."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "detectors"

MODELS = {
    "ctd": {
        "url": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt",
        "sha256": "1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb",
        "filename": "comictextdetector.pt",
    },
    "yolo8_comic": {
        "url": "https://huggingface.co/ogkalu/comic-text-segmenter-yolov8m/resolve/main/comic-text-segmenter.pt",
        "sha256": None,
        "filename": "comic-text-segmenter.pt",
    },
}


def verify_sha256(path: Path, expected: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def download_model(detector_name: str) -> None:
    if detector_name not in MODELS:
        print(f"Unknown detector: {detector_name}")
        print(f"Available: {list(MODELS.keys())}")
        sys.exit(1)

    info = MODELS[detector_name]
    dest_dir = MODELS_DIR / detector_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / info["filename"]

    if dest_path.exists():
        if info["sha256"] and verify_sha256(dest_path, info["sha256"]):
            print(f"Model already exists and verified: {dest_path}")
            return
        print(f"Existing model hash mismatch, re-downloading...")

    print(f"Downloading {detector_name} model...")
    print(f"URL: {info['url']}")
    print(f"Destination: {dest_path}")

    try:
        urlretrieve(info["url"], dest_path)
    except Exception as e:
        print(f"Download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        sys.exit(1)

    if info["sha256"]:
        if verify_sha256(dest_path, info["sha256"]):
            print("SHA256 verified successfully.")
        else:
            print("SHA256 verification FAILED!")
            dest_path.unlink()
            sys.exit(1)
    else:
        print("Download complete (no SHA256 verification available).")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/download_detector_models.py <detector_name>")
        print(f"Available detectors: {list(MODELS.keys())}")
        sys.exit(1)

    detector_name = sys.argv[1]
    download_model(detector_name)


if __name__ == "__main__":
    main()
