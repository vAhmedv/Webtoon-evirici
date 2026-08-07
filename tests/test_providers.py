"""Provider integration testleri."""

from __future__ import annotations

import pytest

from providers.detector.dummy import DummyDetector
from providers.detector.ctd import ComicTextDetector
from providers.detector.yolo8_comic import Yolo8ComicTextDetector
from core.detection import RegionType


def test_dummy_detector_lifecycle() -> None:
    from PIL import Image
    det = DummyDetector(seed=42)
    det.load()
    assert det.is_loaded is True
    img = Image.new("RGB", (100, 100))
    detections = det.detect(img, 0)
    assert len(detections) == 4
    det.unload()
    assert det.is_loaded is False


def test_ctd_provider_missing_model() -> None:
    det = ComicTextDetector("/nonexistent/path")
    with pytest.raises(FileNotFoundError):
        det.load()


def test_yolo8_provider_missing_model() -> None:
    det = Yolo8ComicTextDetector("/nonexistent/path")
    with pytest.raises(FileNotFoundError):
        det.load()


def test_ctd_provider_name() -> None:
    det = ComicTextDetector("/nonexistent/path")
    assert det.name == "ComicTextDetector"


def test_yolo8_provider_name() -> None:
    det = Yolo8ComicTextDetector("/nonexistent/path")
    assert det.name == "YOLOv8 Comic Text Segmenter"
