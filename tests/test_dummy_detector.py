"""DummyDetector testleri."""

from __future__ import annotations

from PIL import Image

import pytest

from providers.detector import DummyDetector
from core.detection import RegionType


def test_dummy_detector_requires_load() -> None:
    """Yükleme öncesi detect RuntimeError fırlatmalı."""
    det = DummyDetector()
    img = Image.new("RGB", (800, 1000))
    with pytest.raises(RuntimeError):
        det.detect(img, window_id=0)


def test_dummy_detector_deterministic_with_seed() -> None:
    """Aynı seed'le aynı çıktıyı üretmeli."""
    img = Image.new("RGB", (800, 1000))
    det1 = DummyDetector(seed=42)
    det1.load()
    out1 = det1.detect(img, window_id=0)

    det2 = DummyDetector(seed=42)
    det2.load()
    out2 = det2.detect(img, window_id=0)

    assert len(out1) == len(out2)
    for d1, d2 in zip(out1, out2):
        assert d1.bbox == d2.bbox
        assert d1.confidence == d2.confidence
        assert d1.type == d2.type
        assert d1.source_window_id == d2.source_window_id


def test_dummy_detector_returns_four_boxes() -> None:
    """Düzenli görüntüde 4 tespit döndürmeli."""
    img = Image.new("RGB", (800, 1000))
    det = DummyDetector(seed=0)
    det.load()
    results = det.detect(img, window_id=5)
    assert len(results) == 4


def test_dummy_detector_expected_types() -> None:
    """Tespit türleri beklenen sırada olmalı."""
    img = Image.new("RGB", (800, 1000))
    det = DummyDetector(seed=0)
    det.load()
    results = det.detect(img, window_id=0)
    types = [d.type for d in results]
    assert types[0] == RegionType.DIALOGUE
    assert types[1] == RegionType.NARRATION
    assert types[2] == RegionType.SFX
    assert types[3] == RegionType.WATERMARK