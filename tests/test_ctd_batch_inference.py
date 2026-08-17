"""Unit tests for ComicTextDetector (CTD) GPU Tile Batching."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from core.detection import BBox, Detection, RegionType
from providers.detector.ctd import ComicTextDetector, _make_dynamic_batch_onnx


@pytest.fixture
def sample_tile_images() -> list[np.ndarray]:
    """Generates 4 synthetic tile images (1024x800) with simulated text blocks."""
    tiles = []
    for i in range(4):
        img = np.ones((1024, 800, 3), dtype=np.uint8) * 245
        # Add high contrast dark blocks with speech-bubble like margins
        y1, y2 = 100 + i * 80, 220 + i * 80
        x1, x2 = 150 + i * 40, 450 + i * 40
        img[y1:y2, x1:x2] = 20
        # Add second block
        img[600:700, 200:500] = 30
        tiles.append(img)
    return tiles


def test_dynamic_batch_surgery_onnx() -> None:
    """_make_dynamic_batch_onnx fonksiyonu ONNX modelini dinamik batch formatına dönüştürür."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    model_path = Path("models/detectors/ctd/comictextdetector.pt.onnx")
    if not model_path.exists():
        pytest.skip("CTD model file not present on disk")

    dynamic_bytes = _make_dynamic_batch_onnx(model_path)
    assert len(dynamic_bytes) > 0

    sess = ort.InferenceSession(dynamic_bytes, providers=["CPUExecutionProvider"])
    # Verify input shape is dynamic
    assert sess.get_inputs()[0].name == "images"

    # Test batch 1 and batch 3
    dummy_b1 = np.zeros((1, 3, 1024, 1024), dtype=np.float32)
    outs_b1 = sess.run(None, {"images": dummy_b1})
    assert len(outs_b1) == 3
    assert outs_b1[0].shape[0] == 1

    dummy_b3 = np.zeros((3, 3, 1024, 1024), dtype=np.float32)
    outs_b3 = sess.run(None, {"images": dummy_b3})
    assert outs_b3[0].shape[0] == 3
    assert outs_b3[1].shape[0] == 3
    assert outs_b3[2].shape[0] == 3


def test_preprocess_batch_shapes(sample_tile_images: list[np.ndarray]) -> None:
    """_preprocess_batch doğru [B, 3, 1024, 1024] tensör ve meta verileri üretir."""
    detector = ComicTextDetector(input_size=1024)
    batch_tensor, meta_list = detector._preprocess_batch(sample_tile_images)

    assert batch_tensor.shape == (4, 3, 1024, 1024)
    assert batch_tensor.dtype == np.float32
    assert batch_tensor.min() >= 0.0
    assert batch_tensor.max() <= 1.0

    assert len(meta_list) == 4
    for meta in meta_list:
        assert meta["im_w"] == 800
        assert meta["im_h"] == 1024
        assert meta["scale_x"] > 0
        assert meta["scale_y"] > 0


def test_ctd_batch_inference_consistency(sample_tile_images: list[np.ndarray]) -> None:
    """Tekli detect() ile toplu detect_batch() çıktılarının kutuları ve skorları birebir eşleşir."""
    model_path = Path("models/detectors/ctd/comictextdetector.pt.onnx")
    if not model_path.exists():
        pytest.skip("CTD model file not present on disk")

    detector = ComicTextDetector(tile_batch_size=4)
    detector.load()

    items = [(img, idx) for idx, img in enumerate(sample_tile_images)]

    # 1. Single sequential inference
    single_results = [detector.detect(img, wid) for img, wid in items]

    # 2. Batched GPU/CPU tile inference
    batch_results = detector.detect_batch(items)

    assert len(single_results) == len(batch_results) == 4

    for i in range(len(items)):
        single_dets = single_results[i]
        batch_dets = batch_results[i]
        assert len(single_dets) == len(batch_dets), f"Mismatch in detection count for tile {i}"

        for d_single, d_batch in zip(single_dets, batch_dets):
            assert d_single.bbox.x1 == pytest.approx(d_batch.bbox.x1, abs=1)
            assert d_single.bbox.y1 == pytest.approx(d_batch.bbox.y1, abs=1)
            assert d_single.bbox.x2 == pytest.approx(d_batch.bbox.x2, abs=1)
            assert d_single.bbox.y2 == pytest.approx(d_batch.bbox.y2, abs=1)
            assert d_single.confidence == pytest.approx(d_batch.confidence, abs=1e-3)
            assert d_single.source_window_id == d_batch.source_window_id

    detector.unload()


def test_ctd_detect_batch_blank_skipping() -> None:
    """detect_batch boş arka plan pencerelerini atlar ve indeks sırasını korur."""
    model_path = Path("models/detectors/ctd/comictextdetector.pt.onnx")
    if not model_path.exists():
        pytest.skip("CTD model file not present on disk")

    detector = ComicTextDetector()
    detector.load()

    blank_tile = np.ones((1024, 800, 3), dtype=np.uint8) * 255
    text_tile = np.ones((1024, 800, 3), dtype=np.uint8) * 245
    text_tile[100:300, 100:500] = 0

    items = [
        (blank_tile, 0),
        (text_tile, 1),
        (blank_tile, 2),
    ]

    results = detector.detect_batch(items)
    assert len(results) == 3
    assert len(results[0]) == 0  # Blank tile
    assert len(results[2]) == 0  # Blank tile
    # Non-blank tile should have detections
    assert len(results[1]) >= 0
    if results[1]:
        assert results[1][0].source_window_id == 1

    detector.unload()


def test_ctd_detect_batch_empty_items() -> None:
    """Boş liste verildiğinde detect_batch boş liste döndürür."""
    detector = ComicTextDetector()
    detector._loaded = True
    assert detector.detect_batch([]) == []


def test_ctd_detect_batch_adaptive_batcher_oom_fallback() -> None:
    """OOM durumunda ElasticAdaptiveBatcher batch boyutunu düşürerek işlemi kurtarır."""
    detector = ComicTextDetector(tile_batch_size=4)
    detector._loaded = True
    detector._ort_sessions = [MagicMock()]
    detector._net = None

    call_count = 0

    def mock_detect_batch_chunk(sub_jobs):
        nonlocal call_count
        call_count += 1
        # First call with batch size 4 triggers simulated OOM
        if len(sub_jobs) > 2 and call_count == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate...")
        # Subsequent smaller batches succeed
        return [
            [Detection(bbox=BBox(x1=10, y1=10, x2=50, y2=50), confidence=0.9, type=RegionType.UNKNOWN, source_window_id=wid)]
            for _, wid in sub_jobs
        ]

    detector._detect_batch_chunk = mock_detect_batch_chunk

    dummy_tiles = [
        (np.random.randint(0, 255, (1024, 800, 3), dtype=np.uint8), i)
        for i in range(4)
    ]

    results = detector.detect_batch(dummy_tiles)
    assert len(results) == 4
    for i, dets in enumerate(results):
        assert len(dets) == 1
        assert dets[0].source_window_id == i


def test_ctd_opencv_fallback_batch() -> None:
    """OpenCV DNN modunda detect_batch tekil pencereleri sırayla işler."""
    detector = ComicTextDetector()
    detector._loaded = True
    detector._ort_sessions = []
    detector._net = MagicMock()

    mock_dets = [Detection(bbox=BBox(x1=20, y1=20, x2=80, y2=80), confidence=0.85, type=RegionType.UNKNOWN, source_window_id=99)]
    detector._detect_single_array = MagicMock(return_value=mock_dets)

    dummy_tile = np.random.randint(0, 255, (1024, 800, 3), dtype=np.uint8)
    results = detector.detect_batch([(dummy_tile, 99)])

    assert len(results) == 1
    assert results[0] == mock_dets
    detector._detect_single_array.assert_called_once()


def test_ctd_16_batch_tile_inference() -> None:
    """Verifies that CTD can process a 16-tile batch using multi-threaded postprocessing."""
    model_path = Path("models/detectors/ctd/comictextdetector.pt.onnx")
    if not model_path.exists():
        pytest.skip("CTD model file not present on disk")

    detector = ComicTextDetector(tile_batch_size=16)
    detector.load()

    dummy_tiles = [
        (np.random.randint(0, 255, (1024, 800, 3), dtype=np.uint8), i)
        for i in range(16)
    ]

    results = detector.detect_batch(dummy_tiles)
    assert len(results) == 16
    for i, dets in enumerate(results):
        assert isinstance(dets, list)

    detector.unload()
