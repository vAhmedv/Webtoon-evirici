"""Tests for LaMa GPU Batching and OCR Pool Expansion."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from core.imaging.inpainter import Inpainter, DEFAULT_LAMA_CHECKPOINT
from core.imaging.lama import LaMaLargeInpainter
from providers.ocr.paddleocr import PaddleOCRProvider
from providers.ocr.paddleocr_vl import PaddleOCRVLOcrProvider


def test_lama_inpaint_batch_empty():
    lama = LaMaLargeInpainter(DEFAULT_LAMA_CHECKPOINT)
    assert lama.inpaint_batch([], []) == []


def test_lama_inpaint_batch_single_item():
    lama = LaMaLargeInpainter(DEFAULT_LAMA_CHECKPOINT)
    img = np.full((50, 50, 3), 200, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)

    # Empty mask returns copy without model call
    res = lama.inpaint_batch([img], [mask])
    assert len(res) == 1
    assert res[0].shape == (50, 50, 3)


def test_lama_inpaint_batch_mocked_forward():
    lama = LaMaLargeInpainter(DEFAULT_LAMA_CHECKPOINT)
    lama._loaded = True
    lama._model = MagicMock()
    lama._torch = MagicMock()
    lama._use_bf16 = False

    img1 = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[20:80, 20:80] = 255

    img2 = np.full((80, 120, 3), 200, dtype=np.uint8)
    mask2 = np.zeros((80, 120), dtype=np.uint8)
    mask2[10:50, 10:50] = 255

    # Mock single inpaint fallback or direct execution
    with patch.object(lama, "inpaint", side_effect=lambda im, mk: im.copy()) as mock_single:
        results = lama.inpaint_batch([img1, img2], [mask1, mask2], batch_size=2)
        assert len(results) == 2
        assert results[0].shape == (100, 100, 3)
        assert results[1].shape == (80, 120, 3)


def test_inpainter_inpaint_batch_interface():
    inpainter = Inpainter(lama_checkpoint=DEFAULT_LAMA_CHECKPOINT)
    img = np.full((60, 60, 3), 255, dtype=np.uint8)
    mask = np.zeros((60, 60), dtype=np.uint8)

    with patch.object(inpainter.lama, "inpaint_batch", return_value=[img]) as mock_batch:
        res = inpainter.inpaint_batch([img], [mask], batch_size=4)
        assert len(res) == 1
        mock_batch.assert_called_once_with([img], [mask], batch_size=4)


def test_ocr_provider_expanded_defaults():
    p_ocr = PaddleOCRProvider()
    assert "max_workers" in p_ocr.recognize_batch.__code__.co_varnames

    vl_ocr = PaddleOCRVLOcrProvider()
    assert "batch_size" in vl_ocr.recognize_batch.__code__.co_varnames
