"""Tests for CTD ONNXRuntime Execution Provider configuration and fallback."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from providers.detector.ctd import ComicTextDetector


def test_ctd_provider_resolution_cpu():
    detector = ComicTextDetector()
    with patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]), \
         patch("onnxruntime.InferenceSession") as mock_sess:
        mock_instance = MagicMock()
        mock_instance.get_inputs.return_value = [MagicMock(name="images", type="tensor(float)")]
        mock_instance.get_outputs.return_value = [MagicMock(name="blk"), MagicMock(name="seg"), MagicMock(name="det")]
        mock_instance.get_providers.return_value = ["CPUExecutionProvider"]
        mock_sess.return_value = mock_instance

        detector.load()
        assert detector.is_loaded
        assert detector.device == "cpu"
        assert len(detector._ort_sessions) == 3


def test_ctd_provider_resolution_cuda():
    detector = ComicTextDetector()
    with patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]), \
         patch("onnxruntime.InferenceSession") as mock_sess:
        mock_instance = MagicMock()
        mock_instance.get_inputs.return_value = [MagicMock(name="images", type="tensor(float)")]
        mock_instance.get_outputs.return_value = [MagicMock(name="blk"), MagicMock(name="seg"), MagicMock(name="det")]
        mock_instance.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        mock_sess.return_value = mock_instance

        detector.load()
        assert detector.is_loaded
        assert detector.device == "cuda"
        assert len(detector._ort_sessions) == 1


def test_ctd_fallback_to_opencv_dnn_on_ort_error():
    detector = ComicTextDetector()
    with patch("onnxruntime.InferenceSession", side_effect=RuntimeError("ORT initialization failed")), \
         patch("cv2.dnn.readNetFromONNX") as mock_dnn:
        mock_net = MagicMock()
        mock_net.getUnconnectedOutLayersNames.return_value = ["blk", "seg", "det"]
        mock_dnn.return_value = mock_net

        detector.load()
        assert detector.is_loaded
        assert detector._net is not None
        assert len(detector._ort_sessions) == 0


def test_ctd_forward_raw_type_adaptation():
    detector = ComicTextDetector()
    detector._loaded = True
    detector._ort_input_name = "images"
    detector._ort_input_type = "tensor(float)"
    detector._ort_output_names = ["blk", "seg", "det"]

    mock_sess = MagicMock()
    mock_sess.run.return_value = [np.zeros((1, 10, 7)), np.zeros((1, 1, 1024, 1024)), np.zeros((1, 2, 1024, 1024))]
    detector._ort_sessions = [mock_sess]

    input_data = np.zeros((1, 3, 1024, 1024), dtype=np.float32)
    outputs, out_dict = detector._forward_raw(input_data)
    assert len(outputs) == 3
    assert "blk" in out_dict
    mock_sess.run.assert_called_once()
