"""Unit tests for ModelManager, ModelSpec, and ModelDownloadDialog."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from core.models.manager import ModelManager, ModelSpec, PIPELINE_MODELS
from gui.dialogs.model_download_dialog import ModelDownloadDialog, ModelDownloadWorker


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_model_spec_structure():
    assert len(PIPELINE_MODELS) >= 3
    spec = PIPELINE_MODELS[0]
    assert spec.id == "ctd_onnx"
    assert spec.filename == "comictextdetector.pt.onnx"
    assert spec.size_bytes > 0
    assert spec.category == "detector"


def test_model_manager_directory_config(tmp_path: Path):
    manager = ModelManager(custom_base_dir=tmp_path / "models")
    assert manager.get_model_dir() == tmp_path / "models"

    new_dir = tmp_path / "new_models"
    manager.set_model_dir(new_dir)
    assert manager.get_model_dir() == new_dir


def test_model_manager_missing_models(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    manager = ModelManager(custom_base_dir=models_dir)

    # All should be missing initially if fallbacks don't exist in tmp_path
    with patch.object(ModelManager, "find_model_path", return_value=None):
        missing = manager.get_missing_models()
        assert len(missing) == len(PIPELINE_MODELS)


def test_model_manager_download_mocked(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    manager = ModelManager(custom_base_dir=models_dir)

    spec = ModelSpec(
        id="test_model",
        name="Test Model",
        filename="test.bin",
        relative_path="test/test.bin",
        url="https://example.com/test.bin",
        size_bytes=1024,
        category="detector",
        description="Test",
    )

    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": "1024"}
    mock_resp.status = 200
    mock_resp.read.side_effect = [b"A" * 512, b"B" * 512, b""]

    progress_records = []
    def _cb(dl, tot, sp, eta):
        progress_records.append((dl, tot))

    with patch("urllib.request.urlopen", return_value=mock_resp):
        out_path = manager.download_model(spec, progress_callback=_cb)
        assert out_path.exists()
        assert out_path.stat().st_size == 1024
        assert len(progress_records) > 0


def test_model_download_worker_and_dialog(qapp, tmp_path: Path):
    manager = ModelManager(custom_base_dir=tmp_path / "models")
    spec = ModelSpec(
        id="test_model",
        name="Test Model",
        filename="test.bin",
        relative_path="test/test.bin",
        url="https://example.com/test.bin",
        size_bytes=1024,
        category="detector",
        description="Test description",
    )

    with patch.object(ModelManager, "download_model", return_value=tmp_path / "models" / "test.bin"):
        dialog = ModelDownloadDialog(manager, [spec])
        assert dialog is not None
        assert dialog.lbl_title.text() != ""
        
        # Wait for worker to finish and process queued signals
        if dialog._worker:
            dialog._worker.wait(2000)
            qapp.processEvents()
            assert dialog.was_successful is True
        dialog.close()
