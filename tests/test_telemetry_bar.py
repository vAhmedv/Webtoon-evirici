"""Unit tests for TelemetryStatusBar and SystemTelemetry."""

import os
from unittest.mock import MagicMock, patch
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from gui.components.telemetry_bar import EngineBadge, SystemTelemetry, TelemetryStatusBar


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_system_telemetry_cpu_fallback():
    with patch("torch.cuda.is_available", return_value=False):
        gpu_info = SystemTelemetry.get_gpu_telemetry()
        assert gpu_info["has_gpu"] is False

    ram_info = SystemTelemetry.get_ram_telemetry()
    assert "percent" in ram_info


def test_system_telemetry_cuda_mock():
    mock_props = MagicMock()
    mock_props.total_memory = 12 * (1024 ** 3)

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4070"), \
         patch("torch.cuda.get_device_properties", return_value=mock_props), \
         patch("torch.cuda.memory_reserved", return_value=4 * (1024 ** 3)), \
         patch("torch.cuda.memory_allocated", return_value=3 * (1024 ** 3)):
        gpu_info = SystemTelemetry.get_gpu_telemetry()
        assert gpu_info["has_gpu"] is True
        assert gpu_info["device_name"] == "NVIDIA RTX 4070"
        assert round(gpu_info["total_gb"]) == 12
        assert round(gpu_info["reserved_gb"]) == 4


def test_engine_badge_active_state(qapp):
    badge = EngineBadge("CTD", tooltip="Test Detector")
    assert badge._is_active is False
    badge.set_active(True)
    assert badge._is_active is True
    badge.set_active(False)
    assert badge._is_active is False


def test_telemetry_status_bar_ui(qapp):
    bar = TelemetryStatusBar()
    assert bar is not None
    assert "DETECT" in bar.engine_badges
    assert "OCR" in bar.engine_badges
    assert "TRANSLATE" in bar.engine_badges

    # Test status message update
    bar.set_status("Analyzing Chapter 1...", is_busy=True)
    assert bar.status_msg.text() == "Analyzing Chapter 1..."

    # Test stage active badge highlighting
    bar.set_active_stage("OCR")
    assert bar.badge_ocr._is_active is True
    assert bar.badge_ctd._is_active is False

    bar.set_active_stage("TRANSLATE")
    assert bar.badge_trans._is_active is True
    assert bar.badge_ocr._is_active is False

    bar.reset_badges()
    assert bar.badge_trans._is_active is False

    # Stop timer on test teardown
    bar.timer.stop()
