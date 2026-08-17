"""Unit tests for HardwareBenchmarkWorker and Live Benchmark integration in BatchSettingsDialog."""

import os
from unittest.mock import MagicMock, patch
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from core.system.hardware_benchmark import HardwareBenchmarkWorker
from gui.dialogs.batch_settings_dialog import BatchSettingsDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_hardware_benchmark_cpu_fallback(qapp):
    with patch("torch.cuda.is_available", return_value=False):
        worker = HardwareBenchmarkWorker(vram_ceiling=0.95)
        completed_results = []
        worker.benchmark_completed.connect(lambda l, o, ll, v: completed_results.append((l, o, ll, v)))
        worker.run()

        assert len(completed_results) == 1
        assert completed_results[0] == (8, 8, 8, 0.0)


def test_hardware_benchmark_simulated_cuda_ceiling(qapp):
    mock_props = MagicMock()
    mock_props.total_memory = 12 * (1024 ** 3)  # 12 GB

    # Simulate memory stepping: 4, 8, 16, 24 safe, 32 hits >= 95% ceiling
    def mock_mem_allocated(device_idx):
        return 11.5 * (1024 ** 3)  # 11.5 / 12 = 95.8%

    steps_recorded = []
    completed_results = []

    worker = HardwareBenchmarkWorker(vram_ceiling=0.95)
    worker.step_updated.connect(lambda b, u, t, p: steps_recorded.append((b, p)))
    worker.benchmark_completed.connect(lambda l, o, ll, v: completed_results.append((l, o, ll, v)))

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=mock_props), \
         patch("torch.zeros", return_value=MagicMock()), \
         patch("torch.empty", return_value=MagicMock()), \
         patch("torch.cuda.synchronize"), \
         patch("torch.cuda.memory_allocated", side_effect=[
             2 * (1024**3),   # batch 8 -> 16%
             4 * (1024**3),   # batch 16 -> 33%
             8 * (1024**3),   # batch 24 -> 66%
             10 * (1024**3),  # batch 32 -> 83%
             11.6 * (1024**3) # batch 48 -> 96.6% (exceeds 95%)
         ]), \
         patch("torch.cuda.memory_reserved", return_value=0), \
         patch("torch.cuda.empty_cache"), \
         patch("time.sleep"):

        worker.run()

    assert len(steps_recorded) >= 4
    assert len(completed_results) == 1
    optimal_lama, optimal_ocr, optimal_llm, max_vram = completed_results[0]
    assert optimal_lama == 32
    assert optimal_ocr >= 32
    assert optimal_llm >= 8


def test_batch_settings_dialog_benchmark_integration(qapp):
    dialog = BatchSettingsDialog()
    assert dialog.btn_benchmark is not None
    assert dialog.bench_progress is not None

    # Test step update handler
    dialog._on_bench_step(batch=32, used_gb=8.5, tot_gb=12.0, pct=70.8)
    assert dialog.bench_progress.value() == int((32 / 256) * 100)
    assert "Batch 32" in dialog.lbl_bench_status.text()

    # Test completion handler auto-populating sliders
    dialog._on_bench_completed(optimal_lama=128, optimal_ocr=160, optimal_llm=32, max_vram_pct=92.5)
    assert dialog.bench_progress.value() == 100
    assert "128" in dialog.lbl_bench_status.text()
    assert dialog.slider_lama.value() == 128
    assert dialog.slider_ocr.value() == 160
    assert dialog.slider_llm.value() == 32

    dialog.close()
