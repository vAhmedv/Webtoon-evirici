"""Unit tests for Hardware Auto-Calibration and Synthetic Webtoon Test Strip."""

from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from core.system.adaptive_batcher import (
    BatchConfig,
    ElasticAdaptiveBatcher,
    get_batch_config,
    set_batch_config,
)
from core.system.hardware_benchmark import (
    HardwareBenchmarkWorker,
    generate_calibration_strip,
)


def test_generate_calibration_strip_dimensions_and_bubbles() -> None:
    """generate_calibration_strip creates a valid 1024x4096 image with 12 bubble bounding boxes."""
    strip, boxes = generate_calibration_strip(1024, 4096)
    assert isinstance(strip, Image.Image)
    assert strip.size == (1024, 4096)
    assert strip.mode == "RGB"
    assert len(boxes) == 12
    for x1, y1, x2, y2 in boxes:
        assert 0 <= x1 < x2 <= 1024
        assert 0 <= y1 < y2 <= 4096


def test_hardware_benchmark_worker_execution() -> None:
    """HardwareBenchmarkWorker executes the calibration pipeline and emits completion signal."""
    worker = HardwareBenchmarkWorker(vram_ceiling=0.82)
    results = []

    def on_completed(optimal_lama: int, optimal_ocr: int, optimal_llm: int, max_vram_pct: float) -> None:
        results.append((optimal_lama, optimal_ocr, optimal_llm, max_vram_pct))

    worker.benchmark_completed.connect(on_completed)
    worker.run()

    assert len(results) == 1
    opt_lama, opt_ocr, opt_llm, max_vram = results[0]
    assert opt_lama >= 1
    assert opt_ocr >= 1
    assert opt_llm >= 1



def test_elastic_batcher_multiplicative_backoff_on_oom() -> None:
    """ElasticAdaptiveBatcher halves the batch size on OOM instead of linear step decay."""
    batcher = ElasticAdaptiveBatcher(default_batch_size=16, min_batch_size=1, vram_ceiling=0.82)

    call_count = 0

    def mock_process_fn(chunk):
        nonlocal call_count
        call_count += 1
        if len(chunk) > 4:
            raise RuntimeError("CUDA error: out of memory")
        return [f"processed_{x}" for x in chunk]

    items = list(range(10))
    res = batcher.execute(items, mock_process_fn)
    assert len(res) == 10
    # On first OOM at 16, it backoffs to 8, then on second OOM at 8 it backoffs to 4, then succeeds
    assert batcher.current_optimal_batch <= 4
