"""Benchmark report testleri."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from benchmark.metrics import DetectorBenchmarkResult
from benchmark.report import generate_comparison_md, save_results


def test_generate_comparison_md() -> None:
    results = [
        DetectorBenchmarkResult(
            detector_name="A",
            model_version="v1",
            device="cuda",
            chapter_path="ch1",
            pages=10,
            windows=5,
            detections_before_merge=20,
            regions_after_merge=15,
            true_positives=12,
            false_positives=3,
            false_negatives=2,
            precision=0.8,
            recall=0.857,
            f1=0.828,
            boundary_success=1,
            boundary_total=1,
            no_text_fp=0,
            load_time=0.5,
            inference_time=2.0,
            avg_window_time=0.4,
            peak_vram_mb=1024.0,
        ),
    ]
    md = generate_comparison_md(results)
    assert "Detector Benchmark Report" in md
    assert "A" in md


def test_save_results_creates_files(tmp_path: Path) -> None:
    results = [
        DetectorBenchmarkResult(
            detector_name="A",
            model_version="v1",
            device="cuda",
            chapter_path="ch1",
            pages=10,
            windows=5,
            detections_before_merge=20,
            regions_after_merge=15,
            true_positives=12,
            false_positives=3,
            false_negatives=2,
            precision=0.8,
            recall=0.857,
            f1=0.828,
            boundary_success=1,
            boundary_total=1,
            no_text_fp=0,
            load_time=0.5,
            inference_time=2.0,
            avg_window_time=0.4,
            peak_vram_mb=1024.0,
        ),
    ]
    md_path = save_results(results, tmp_path / "bench")
    assert md_path.exists()
    assert (tmp_path / "bench" / "comparison.json").exists()
    assert (tmp_path / "bench" / "comparison.md").exists()
