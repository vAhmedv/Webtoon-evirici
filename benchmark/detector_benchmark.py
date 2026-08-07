"""Detector benchmark orchestration.

Birden fazla detector'ı aynı chapter üzerinde çalıştırır ve sonuçları karşılaştırır.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from loguru import logger

from application.cancellation import CancellationToken, CancelledError
from application.chapter_analyzer import AnalysisResult, ChapterAnalyzer
from application.progress import ProgressEvent
from benchmark.metrics import DetectorBenchmarkResult, WindowMetrics, compute_metrics, match_detections
from benchmark.report import save_results
from core.config import Config, load_config
from core.detection import Region, RegionType
from providers.detector.base import DetectorProvider


ProgressCallback = callable[[ProgressEvent], None]


class BenchmarkResult:
    """Tek bir detector'ın benchmark sonucu."""

    def __init__(
        self,
        detector_name: str,
        model_version: str,
        device: str,
        analysis: AnalysisResult,
        load_time: float,
        inference_time: float,
        peak_vram_mb: float | None,
        warnings: list[str],
    ) -> None:
        self.detector_name = detector_name
        self.model_version = model_version
        self.device = device
        self.analysis = analysis
        self.load_time = load_time
        self.inference_time = inference_time
        self.peak_vram_mb = peak_vram_mb
        self.warnings = warnings


def run_benchmark(
    chapter_path: str | Path,
    output_path: str | Path,
    detectors: Sequence[DetectorProvider],
    config: Config | None = None,
    annotations_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
    cancellation_token: CancellationToken | None = None,
) -> list[BenchmarkResult]:
    """Birden fazla detector'ı benchmark eder.

    Args:
        chapter_path: Bölüm klasörü.
        output_path: Çıktı klasörü.
        detectors: Benchmark edilecek detector'lar.
        config: Yapılandırma.
        annotations_path: Ground truth annotation JSON yolu (opsiyonel).
        progress_callback: İlerleme callback'i.
        cancellation_token: İptal belirteci.

    Returns:
        BenchmarkResult listesi.
    """
    cfg = config if config is not None else load_config()
    analyzer = ChapterAnalyzer(cfg)
    results: list[BenchmarkResult] = []

    annotations = None
    if annotations_path is not None and Path(annotations_path).exists():
        from benchmark.annotations import load_annotations, annotations_to_regions
        raw_anns = load_annotations(annotations_path)
        annotations = annotations_to_regions(raw_anns)

    for idx, detector in enumerate(detectors, start=1):
        if cancellation_token and cancellation_token.is_cancelled:
            raise CancelledError()

        detector_name = detector.name
        logger.info(f"Benchmarking detector {idx}/{len(detectors)}: {detector_name}")

        if progress_callback:
            progress_callback(
                ProgressEvent(
                    stage=f"Benchmarking {detector_name}",
                    current=idx,
                    total=len(detectors),
                )
            )

        load_start = time.time()
        try:
            detector.load()
        except Exception as e:
            logger.error(f"Detector {detector_name} yüklenemedi: {e}")
            results.append(
                BenchmarkResult(
                    detector_name=detector_name,
                    model_version="unknown",
                    device="unknown",
                    analysis=AnalysisResult([], [], [], 0.0),
                    load_time=time.time() - load_start,
                    inference_time=0.0,
                    peak_vram_mb=None,
                    warnings=[f"Load failed: {e}"],
                )
            )
            continue

        load_time = time.time() - load_start
        inference_start = time.time()

        peak_vram_mb = _measure_peak_vram()

        analysis = analyzer.analyze(
            chapter_path=chapter_path,
            output_path=output_path / detector_name,
            detector=detector,
            progress_callback=None,
            cancellation_token=cancellation_token,
        )

        inference_time = time.time() - inference_start
        peak_vram_mb = _measure_peak_vram()

        try:
            detector.unload()
        except Exception:
            pass

        warnings = _collect_warnings(analysis)

        results.append(
            BenchmarkResult(
                detector_name=detector_name,
                model_version=getattr(detector, "version", "unknown"),
                device=getattr(detector, "device", "unknown"),
                analysis=analysis,
                load_time=load_time,
                inference_time=inference_time,
                peak_vram_mb=peak_vram_mb,
                warnings=warnings,
            )

    if progress_callback:
        progress_callback(ProgressEvent(stage="Benchmark complete", current=len(detectors), total=len(detectors)))

    return results


def compute_benchmark_metrics(
    results: Sequence[BenchmarkResult],
    annotations: Sequence[Region] | None = None,
) -> list[DetectorBenchmarkResult]:
    """Benchmark sonuçlarından metrikleri hesaplar."""
    benchmark_metrics: list[DetectorBenchmarkResult] = []

    for res in results:
        analysis = res.analysis
        det_regions = analysis.regions

        tp = fp = fn = 0
        boundary_success = 0
        boundary_total = 0
        no_text_fp = 0
        window_metrics: list[WindowMetrics] = []

        if annotations is not None and len(annotations) > 0:
            _, tp, fp, fn = match_detections(annotations, det_regions)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        benchmark_metrics.append(
            DetectorBenchmarkResult(
                detector_name=res.detector_name,
                model_version=res.model_version,
                device=res.device,
                chapter_path=res.analysis.pages[0].path.parent.name if res.analysis.pages else "unknown",
                pages=len(analysis.pages),
                windows=len(analysis.windows),
                detections_before_merge=0,
                regions_after_merge=len(det_regions),
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                boundary_success=boundary_success,
                boundary_total=boundary_total,
                no_text_fp=no_text_fp,
                load_time=res.load_time,
                inference_time=res.inference_time,
                avg_window_time=res.inference_time / len(analysis.windows) if analysis.windows else 0.0,
                peak_vram_mb=res.peak_vram_mb,
                warnings=res.warnings,
                window_metrics=window_metrics,
            )
        )

    return benchmark_metrics


def _measure_peak_vram() -> float | None:
    """Peak VRAM kullanımını ölçer (MB)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return None


def _collect_warnings(result: AnalysisResult) -> list[str]:
    """AnalysisResult uyarılarını topla."""
    return list(result.warnings)
