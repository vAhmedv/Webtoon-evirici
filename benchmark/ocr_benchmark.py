"""OCR benchmark orchestration.

Birden fazla OCR provider'ı aynı chapter/regions üzerinde çalıştırır ve sonuçları karşılaştırır.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from loguru import logger

from application.cancellation import CancellationToken, CancelledError
from benchmark.report import save_results
from core.config import Config, load_config
from core.detection import Region
from core.imaging.region_cropper import RegionCrop, RegionCropper
from providers.ocr.base import OCRProvider, OCRResult


class OCRBenchmarkResult:
    """Tek bir OCR provider'ın benchmark sonucu."""

    def __init__(
        self,
        provider_name: str,
        model_version: str,
        ocr_results: list[tuple[int, OCRResult]],
        load_time: float,
        inference_time: float,
        warnings: list[str],
    ) -> None:
        self.provider_name = provider_name
        self.model_version = model_version
        self.ocr_results = ocr_results
        self.load_time = load_time
        self.inference_time = inference_time
        self.warnings = warnings

    @property
    def total_regions(self) -> int:
        return len(self.ocr_results)

    @property
    def success_count(self) -> int:
        return sum(1 for _, r in self.ocr_results if r.text)

    @property
    def blank_count(self) -> int:
        return sum(1 for _, r in self.ocr_results if not r.text)

    @property
    def avg_confidence(self) -> float:
        if not self.ocr_results:
            return 0.0
        return sum(r.confidence for _, r in self.ocr_results) / len(self.ocr_results)

    @property
    def total_time(self) -> float:
        return self.load_time + self.inference_time


def run_ocr_benchmark(
    regions: Sequence[Region],
    cropper: RegionCropper,
    providers: Sequence[OCRProvider],
    progress_callback: callable | None = None,
    cancellation_token: CancellationToken | None = None,
) -> list[OCRBenchmarkResult]:
    """Birden fazla OCR provider'ı benchmark eder.

    Args:
        regions: OCR yapılacak Region listesi (global koordinatlı).
        cropper: RegionCropper instance.
        providers: Benchmark edilecek OCR provider'ları.
        progress_callback: İlerleme callback'i.
        cancellation_token: İptal belirteci.

    Returns:
        OCRBenchmarkResult listesi.
    """
    results: list[OCRBenchmarkResult] = []

    for idx, provider in enumerate(providers, start=1):
        if cancellation_token and cancellation_token.is_cancelled:
            raise CancelledError()

        provider_name = provider.name
        logger.info(f"Benchmarking OCR {idx}/{len(providers)}: {provider_name}")

        if progress_callback:
            progress_callback(
                type("ProgressEvent", (), {"stage": f"OCR {provider_name}", "current": idx, "total": len(providers)})()
            )

        load_start = time.time()
        try:
            provider.load()
        except Exception as e:
            logger.error(f"OCR provider {provider_name} yüklenemedi: {e}")
            results.append(
                OCRBenchmarkResult(
                    provider_name=provider_name,
                    model_version="unknown",
                    ocr_results=[],
                    load_time=time.time() - load_start,
                    inference_time=0.0,
                    warnings=[f"Load failed: {e}"],
                )
            )
            continue

        load_time = time.time() - load_start
        inference_start = time.time()

        ocr_results: list[tuple[int, OCRResult]] = []
        warnings: list[str] = []

        for region in regions:
            if cancellation_token and cancellation_token.is_cancelled:
                raise CancelledError()

            try:
                crop = cropper.crop_region(region)
                result = provider.recognize(crop.image, region_bbox=region.global_bbox)
                ocr_results.append((region.id, result))
                warnings.extend(result.warnings)
            except Exception as e:
                logger.error(f"OCR failed for region {region.id}: {e}")
                warnings.append(f"Region {region.id}: {e}")

        inference_time = time.time() - inference_start

        try:
            provider.unload()
        except Exception:
            pass

        results.append(
            OCRBenchmarkResult(
                provider_name=provider_name,
                model_version=getattr(provider, "version", "unknown"),
                ocr_results=ocr_results,
                load_time=load_time,
                inference_time=inference_time,
                warnings=warnings,
            )
        )

    if progress_callback:
        progress_callback(type("ProgressEvent", (), {"stage": "OCR benchmark complete", "current": len(providers), "total": len(providers)})())

    return results


def compute_ocr_metrics(
    results: Sequence[OCRBenchmarkResult],
    ground_truth: dict[int, str] | None = None,
) -> list[dict]:
    """OCR benchmark sonuçlarından metrikleri hesaplar."""
    metrics = []

    for res in results:
        total_conf = sum(r.confidence for _, r in res.ocr_results)
        avg_conf = total_conf / len(res.ocr_results) if res.ocr_results else 0.0

        metric = {
            "provider_name": res.provider_name,
            "model_version": res.model_version,
            "total_regions": res.total_regions,
            "success_count": res.success_count,
            "blank_count": res.blank_count,
            "avg_confidence": avg_conf,
            "load_time": res.load_time,
            "inference_time": res.inference_time,
            "total_time": res.total_time,
            "warnings": res.warnings,
        }

        if ground_truth:
            cer_values = []
            wer_values = []
            exact_matches = 0
            for region_id, ocr_result in res.ocr_results:
                gt = ground_truth.get(region_id, "")
                if gt:
                    pred = ocr_result.text
                    cer = _cer(pred, gt)
                    wer = _wer(pred, gt)
                    cer_values.append(cer)
                    wer_values.append(wer)
                    if pred.strip() == gt.strip():
                        exact_matches += 1

            metric["cer"] = sum(cer_values) / len(cer_values) if cer_values else None
            metric["wer"] = sum(wer_values) / len(wer_values) if wer_values else None
            metric["exact_match"] = exact_matches
            metric["exact_match_rate"] = exact_matches / len(ground_truth) if ground_truth else None

        metrics.append(metric)

    return metrics


def _cer(pred: str, gt: str) -> float:
    """Character Error Rate."""
    if not gt:
        return 0.0 if not pred else 1.0
    # Simple Levenshtein distance
    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n] / max(len(gt), 1)


def _wer(pred: str, gt: str) -> float:
    """Word Error Rate."""
    pred_words = pred.split()
    gt_words = gt.split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    m, n = len(pred_words), len(gt_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i - 1] == gt_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n] / max(len(gt_words), 1)
