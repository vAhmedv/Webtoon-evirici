"""Benchmark metrikleri.

IoU eşleştirme, precision/recall/F1, TP/FP/FN hesaplama.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from core.detection import BBox, Region, RegionStatus, RegionType


@dataclass(frozen=True)
class DetectionMatch:
    """Tek bir ground truth ve detection eşleşmesi."""

    gt_index: int
    det_index: int
    iou: float
    matched: bool


@dataclass(frozen=True)
class WindowMetrics:
    """Tek bir window için metrikler."""

    window_id: int
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    matches: list[DetectionMatch] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    boundary_crossing_count: int = 0
    boundary_crossing_success: int = 0
    no_text_false_positives: int = 0


@dataclass(frozen=True)
class DetectorBenchmarkResult:
    """Tek bir detector'ın tüm chapter üzerindeki benchmark sonucu."""

    detector_name: str
    model_version: str
    device: str
    chapter_path: str
    pages: int
    windows: int
    detections_before_merge: int
    regions_after_merge: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    boundary_success: int
    boundary_total: int
    no_text_fp: int
    load_time: float
    inference_time: float
    avg_window_time: float
    peak_vram_mb: float | None
    warnings: list[str] = field(default_factory=list)
    window_metrics: list[WindowMetrics] = field(default_factory=list)


def compute_iou(a: BBox, b: BBox) -> float:
    return a.iou(b)


def match_detections(
    gt_regions: Sequence[Region],
    det_regions: Sequence[Region],
    iou_threshold: float = 0.5,
) -> tuple[list[DetectionMatch], int, int, int]:
    """Ground truth ve detection region'larını eşleştirir.

    Args:
        gt_regions: Ground truth region listesi.
        det_regions: Detection region listesi.
        iou_threshold: Eşleşme için minimum IoU.

    Returns:
        (matches, tp, fp, fn) tuple.
    """
    matches: list[DetectionMatch] = []
    used_gt: set[int] = set()
    used_det: set[int] = set()

    for i, gt in enumerate(gt_regions):
        for j, det in enumerate(det_regions):
            if j in used_det:
                continue
            iou = compute_iou(gt.global_bbox, det.global_bbox)
            if iou >= iou_threshold:
                matches.append(DetectionMatch(gt_index=i, det_index=j, iou=iou, matched=True))
                used_gt.add(i)
                used_det.add(j)
                break

    tp = len(used_gt)
    fp = len(det_regions) - len(used_det)
    fn = len(gt_regions) - len(used_gt)

    return matches, tp, fp, fn


def compute_metrics(
    gt_regions: Sequence[Region],
    det_regions: Sequence[Region],
    iou_threshold: float = 0.5,
) -> WindowMetrics:
    """Tek bir window için precision/recall/F1 hesaplar."""
    matches, tp, fp, fn = match_detections(gt_regions, det_regions, iou_threshold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return WindowMetrics(
        window_id=0,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        matches=matches,
        precision=precision,
        recall=recall,
        f1=f1,
    )
