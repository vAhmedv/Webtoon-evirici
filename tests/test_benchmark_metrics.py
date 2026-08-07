"""Benchmark metrikleri testleri."""

from __future__ import annotations

import pytest

from benchmark.metrics import DetectionMatch, WindowMetrics, compute_iou, compute_metrics, match_detections
from core.detection import BBox, Region, RegionStatus, RegionType


def test_compute_iou() -> None:
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    b = BBox(x1=5, y1=5, x2=15, y2=15)
    iou = compute_iou(a, b)
    assert 0.0 < iou < 1.0


def test_match_detections_perfect() -> None:
    gt = [Region(id=0, global_bbox=BBox(x1=0, y1=0, x2=10, y2=10), type=RegionType.DIALOGUE, detection_confidence=1.0, source_window_ids=(0,))]
    det = [Region(id=0, global_bbox=BBox(x1=0, y1=0, x2=10, y2=10), type=RegionType.DIALOGUE, detection_confidence=0.9, source_window_ids=(0,))]
    matches, tp, fp, fn = match_detections(gt, det, iou_threshold=0.5)
    assert tp == 1
    assert fp == 0
    assert fn == 0
    assert len(matches) == 1


def test_match_detections_false_positive() -> None:
    gt = [Region(id=0, global_bbox=BBox(x1=0, y1=0, x2=10, y2=10), type=RegionType.DIALOGUE, detection_confidence=1.0, source_window_ids=(0,))]
    det = [Region(id=0, global_bbox=BBox(x1=100, y1=100, x2=110, y2=110), type=RegionType.DIALOGUE, detection_confidence=0.9, source_window_ids=(0,))]
    matches, tp, fp, fn = match_detections(gt, det, iou_threshold=0.5)
    assert tp == 0
    assert fp == 1
    assert fn == 1


def test_compute_metrics_values() -> None:
    gt = [Region(id=0, global_bbox=BBox(x1=0, y1=0, x2=10, y2=10), type=RegionType.DIALOGUE, detection_confidence=1.0, source_window_ids=(0,))]
    det = [Region(id=0, global_bbox=BBox(x1=0, y1=0, x2=10, y2=10), type=RegionType.DIALOGUE, detection_confidence=0.9, source_window_ids=(0,))]
    m = compute_metrics(gt, det, iou_threshold=0.5)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_compute_metrics_empty() -> None:
    m = compute_metrics([], [], iou_threshold=0.5)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0
