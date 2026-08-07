"""Benchmark annotation testleri."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from benchmark.annotations import GroundTruthAnnotation, load_annotations, save_annotations, annotations_to_regions
from core.detection import BBox, RegionType


def test_save_and_load_annotations(tmp_path: Path) -> None:
    anns = [
        GroundTruthAnnotation(
            bbox=BBox(x1=10, y1=20, x2=100, y2=200),
            type=RegionType.DIALOGUE,
            page_index=0,
            window_id=0,
        ),
        GroundTruthAnnotation(
            bbox=BBox(x1=300, y1=400, x2=500, y2=600),
            type=RegionType.SFX,
            page_index=1,
            window_id=1,
        ),
    ]
    out = tmp_path / "gt.json"
    save_annotations(anns, out)
    assert out.exists()

    loaded = load_annotations(out)
    assert len(loaded) == 2
    assert loaded[0].type == RegionType.DIALOGUE
    assert loaded[0].bbox.x1 == 10
    assert loaded[1].type == RegionType.SFX


def test_annotations_to_regions() -> None:
    anns = [
        GroundTruthAnnotation(
            bbox=BBox(x1=0, y1=0, x2=10, y2=10),
            type=RegionType.NARRATION,
            page_index=0,
            window_id=0,
        ),
    ]
    regions = annotations_to_regions(anns)
    assert len(regions) == 1
    assert regions[0].type == RegionType.NARRATION
    assert regions[0].global_bbox.x1 == 0
