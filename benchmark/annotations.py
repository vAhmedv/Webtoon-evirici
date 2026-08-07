"""Benchmark annotation formatı.

Ground truth JSON okuma/yazma.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.detection import BBox, Region, RegionStatus, RegionType


@dataclass(frozen=True)
class GroundTruthAnnotation:
    """Tek bir ground truth bölgesi."""

    bbox: BBox
    type: RegionType
    page_index: int
    window_id: int | None = None


def load_annotations(path: str | Path) -> list[GroundTruthAnnotation]:
    """JSON annotation dosyasını okur.

    Format:
    [
      {
        "bbox": [x1, y1, x2, y2],
        "type": "dialogue",
        "page_index": 0,
        "window_id": 0
      },
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: list[GroundTruthAnnotation] = []
    for item in data:
        bbox = BBox(
            x1=int(item["bbox"][0]),
            y1=int(item["bbox"][1]),
            x2=int(item["bbox"][2]),
            y2=int(item["bbox"][3]),
        )
        rtype = RegionType(item["type"])
        page_index = int(item["page_index"])
        window_id = item.get("window_id")
        results.append(
            GroundTruthAnnotation(
                bbox=bbox,
                type=rtype,
                page_index=page_index,
                window_id=int(window_id) if window_id is not None else None,
            )
        )
    return results


def save_annotations(annotations: list[GroundTruthAnnotation], path: str | Path) -> None:
    """Annotation listesini JSON olarak kaydeder."""
    data = []
    for ann in annotations:
        data.append(
            {
                "bbox": [ann.bbox.x1, ann.bbox.y1, ann.bbox.x2, ann.bbox.y2],
                "type": ann.type.value,
                "page_index": ann.page_index,
                "window_id": ann.window_id,
            }
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def annotations_to_regions(
    annotations: list[GroundTruthAnnotation],
) -> list[Region]:
    """Ground truth annotation'ları Region listesine çevirir."""
    regions: list[Region] = []
    for idx, ann in enumerate(annotations):
        regions.append(
            Region(
                id=idx,
                global_bbox=ann.bbox,
                type=ann.type,
                detection_confidence=1.0,
                source_window_ids=(ann.window_id,) if ann.window_id is not None else (),
                status=RegionStatus.AUTO,
            )
        )
    return regions
