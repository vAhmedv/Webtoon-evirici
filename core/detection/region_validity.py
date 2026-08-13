"""Conservative CTD region validation before expensive OCR repair stages.

The primary OCR pass is intentionally part of the evidence: a readable crop must
not be discarded merely because one of CTD's auxiliary masks is weak.  The gate
only rejects combinations that independently describe a strong non-text case.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from core.detection.bbox import BBox
from core.detection.detection import Region


Polygon = list[list[float]]


@dataclass(frozen=True)
class RegionValidityDecision:
    """Result of the pre-repair CTD evidence gate."""

    is_valid: bool
    reason: str
    recovered_bbox: BBox | None
    evidence: dict[str, object]


def evaluate_region_validity(region: Region, primary_text: str | None) -> RegionValidityDecision:
    """Validate a CTD crop without relying on an OCR-confidence threshold.

    A recoverable bbox is returned when retained CTD polygons extend outside the
    canonical bbox.  Callers should recrop/re-run primary OCR, then evaluate the
    recovered region again before invoking a verifier or visual repair model.
    """

    bbox = region.global_bbox
    metadata = region.metadata if isinstance(region.metadata, dict) else {}
    lines = _polygons(metadata.get("line_polygons"))
    segments = _polygons(metadata.get("segmentation_polygons"))
    text = (primary_text or "").strip()
    alnum_count = sum(char.isalnum() for char in text)

    bbox_area = float(max(1, bbox.area))
    line_areas = [_polygon_area(polygon) for polygon in lines]
    segment_area = sum(_polygon_area(polygon) for polygon in segments)
    line_area_ratio = sum(line_areas) / bbox_area
    segment_area_ratio = segment_area / bbox_area
    line_aspects = [_polygon_aspect(polygon) for polygon in lines]
    boundary_touches = _boundary_touches(segments, bbox)
    recovered_bbox = _recover_bbox(region, lines, segments, alnum_count >= 2)

    evidence: dict[str, object] = {
        "primary_alnum_count": alnum_count,
        "line_polygon_count": len(lines),
        "segmentation_polygon_count": len(segments),
        "line_area_ratio": round(line_area_ratio, 4),
        "segmentation_area_ratio": round(segment_area_ratio, 4),
        "segmentation_boundary_touches": boundary_touches,
        "max_line_aspect": round(max(line_aspects, default=0.0), 4),
    }

    if recovered_bbox is not None:
        return RegionValidityDecision(True, "ctd_geometry_recovery", recovered_bbox, evidence)

    # A readable multi-character primary result is strong text evidence.  It is
    # overridden only by an independently strong tiny/artwork geometry pattern.
    tiny_sparse = min(bbox.width, bbox.height) < 20 and segment_area_ratio < 0.20
    if tiny_sparse:
        return RegionValidityDecision(False, "tiny_sparse_text_geometry", None, evidence)

    largest_line_ratio = max(line_areas, default=0.0) / bbox_area
    artwork_like = (
        bbox.width >= 40
        and bbox.height >= 40
        and bool(lines)
        and max(line_aspects, default=0.0) < 1.5
        and largest_line_ratio >= 0.35
        and segment_area_ratio < 0.20
    )
    if artwork_like:
        return RegionValidityDecision(False, "artwork_like_text_geometry", None, evidence)

    if alnum_count >= 2:
        return RegionValidityDecision(True, "primary_text_evidence", None, evidence)

    if not lines and not segments:
        return RegionValidityDecision(False, "empty_text_geometry", None, evidence)

    # A mask clipped on at least three sides, with no meaningful primary text
    # and sparse foreground, is a fragment rather than a verifier candidate.
    if boundary_touches >= 3 and segment_area_ratio < 0.25:
        return RegionValidityDecision(False, "clipped_junk_fragment", None, evidence)

    # A square-ish detector line without segmentation support is commonly a
    # synthetic YOLO block over artwork.  Elongated DBNet lines remain eligible.
    if not segments and max(line_aspects, default=0.0) < 1.5:
        return RegionValidityDecision(False, "unconfirmed_artwork_geometry", None, evidence)

    return RegionValidityDecision(True, "ctd_text_geometry", None, evidence)


def _polygons(value: object) -> list[Polygon]:
    if not isinstance(value, list) or not value:
        return []
    if _is_point(value[0]):
        candidates: Iterable[object] = [value]
    else:
        candidates = value
    result: list[Polygon] = []
    for candidate in candidates:
        if not isinstance(candidate, list) or len(candidate) < 3:
            continue
        if not all(_is_point(point) for point in candidate):
            continue
        result.append([[float(point[0]), float(point[1])] for point in candidate])
    return result


def _is_point(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    )


def _polygon_area(polygon: Polygon) -> float:
    return abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
        / 2.0
    )


def _polygon_aspect(polygon: Polygon) -> float:
    if len(polygon) == 4:
        edges = [
            math.dist(polygon[index], polygon[(index + 1) % len(polygon)])
            for index in range(4)
        ]
        nonzero = [edge for edge in edges if edge > 0.0]
        if nonzero:
            return max(nonzero) / min(nonzero)
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    short_side = min(width, height)
    return max(width, height) / short_side if short_side > 0.0 else float("inf")


def _boundary_touches(polygons: list[Polygon], bbox: BBox) -> int:
    points = [point for polygon in polygons for point in polygon]
    if not points:
        return 0
    tolerance = max(2.0, min(bbox.width, bbox.height) * 0.05)
    return sum(
        (
            min(point[0] for point in points) <= bbox.x1 + tolerance,
            max(point[0] for point in points) >= bbox.x2 - tolerance,
            min(point[1] for point in points) <= bbox.y1 + tolerance,
            max(point[1] for point in points) >= bbox.y2 - tolerance,
        )
    )


def _recover_bbox(
    region: Region,
    lines: list[Polygon],
    segments: list[Polygon],
    has_primary_text: bool,
) -> BBox | None:
    """Recover only bounded extensions supported by retained CTD geometry."""

    if not lines:
        return None
    points = [point for polygon in (*lines, *segments) for point in polygon]
    if not points:
        return None
    bbox = region.global_bbox
    x1 = max(0, math.floor(min([bbox.x1, *(point[0] for point in points)])))
    y1 = max(0, math.floor(min([bbox.y1, *(point[1] for point in points)])))
    x2 = math.ceil(max([bbox.x2, *(point[0] for point in points)]))
    y2 = math.ceil(max([bbox.y2, *(point[1] for point in points)]))
    extension = max(bbox.x1 - x1, bbox.y1 - y1, x2 - bbox.x2, y2 - bbox.y2)
    if extension <= max(2.0, min(bbox.width, bbox.height) * 0.02):
        return None
    candidate = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
    if candidate.area > bbox.area * 4:
        return None
    # With no readable primary result, require both CTD evidence channels.
    if not has_primary_text and not segments:
        return None
    return candidate
