"""Detection paketi.

Detector sonuçları, koordinat dönüşümleri, merge, cache ve kalite kapısı.
"""

from .bbox import BBox
from .cache import CACHE_PATH, CACHE_FILENAME, DetectionCache, compute_image_hash
from .detection import Detection, Region, RegionStatus, RegionType
from .merge import merge_duplicates
from .coordinate import (
    window_bbox_to_global,
    global_bbox_to_window,
    window_polygon_to_global,
    global_polygon_to_window,
)

__all__ = [
    "BBox",
    "CACHE_FILENAME",
    "CACHE_PATH",
    "Detection",
    "DetectionCache",
    "Region",
    "RegionStatus",
    "RegionType",
    "compute_image_hash",
    "merge_duplicates",
    "window_bbox_to_global",
    "global_bbox_to_window",
    "window_polygon_to_global",
    "global_polygon_to_window",
]