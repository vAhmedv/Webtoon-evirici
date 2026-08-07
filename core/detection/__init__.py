"""Detection paketi.

Detector sonuçları, koordinat dönüşümleri, merge ve kalite kapısı.
"""

from .bbox import BBox
from .detection import Detection, Region, RegionStatus, RegionType
from .merge import merge_duplicates
from .coordinate import window_bbox_to_global, global_bbox_to_window

__all__ = [
    "BBox",
    "Detection",
    "Region",
    "RegionStatus",
    "RegionType",
    "merge_duplicates",
    "window_bbox_to_global",
    "global_bbox_to_window",
]