"""Serileştirme paketi."""

from .serializer import (
    detection_to_dict,
    dict_to_detection,
    dict_to_region,
    region_to_dict,
)

__all__ = [
    "region_to_dict",
    "dict_to_region",
    "detection_to_dict",
    "dict_to_detection",
]
