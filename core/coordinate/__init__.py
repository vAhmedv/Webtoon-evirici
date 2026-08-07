"""Koordinat sistemi paketi."""

from .global_coords import GlobalCoordinateSystem, compute_y_offsets
from .sliding_window import generate_windows, generate_windows_for_pages

__all__ = [
    "GlobalCoordinateSystem",
    "compute_y_offsets",
    "generate_windows",
    "generate_windows_for_pages",
]