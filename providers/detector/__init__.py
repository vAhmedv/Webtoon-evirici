"""Detector sağlayıcı paketi."""

from .base import DetectorProvider
from .dummy import DummyDetector

__all__ = ["DetectorProvider", "DummyDetector"]