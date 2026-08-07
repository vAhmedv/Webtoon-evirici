"""Detector kayıt defteri.

Mevcut detector provider'ları kaydeder ve döndürür.
"""

from __future__ import annotations

from typing import Callable

from providers.detector.base import DetectorProvider
from providers.detector.dummy import DummyDetector


class DetectorRegistry:
    """Detector provider kayıt defteri."""

    def __init__(self) -> None:
        self._providers: dict[str, Callable[[], DetectorProvider]] = {}
        self._status: dict[str, str] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register("DummyDetector", DummyDetector, status="development/test")
        try:
            from providers.detector.ctd import ComicTextDetector
            self.register("ComicTextDetector", ComicTextDetector, status="experimental")
        except Exception:
            pass
        try:
            from providers.detector.yolo8_comic import Yolo8ComicTextDetector
            self.register("YOLOv8 Comic Text Segmenter", Yolo8ComicTextDetector, status="stable/default")
        except Exception:
            pass

    def register(self, name: str, factory: Callable[[], DetectorProvider], status: str = "stable") -> None:
        self._providers[name] = factory
        self._status[name] = status

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def get_status(self, name: str) -> str:
        return self._status.get(name, "unknown")

    def create(self, name: str) -> DetectorProvider:
        if name not in self._providers:
            raise KeyError(f"Unknown detector: {name}")
        return self._providers[name]()


_registry = DetectorRegistry()


def get_registry() -> DetectorRegistry:
    return _registry
