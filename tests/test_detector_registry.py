"""Detector registry testleri."""

from __future__ import annotations

import pytest

from providers.detector.registry import get_registry
from providers.detector.dummy import DummyDetector


def test_registry_lists_dummy() -> None:
    registry = get_registry()
    names = registry.list_providers()
    assert "DummyDetector" in names


def test_registry_creates_dummy() -> None:
    registry = get_registry()
    provider = registry.create("DummyDetector")
    assert isinstance(provider, DummyDetector)
    assert provider.name == "dummy"


def test_registry_unknown_detector_raises() -> None:
    registry = get_registry()
    with pytest.raises(KeyError):
        registry.create("NonExistentDetector")
