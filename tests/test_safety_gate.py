"""Safety gate testleri."""

from __future__ import annotations

import pytest

from core.detection import RegionStatus, RegionType
from core.detection.merge import _assign_status


def test_dialogue_high_confidence_auto() -> None:
    assert _assign_status(RegionType.DIALOGUE, 0.9) == RegionStatus.AUTO


def test_dialogue_low_confidence_review() -> None:
    assert _assign_status(RegionType.DIALOGUE, 0.3) == RegionStatus.REVIEW


def test_narration_high_confidence_auto() -> None:
    assert _assign_status(RegionType.NARRATION, 0.8) == RegionStatus.AUTO


def test_narration_low_confidence_review() -> None:
    assert _assign_status(RegionType.NARRATION, 0.4) == RegionStatus.REVIEW


def test_sfx_skipped() -> None:
    assert _assign_status(RegionType.SFX, 0.9) == RegionStatus.SKIP


def test_watermark_skipped() -> None:
    assert _assign_status(RegionType.WATERMARK, 0.9) == RegionStatus.SKIP


def test_unknown_review() -> None:
    assert _assign_status(RegionType.UNKNOWN, 0.9) == RegionStatus.REVIEW