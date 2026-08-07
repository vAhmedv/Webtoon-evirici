"""Bölüm analiz hizmeti testleri."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from application.cancellation import CancellationToken, CancelledError
from application.chapter_analyzer import AnalysisResult, ChapterAnalyzer
from application.progress import ProgressEvent
from core.config import Config
from core.detection import RegionStatus
from providers.detector.dummy import DummyDetector


def test_analyzer_runs_with_synthetic_chapter(tmp_path: Path) -> None:
    """Sentetik test bölümü ile analyzer çalışır."""
    chapter_dir = tmp_path / "chapter"
    chapter_dir.mkdir()

    from PIL import Image, ImageDraw

    for i in range(3):
        img = Image.new("RGB", (800, 1000), (200, 200, 200))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), f"Page {i}", fill=(0, 0, 0))
        img.save(chapter_dir / f"{i:03d}.webp", "WEBP", quality=95)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = Config()
    analyzer = ChapterAnalyzer(config)
    detector = DummyDetector(seed=42)

    events: list[ProgressEvent] = []

    def on_progress(event: ProgressEvent) -> None:
        events.append(event)

    result = analyzer.analyze(
        chapter_path=chapter_dir,
        output_path=output_dir,
        detector=detector,
        progress_callback=on_progress,
    )

    assert isinstance(result, AnalysisResult)
    assert len(result.pages) == 3
    assert result.windows is not None
    assert len(result.regions) > 0
    assert result.elapsed_time > 0

    assert (output_dir / "analysis" / "regions.json").exists()
    assert (output_dir / "analysis" / "summary.json").exists()
    assert (output_dir / "analysis" / "preview.png").exists()

    with open(output_dir / "analysis" / "summary.json") as f:
        summary = json.load(f)
    assert summary["pages"] == 3
    assert summary["windows"] == len(result.windows)
    assert summary["regions"] == len(result.regions)

    assert len(events) > 0
    stages = [e.stage for e in events]
    assert "Loading chapter" in stages
    assert "Detecting" in stages
    assert "Completed" in stages


def test_cancellation_is_honored(tmp_path: Path) -> None:
    """İptal belirteci pipeline'ı durdurur."""
    chapter_dir = tmp_path / "chapter"
    chapter_dir.mkdir()

    from PIL import Image

    for i in range(2):
        img = Image.new("RGB", (800, 1000), (200, 200, 200))
        img.save(chapter_dir / f"{i:03d}.webp", "WEBP", quality=95)

    config = Config()
    analyzer = ChapterAnalyzer(config)
    detector = DummyDetector(seed=42)
    token = CancellationToken()
    token.cancel()

    with pytest.raises(CancelledError):
        analyzer.analyze(
            chapter_path=chapter_dir,
            output_path=tmp_path / "output",
            detector=detector,
            cancellation_token=token,
        )


def test_invalid_chapter_path_raises() -> None:
    """Geçersiz chapter yolu hata üretir."""
    config = Config()
    analyzer = ChapterAnalyzer(config)
    detector = DummyDetector(seed=42)

    with pytest.raises((FileNotFoundError, ValueError)):
        analyzer.analyze(
            chapter_path="/nonexistent/path",
            output_path="out",
            detector=detector,
        )


def test_dummy_detector_produces_regions(tmp_path: Path) -> None:
    """DummyDetector region üretir."""
    chapter_dir = tmp_path / "chapter"
    chapter_dir.mkdir()

    from PIL import Image

    img = Image.new("RGB", (800, 1000), (200, 200, 200))
    img.save(chapter_dir / "001.webp", "WEBP", quality=95)

    config = Config()
    analyzer = ChapterAnalyzer(config)
    detector = DummyDetector(seed=123)

    result = analyzer.analyze(
        chapter_path=chapter_dir,
        output_path=tmp_path / "output",
        detector=detector,
    )

    assert len(result.regions) > 0
    statuses = {r.status for r in result.regions}
    assert RegionStatus.SKIP in statuses or RegionStatus.AUTO in statuses or RegionStatus.REVIEW in statuses
