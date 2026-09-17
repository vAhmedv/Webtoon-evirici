"""GUI inceleme kaydı testleri (Qt yok / offscreen).

Anti-overfit: sentetik bölgeler, göreli yollar (tmp_path).
"""

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.detection import BBox, Region, RegionStatus, RegionType
from gui.main_window import save_review_regions


def _region(rid: int, text: str, tr: str | None, status: RegionStatus) -> Region:
    return Region(
        id=rid,
        global_bbox=BBox(10, 10, 100, 40),
        type=RegionType.UNKNOWN,
        detection_confidence=0.7,
        source_window_ids=(0,),
        status=status,
        text=text,
        translation=tr,
        ocr_confidence=0.8,
        metadata={},
    )


def _write_analysis(tmp_path: Path) -> tuple[Path, list[Region]]:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    regions = [
        _region(1, "HELLO", "Merhaba", RegionStatus.AUTO),
        _region(2, "BYE", None, RegionStatus.REVIEW),
    ]
    (analysis / "regions.json").write_text(
        json.dumps(
            {
                "text_blocks_count": 1,
                "text_blocks": [{"id": 5, "member_ids": [1], "translation": "Merhaba"}],
                "regions": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (analysis / "summary.json").write_text(
        json.dumps({"translated": 0, "skipped": 0, "review": 0}, ensure_ascii=False),
        encoding="utf-8",
    )
    return analysis, regions


def test_save_review_regions_updates_files(tmp_path: Path) -> None:
    analysis, regions = _write_analysis(tmp_path)
    stats = save_review_regions(analysis, regions, {5: "Merhaba!"})
    assert stats["saved_regions"] == 2
    assert stats["synced_blocks"] == 1

    data = json.loads((analysis / "regions.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in data["regions"]] == [1, 2]
    assert data["regions"][0]["translation"] == "Merhaba"
    assert data["text_blocks"][0]["translation"] == "Merhaba!"

    summary = json.loads((analysis / "summary.json").read_text(encoding="utf-8"))
    assert summary["translated"] == 1
    assert summary["review"] == 1
    assert summary["skipped"] == 0


def test_save_review_regions_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        save_review_regions(tmp_path / "yok", [], {})


def test_inspector_flush_emits_once() -> None:
    """Debounce: bekleyen düzenleme tek seferde yayınlanır."""
    from PySide6.QtWidgets import QApplication

    from gui.components.right_inspector import RightInspector

    app = QApplication.instance() or QApplication([])
    inspector = RightInspector()
    received: list[tuple[int, str]] = []
    inspector.translation_updated.connect(lambda rid, txt: received.append((rid, txt)))

    region = _region(9, "HELLO", "Merhaba", RegionStatus.REVIEW)
    inspector.display_region(region, 0, 1)
    inspector.tr_text.setPlainText("Selam")
    assert received == []  # debounce: hemen yayınlanmaz
    inspector._flush_pending_edit()
    assert received == [(9, "Selam")]
    inspector._flush_pending_edit()
    inspector.close()
