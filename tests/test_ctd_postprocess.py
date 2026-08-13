from __future__ import annotations

import pytest

from providers.detector.ctd import ComicTextDetector


def test_yolo_block_postprocess_restores_upstream_semantics() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    detector = ComicTextDetector("/nonexistent/path")

    raw = np.array(
        [[
            # Same-class overlap: only the higher combined score survives.
            [50.0, 50.0, 40.0, 20.0, 0.9, 0.9, 0.1],
            [51.0, 50.0, 40.0, 20.0, 0.8, 0.8, 0.2],
            # Same geometry, different class: class-aware NMS keeps it.
            [50.0, 50.0, 40.0, 20.0, 0.9, 0.1, 0.9],
            # Boxes are clipped at both image boundaries.
            [5.0, 5.0, 20.0, 20.0, 0.9, 0.9, 0.1],
            [95.0, 95.0, 20.0, 20.0, 0.9, 0.9, 0.1],
            # Objectness alone passes, but objectness * class does not.
            [75.0, 20.0, 10.0, 10.0, 0.95, 0.3, 0.2],
        ]],
        dtype=np.float32,
    )

    blocks = detector._postprocess_yolo_blocks(raw, 1.0, 1.0, 100, 100)

    assert len(blocks) == 4
    assert sum(block["bbox"] == [30.0, 40.0, 70.0, 60.0] for block in blocks) == 2
    assert any(block["bbox"] == [0.0, 0.0, 15.0, 15.0] for block in blocks)
    assert any(block["bbox"] == [85.0, 85.0, 100.0, 100.0] for block in blocks)
    assert all(block["confidence"] == pytest.approx(0.81) for block in blocks)


def test_yolo_block_postprocess_decodes_real_runtime_row() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    detector = ComicTextDetector("/nonexistent/path")
    raw = np.array(
        [[[
            405.35516357421875,
            382.7852783203125,
            272.05914306640625,
            120.36568450927734,
            0.6141827702522278,
            0.995806097984314,
            0.004258374217897654,
        ]]],
        dtype=np.float32,
    )

    blocks = detector._postprocess_yolo_blocks(raw, 1.0, 1.0, 800, 1024)

    assert len(blocks) == 1
    assert blocks[0]["bbox"] == pytest.approx(
        [269.325592, 322.602448, 541.384766, 442.968109],
        abs=1e-4,
    )
    assert blocks[0]["confidence"] == pytest.approx(0.611607, abs=1e-5)
