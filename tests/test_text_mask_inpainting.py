from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.detection import BBox, Detection, Region, RegionStatus, RegionType
from core.detection.merge import merge_duplicates
from core.detection.text_block import group_text_blocks
from core.imaging.inpainter import Inpainter
from core.imaging.text_mask import TextMask, TextMaskBuilder
from core.imaging.text_mask import merge_xor_components, self_or_inverse
from providers.detector.ctd import ComicTextDetector
from core.models import Page


def _region(region_id: int, bbox: tuple[int, int, int, int], text: str, metadata=None) -> Region:
    return Region(
        id=region_id,
        global_bbox=BBox(*bbox),
        type=RegionType.DIALOGUE,
        detection_confidence=0.95,
        source_window_ids=(0,),
        status=RegionStatus.AUTO,
        text=text,
        metadata=metadata or {},
    )


def test_ctd_compact_geometry_survives_duplicate_merge() -> None:
    first = Detection(
        BBox(10, 10, 80, 40), 0.9, RegionType.DIALOGUE, 0,
        metadata={"line_polygons": [[[12, 15], [70, 15], [70, 30], [12, 30]]], "ctd_block_bbox": [10, 10, 80, 40]},
    )
    second = Detection(
        BBox(11, 10, 81, 40), 0.8, RegionType.DIALOGUE, 1,
        metadata={"line_polygons": [[[13, 16], [71, 16], [71, 31], [13, 31]]], "ctd_block_bbox": [11, 10, 81, 40]},
    )
    merged = merge_duplicates([first, second])
    assert len(merged) == 1
    assert len(merged[0].metadata["line_polygons"]) == 2
    assert len(merged[0].metadata["ctd_block_bboxes"]) == 2


def test_mask_refinement_selects_glyphs_and_uniform_fill_is_mask_only() -> None:
    image = Image.new("RGB", (180, 90), "white")
    ImageDraw.Draw(image).text((45, 32), "HELLO", fill="black")
    region = _region(
        1, (38, 24, 125, 58), "HELLO",
        {"line_polygons": [[[38, 24], [125, 24], [125, 58], [38, 58]]]},
    )
    built = TextMaskBuilder().build_for_region(image, region)
    assert 0 < np.count_nonzero(built.refined) < np.count_nonzero(built.raw) * 0.65
    assert built.is_uniform_background

    inpainter = Inpainter()
    output = np.asarray(inpainter.inpaint_regions(image, [region]))
    source = np.asarray(image)
    x1, y1, x2, y2 = built.crop_bbox
    full_mask = np.zeros(source.shape[:2], dtype=bool)
    full_mask[y1:y2, x1:x2] = built.refined > 0
    assert np.array_equal(output[~full_mask], source[~full_mask])


def test_complex_result_is_composited_only_inside_refined_mask() -> None:
    source = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    refined = np.zeros((8, 8), dtype=np.uint8)
    refined[3:5, 2:6] = 255
    text_mask = TextMask((4, 4, 12, 12), source[4:12, 4:12].copy(), np.full((8, 8), 255, np.uint8), refined, (0, 0, 0), False)

    class FakeLaMa:
        def inpaint(self, image, mask):
            return np.full_like(image, (0, 255, 255))

    inpainter = Inpainter()
    inpainter.lama = FakeLaMa()
    output = inpainter._apply_mask(source, text_mask, "test")
    full_mask = np.zeros(source.shape[:2], dtype=bool)
    full_mask[7:9, 6:10] = True
    assert np.array_equal(output[~full_mask], source[~full_mask])
    assert np.all(output[full_mask] == np.array([0, 255, 255]))


def test_component_matching_rejects_unrelated_segmentation_and_protects_frame() -> None:
    image = Image.new("RGB", (160, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 151, 91), outline="black", width=3)
    draw.text((55, 42), "TEXT", fill="black")
    region = _region(10, (48, 34, 112, 67), "TEXT", {
        "line_polygons": [[[48, 34], [112, 34], [112, 67], [48, 67]]],
        "segmentation_polygons": [
            [[50, 36], [110, 36], [110, 65], [50, 65]],
            [[8, 8], [151, 8], [151, 12], [8, 12]],
        ],
    })
    built = TextMaskBuilder().build_for_region(image, region)
    assert not np.any(built.refined[:14])
    assert built.protected_pixels > 0
    assert built.dilation_radius >= 1


def test_upstream_xor_component_refinement_selects_glyphs_not_line_strip() -> None:
    predicted = np.zeros((30, 100), np.uint8)
    predicted[8:22, 10:90] = 255
    candidate = np.zeros_like(predicted)
    for x in range(15, 85, 14):
        candidate[10:20, x:x + 5] = 255
    chosen, _ = self_or_inverse(candidate, predicted)
    merged = merge_xor_components([(chosen, 0)], predicted)
    assert 0 < np.count_nonzero(merged) < np.count_nonzero(predicted) * .5
    assert not np.all(merged[15, 10:90] > 0)


def test_dbnet_representer_uses_rotated_box_score_unclip_and_original_mapping() -> None:
    detector = ComicTextDetector(input_size=64)
    output = np.zeros((1, 2, 64, 64), np.float32)
    output[0, 0, 20:30, 10:40] = .9
    lines = detector._postprocess_dbnet_lines(output, im_w=128, im_h=64)
    assert len(lines) == 1
    polygon = lines[0]["polygon"]
    assert polygon[:, 0].min() < 20 and polygon[:, 0].max() > 78
    assert lines[0]["score"] > .8


def test_ctd_segmentation_mapping_removes_right_and_bottom_letterbox_padding() -> None:
    detector = ComicTextDetector(input_size=64)
    output = np.zeros((1, 1, 64, 64), np.float32)
    output[0, 0, :32, :48] = 1
    mapped = detector._postprocess_segmentation_mask(output, 96, 64, dw=16, dh=32)
    assert mapped.shape == (64, 96)
    assert np.all(mapped == 255)


def test_ctd_dbnet_line_can_modestly_complete_yolo_bbox() -> None:
    detector = ComicTextDetector()
    line = {"polygon": np.array([[10, 20], [90, 20], [90, 35], [10, 35]]), "score": .9}
    grouped = detector._group_blocks_and_lines(
        [{"bbox": [45, 18, 95, 38], "confidence": .9}], [line], np.full((100, 120), 255, np.uint8), 120, 100,
    )
    assert grouped[0]["bbox"] == [10.0, 18, 95, 38]
    assert grouped[0]["lines"] == [line]


def test_ctd_oversized_dbnet_line_uses_contained_yolo_without_duplicate() -> None:
    detector = ComicTextDetector()
    yolo_bbox = [270, 315, 540, 443]
    oversized = {
        "polygon": np.array([[190, 0], [610, 0], [610, 520], [190, 520]]),
        "score": .74,
    }

    grouped = detector._group_blocks_and_lines(
        [{"bbox": yolo_bbox.copy(), "confidence": .67}],
        [oversized],
        np.full((1024, 800), 255, np.uint8),
        800,
        1024,
    )

    assert len(grouped) == 1
    assert grouped[0]["bbox"] == yolo_bbox
    assert grouped[0]["lines"] == [oversized]


def test_ctd_preserves_dbnet_only_line_when_no_yolo_matches() -> None:
    detector = ComicTextDetector()
    unmatched = {
        "polygon": np.array([[5, 60], [35, 60], [35, 80], [5, 80]]),
        "score": .8,
    }

    grouped = detector._group_blocks_and_lines(
        [{"bbox": [60, 5, 90, 25], "confidence": .9}],
        [unmatched],
        np.full((100, 100), 255, np.uint8),
        100,
        100,
    )

    assert len(grouped) == 2
    assert grouped[1]["bbox"] == [5.0, 60.0, 35.0, 80.0]
    assert grouped[1]["lines"] == [unmatched]


def test_ctd_groups_residual_dbnet_lines_without_stealing_yolo_lines() -> None:
    detector = ComicTextDetector()
    residual_a = {
        "polygon": np.array([[5, 60], [35, 60], [35, 75], [5, 75]]),
        "score": .8,
    }
    residual_b = {
        "polygon": np.array([[10, 68], [40, 68], [40, 82], [10, 82]]),
        "score": .75,
    }

    grouped = detector._group_blocks_and_lines(
        [{"bbox": [60, 5, 90, 25], "confidence": .9}],
        [residual_a, residual_b],
        np.full((100, 100), 255, np.uint8),
        100,
        100,
    )

    assert len(grouped) == 2
    assert grouped[1]["bbox"] == [5.0, 60.0, 40.0, 82.0]
    assert grouped[1]["lines"] == [residual_a, residual_b]


def test_residual_second_pass_expands_once_and_tracks_final_identity() -> None:
    source = np.full((20, 20, 3), 255, np.uint8)
    source[9:11, 7:13] = 0
    raw = np.zeros((10, 10), np.uint8)
    raw[4:6, 2:8] = 255
    refined = np.zeros_like(raw)
    refined[4:6, 3:7] = 255
    mask = TextMask((5, 5, 15, 15), source[5:15, 5:15].copy(), raw, refined,
                    (255, 255, 255), True, dilation_radius=1)
    inpainter = Inpainter()
    output = inpainter._apply_mask(source, mask, "residual")
    record = inpainter.debug_records[-1]
    assert record["second_pass"] is True
    assert record["mask_pixels"] > np.count_nonzero(refined)
    final_mask = np.zeros(source.shape[:2], bool)
    # Expanded mask remains local to the original raw geometry.
    final_mask[8:12, 6:14] = True
    assert np.array_equal(output[~final_mask], source[~final_mask])


def test_grouping_recovers_container_and_hyphen_continuations_without_cross_bubble_merge() -> None:
    coords = GlobalCoordinateSystem((Page(0, Path("page.png"), 500, 1000, 0),))
    shared = {"ctd_block_bbox": [80, 80, 310, 190], "ctd_block_ids": ["same"]}
    same_a = _region(1, (100, 90, 280, 110), "THIS IS", shared)
    same_b = _region(2, (105, 146, 285, 166), "THE CONTINUATION", shared)
    blocks = group_text_blocks([same_a, same_b], coords)
    assert [block.member_ids for block in blocks] == [(1, 2)]

    hyphen_a = _region(3, (100, 300, 280, 320), "UNDERSTAND ITS STRUC-", {"ctd_block_bbox": [100, 300, 280, 320]})
    hyphen_b = _region(4, (105, 355, 285, 375), "TURE FIRST", {"ctd_block_bbox": [105, 355, 285, 375]})
    blocks = group_text_blocks([hyphen_a, hyphen_b], coords)
    assert [block.member_ids for block in blocks] == [(3, 4)]

    bubble_a = _region(5, (50, 500, 180, 525), "STOP.", {"ctd_block_bbox": [45, 490, 185, 535]})
    bubble_b = _region(6, (55, 540, 185, 565), "GO NOW", {"ctd_block_bbox": [50, 535, 190, 575]})
    blocks = group_text_blocks([bubble_a, bubble_b], coords)
    assert {block.member_ids for block in blocks} == {(5,), (6,)}

    phantom = _region(7, (300, 545, 385, 565), "MLANSTC", {"ctd_block_ids": ["other"]})
    proper = _region(8, (55, 575, 185, 595), "BUT THE RESULT", {"ctd_block_ids": ["speech"]})
    blocks = group_text_blocks([bubble_b, phantom, proper], coords)
    assert all(not ({7, 8} <= set(block.member_ids)) for block in blocks)


def test_textblock_line_mapping_is_complete_and_explicit_container_conflict_wins() -> None:
    coords = GlobalCoordinateSystem((Page(0, Path("page.png"), 500, 1000, 0),))
    first = _region(20, (80, 100, 220, 125), "STRUC-", {
        "ctd_block_ids": ["same"],
        "ctd_line_memberships": [{"line_id": "a", "polygon": [[80, 100], [220, 100], [220, 125], [80, 125]]}],
    })
    continuation = _region(21, (90, 135, 200, 160), "TURE", {
        "ctd_block_ids": ["same"],
        "ctd_line_memberships": [{"line_id": "b", "polygon": [[90, 135], [200, 135], [200, 160], [90, 160]]}],
    })
    phantom = _region(22, (95, 165, 205, 190), "MLANSTC", {"ctd_block_ids": ["other"]})
    blocks = group_text_blocks([first, continuation, phantom], coords)
    block = next(item for item in blocks if 20 in item.member_ids)
    assert block.member_ids == (20, 21)
    assert block.source_text == "STRUC- TURE"
    assert {item["line_id"] for item in block.metadata["ctd_line_mapping"]} == {"a", "b"}


def test_hyphenated_continuation_with_quote_and_offset_grouping() -> None:
    coords = GlobalCoordinateSystem((Page(0, Path("page.png"), 500, 1000, 0),))
    upper = _region(30, (100, 300, 320, 325), "UNDERSTAND ITS STRUC-", {"ctd_block_bbox": [100, 300, 320, 375]})
    lower = _region(31, (105, 335, 180, 360), 'TURE"...', {"ctd_block_bbox": [100, 300, 320, 375]})
    blocks = group_text_blocks([upper, lower], coords)
    assert len(blocks) == 1
    assert blocks[0].member_ids == (30, 31)
    assert blocks[0].source_text == 'UNDERSTAND ITS STRUC- TURE"...'
