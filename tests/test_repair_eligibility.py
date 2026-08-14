from types import SimpleNamespace

from core.detection import BBox, Region, RegionStatus, RegionType
from core.detection.repair_eligibility import evaluate_repair_eligibility


def _region(*, lines=1, segments=1, touches=0, aspect=2.0, valid=True) -> Region:
    return Region(
        id=1,
        global_bbox=BBox(0, 0, 100, 50),
        type=RegionType.UNKNOWN,
        detection_confidence=0.9,
        source_window_ids=(1,),
        status=RegionStatus.REVIEW,
        metadata={
            "region_validity": {
                "valid": valid,
                "reason": "ctd_text_geometry" if valid else "artwork_like_text_geometry",
                "line_polygon_count": lines,
                "segmentation_polygon_count": segments,
                "segmentation_boundary_touches": touches,
                "max_line_aspect": aspect,
            }
        },
    )


def _verdict(reason="word_difference"):
    return SimpleNamespace(requires_review=True, needs_repair=True, reason=reason)


def test_strong_validity_rejection_is_never_qwen_eligible() -> None:
    decision = evaluate_repair_eligibility(_region(valid=False), _verdict())
    assert not decision.eligible
    assert decision.reason == "strong_region_validity_rejection"


def test_verifier_only_single_weak_fragment_is_not_qwen_eligible() -> None:
    decision = evaluate_repair_eligibility(
        _region(lines=1, segments=2, touches=3, aspect=1.2),
        _verdict("primary_empty_verifier_filled"),
    )
    assert not decision.eligible
    assert decision.reason == "verifier_only_weak_text_geometry"


def test_empty_primary_with_multiple_ctd_text_lines_remains_eligible() -> None:
    decision = evaluate_repair_eligibility(
        _region(lines=3, segments=8, touches=1, aspect=2.2),
        _verdict("primary_empty_verifier_filled"),
    )
    assert decision.eligible


def test_real_lexical_ambiguity_with_text_geometry_is_eligible() -> None:
    assert evaluate_repair_eligibility(_region(), _verdict()).eligible
