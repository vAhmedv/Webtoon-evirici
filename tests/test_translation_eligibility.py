from core.detection import BBox, Region, RegionStatus, RegionType
from core.detection.text_block import TextBlock
from core.detection.translation_eligibility import (
    eligible_members,
    eligible_source_text,
    evaluate_translation_eligibility,
)


def _member(region_id: int, status: RegionStatus, *, validity=True) -> Region:
    return Region(
        id=region_id,
        global_bbox=BBox(0, region_id * 20, 100, region_id * 20 + 15),
        type=RegionType.DIALOGUE,
        detection_confidence=0.9,
        source_window_ids=(1,),
        status=status,
        text=f"TEXT {region_id}",
        metadata={"region_validity": {"valid": validity}},
    )


def _block(*members: Region) -> TextBlock:
    return TextBlock(
        id=1,
        member_ids=tuple(member.id for member in members),
        members=members,
        merged_bbox=BBox(0, 0, 100, 100),
        source_text=" ".join(member.text or "" for member in members),
    )


def test_all_auto_story_members_are_translation_eligible() -> None:
    assert evaluate_translation_eligibility(_block(_member(1, RegionStatus.AUTO))).eligible


def test_mixed_auto_review_block_is_not_translation_eligible() -> None:
    # Kısmi blok: 1 uygun üye varsa blok eligible, yalnızca uygun üye çevrilir.
    decision = evaluate_translation_eligibility(
        _block(_member(1, RegionStatus.AUTO), _member(2, RegionStatus.REVIEW))
    )
    assert decision.eligible
    assert decision.reason == "partial_members_filtered"
    assert decision.eligible_count == 1


def test_sfx_only_block_is_not_translation_eligible() -> None:
    sfx = _member(1, RegionStatus.AUTO)
    object.__setattr__(sfx, "type", RegionType.SFX)
    decision = evaluate_translation_eligibility(_block(sfx))
    assert not decision.eligible
    assert decision.reason == "non_story_member"


def test_partial_block_filters_sfx_member() -> None:
    auto = _member(1, RegionStatus.AUTO)
    sfx = _member(2, RegionStatus.AUTO)
    object.__setattr__(sfx, "type", RegionType.SFX)
    block = _block(auto, sfx)
    assert [m.id for m in eligible_members(block)] == [1]
    assert eligible_source_text(block) == "TEXT 1"


def test_validity_rejected_member_is_not_translation_eligible_even_if_auto() -> None:
    decision = evaluate_translation_eligibility(
        _block(_member(1, RegionStatus.AUTO, validity=False))
    )
    assert not decision.eligible
    assert decision.reason == "strong_validity_rejection"
