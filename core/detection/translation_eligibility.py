"""Translation eligibility is intentionally separate from TextBlock grouping."""

from __future__ import annotations

from dataclasses import dataclass

from core.detection import RegionStatus, RegionType
from core.detection.text_block import TextBlock


@dataclass(frozen=True)
class TranslationEligibilityDecision:
    eligible: bool
    reason: str


def evaluate_translation_eligibility(block: TextBlock) -> TranslationEligibilityDecision:
    if not block.members:
        return TranslationEligibilityDecision(False, "empty_block")
    if not block.source_text or not block.source_text.strip():
        return TranslationEligibilityDecision(False, "blank_source_text")
    if any(member.status is not RegionStatus.AUTO for member in block.members):
        return TranslationEligibilityDecision(False, "non_auto_member")
    if any(member.type in (RegionType.SFX, RegionType.WATERMARK) for member in block.members):
        return TranslationEligibilityDecision(False, "non_story_member")
    for member in block.members:
        validity = member.metadata.get("region_validity") if isinstance(member.metadata, dict) else None
        if isinstance(validity, dict) and validity.get("valid") is False:
            return TranslationEligibilityDecision(False, "strong_validity_rejection")
    return TranslationEligibilityDecision(True, "all_members_auto_story_text")
