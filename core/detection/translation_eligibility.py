"""Translation eligibility is intentionally separate from TextBlock grouping."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.detection import Region, RegionStatus, RegionType
from core.detection.text_block import TextBlock


NON_STORY_TYPES = frozenset({RegionType.SFX, RegionType.WATERMARK})

# Faz 2 parça-kuralı: EN kaynakta tek başına tam cümle olamayacak işlev
# sözcükleri. Cümle bölünmesiyle kopan kırıntılar ("THE", "I" + devamı başka
# blokta) çevrilip basılmasın; tam cümle komşu blokta zaten çevrilir.
# Tam tek-kelimelik ünlemler ("EVET.", "DUR!") etkilenmez — listede yoklar.
FRAGMENT_TOKENS = frozenset({"THE", "A", "AN", "I"})


def is_translatable_member(member: Region) -> bool:
    """Tek üyenin çeviriye uygun olup olmadığı (blok-veto yok)."""
    if member.status is not RegionStatus.AUTO:
        return False
    if member.type in NON_STORY_TYPES:
        return False
    if not member.text or not member.text.strip():
        return False
    validity = member.metadata.get("region_validity") if isinstance(member.metadata, dict) else None
    if isinstance(validity, dict) and validity.get("valid") is False:
        return False
    return True


def eligible_members(block: TextBlock) -> tuple[Region, ...]:
    """Blok içinden çevrilebilir üyeleri döndür (SFX/REVIEW filtreli)."""
    return tuple(m for m in block.members if is_translatable_member(m))


def eligible_source_text(block: TextBlock) -> str:
    """Yalnızca çevrilebilir üyelerin birleştirilmiş metni."""
    return " ".join(
        (m.text or "").strip() for m in eligible_members(block) if (m.text or "").strip()
    )


@dataclass(frozen=True)
class TranslationEligibilityDecision:
    eligible: bool
    reason: str
    eligible_count: int = 0
    total_count: int = 0


def evaluate_translation_eligibility(block: TextBlock) -> TranslationEligibilityDecision:
    total = len(block.members)
    if not block.members:
        return TranslationEligibilityDecision(False, "empty_block", 0, total)
    eligible = eligible_members(block)
    if eligible:
        # Faz 2: uygun metin kelime içeriği taşımıyorsa (yalnız noktalama:
        # "?", "!") veya kopuk işlev-sözcük kırıntısıysa ele.
        norm = re.sub(r"\s+", " ", eligible_source_text(block).strip())
        if not re.search(r"\w", norm, re.UNICODE):
            return TranslationEligibilityDecision(False, "no_word_content", len(eligible), total)
        bare = re.sub(r"[^\w\s]", "", norm, flags=re.UNICODE).strip().upper()
        if bare in FRAGMENT_TOKENS:
            return TranslationEligibilityDecision(False, "fragment", len(eligible), total)
    if not eligible:
        # Tamamen uygunsuz: sebebi teşhis et (önceki katı davranış korunur)
        if not block.source_text or not block.source_text.strip():
            return TranslationEligibilityDecision(False, "blank_source_text", 0, total)
        if any(member.status is not RegionStatus.AUTO for member in block.members):
            # Tümü non-AUTO ise eski reason, kısmi ise aşağıda partial olur
            pass
        if any(member.type in (RegionType.SFX, RegionType.WATERMARK) for member in block.members):
            if all(m.type in (RegionType.SFX, RegionType.WATERMARK) for m in block.members):
                return TranslationEligibilityDecision(False, "non_story_member", 0, total)
        for member in block.members:
            validity = member.metadata.get("region_validity") if isinstance(member.metadata, dict) else None
            if isinstance(validity, dict) and validity.get("valid") is False:
                if len(block.members) == 1:
                    return TranslationEligibilityDecision(False, "strong_validity_rejection", 0, total)
        # Karışık blokta hiç uygun üye yoksa genel ret
        if any(member.status is not RegionStatus.AUTO for member in block.members):
            return TranslationEligibilityDecision(False, "non_auto_member", 0, total)
        return TranslationEligibilityDecision(False, "no_eligible_members", 0, total)
    if len(eligible) < total:
        return TranslationEligibilityDecision(True, "partial_members_filtered", len(eligible), total)
    # Tam uygunluk: eski katı kontrollerden geç
    for member in block.members:
        validity = member.metadata.get("region_validity") if isinstance(member.metadata, dict) else None
        if isinstance(validity, dict) and validity.get("valid") is False:
            return TranslationEligibilityDecision(False, "strong_validity_rejection", len(eligible), total)
    return TranslationEligibilityDecision(True, "all_members_auto_story_text", len(eligible), total)
