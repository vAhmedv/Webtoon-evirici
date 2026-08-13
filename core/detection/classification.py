"""Multi-feature Region classification and generic watermark detection module."""

from __future__ import annotations

import re
from typing import Sequence
from loguru import logger

from core.coordinate.global_coords import GlobalCoordinateSystem
from core.detection.detection import Region, RegionStatus, RegionType
from core.ocr_normalizer import normalize_ocr_text

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def classify_regions(
    regions: Sequence[Region],
    coords: GlobalCoordinateSystem,
) -> list[Region]:
    """Çok özellikli (multi-feature) bölge sınıflandırması uygular.

    SFX, Watermark, Non-Text ve Story Text ayırımı yapar.
    Generic çapraz sayfa watermark tespiti içerir.
    """
    if not regions:
        return []

    # 1. Generic Cross-Page Watermark Detector
    # Sayfa göreli Y koordinatı (0.0-1.0) ve normalize metin bazlı tekrar tespiti
    page_rel_map: dict[str, set[int]] = {}
    
    for r in regions:
        if not r.text or len(r.text.strip()) < 3:
            continue
        norm_txt = normalize_ocr_text(r.text).upper()
        center_y = (r.global_bbox.y1 + r.global_bbox.y2) // 2
        page_idx, page_y = coords.global_to_page(center_y)
        page_obj = coords.pages[page_idx] if page_idx < len(coords.pages) else None
        
        if page_obj and page_obj.height > 0:
            rel_y = round(page_y / page_obj.height, 2)
            # Sayfa üst (%10) veya alt (%10) kenarlarında tekrar eden metinler
            if rel_y <= 0.10 or rel_y >= 0.90:
                page_rel_map.setdefault(norm_txt, set()).add(page_idx)

    # Birden fazla farklı sayfada aynı kenar konumunda tekrar eden metinler watermark'tır
    watermark_texts = {txt for txt, pages in page_rel_map.items() if len(pages) >= 2}

    classified_regions: list[Region] = []
    
    for r in regions:
        bbox = r.global_bbox
        txt = r.text or ""
        norm_txt = normalize_ocr_text(txt).upper()
        
        # 1a. Watermark kontrolü
        if norm_txt in watermark_texts:
            classified_regions.append(
                _replace_region_status(
                    r,
                    reg_type=RegionType.WATERMARK,
                    status=RegionStatus.SKIP,
                    reason="generic_cross_page_watermark_skip",
                )
            )
            continue

        # 1b. Güçlü Detector Sinyalleri
        if r.type in (RegionType.SFX, RegionType.WATERMARK):
            classified_regions.append(
                _replace_region_status(
                    r,
                    reg_type=r.type,
                    status=RegionStatus.SKIP,
                    reason="detector_sfx_watermark_skip",
                )
            )
            continue

        # 1c. Boş / Çok Küçük Gürültü Bölgeleri
        if bbox.height < 10 or bbox.width < 10:
            classified_regions.append(
                _replace_region_status(
                    r,
                    reg_type=RegionType.UNKNOWN,
                    status=RegionStatus.SKIP,
                    reason="tiny_noise_box_skip",
                )
            )
            continue

        # 1d. CJK Script Değerlendirmesi (Tek başına SKIP sebebi değildir)
        if _contains_cjk(txt):
            # CJK + stilize geometri (örneğin dev dikey/yatay sfx harfi veya aşırı oran) -> SFX
            aspect_ratio = max(bbox.width, bbox.height) / max(1, min(bbox.width, bbox.height))
            if aspect_ratio > 3.5 or len(norm_txt) <= 2:
                classified_regions.append(
                    _replace_region_status(
                        r,
                        reg_type=RegionType.SFX,
                        status=RegionStatus.SKIP,
                        reason="cjk_stylized_sfx_skip",
                    )
                )
            else:
                # Belirsiz CJK metni -> REVIEW
                classified_regions.append(
                    _replace_region_status(
                        r,
                        reg_type=RegionType.UNKNOWN,
                        status=RegionStatus.REVIEW,
                        reason="ambiguous_cjk_review",
                    )
                )
            continue

        # 1e. Hikaye Metni (STORY_TEXT) vs UNKNOWN Değerlendirmesi
        if r.type in (RegionType.DIALOGUE, RegionType.NARRATION):
            classified_regions.append(r)
        elif r.type == RegionType.UNKNOWN:
            # UNKNOWN için içerik ve alfabe kontrolü
            if norm_txt and len(norm_txt) >= 2 and any(c.isalpha() for c in norm_txt):
                classified_regions.append(r)
            elif not norm_txt or not any(c.isalnum() for c in norm_txt):
                classified_regions.append(
                    _replace_region_status(
                        r,
                        reg_type=RegionType.UNKNOWN,
                        status=RegionStatus.SKIP,
                        reason="non_text_noise_skip",
                    )
                )
            else:
                # Belirsiz UNKNOWN -> REVIEW
                classified_regions.append(
                    _replace_region_status(
                        r,
                        reg_type=RegionType.UNKNOWN,
                        status=RegionStatus.REVIEW,
                        reason="ambiguous_unknown_review",
                    )
                )
        else:
            classified_regions.append(r)

    return classified_regions


def _replace_region_status(
    region: Region,
    reg_type: RegionType,
    status: RegionStatus,
    reason: str,
) -> Region:
    return Region(
        id=region.id,
        global_bbox=region.global_bbox,
        type=reg_type,
        detection_confidence=region.detection_confidence,
        source_window_ids=region.source_window_ids,
        status=status,
        text=region.text,
        ocr_confidence=region.ocr_confidence,
        translation=region.translation,
        review_reason=reason,
        metadata=dict(region.metadata),
    )
