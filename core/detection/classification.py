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


_DIALOGUE_EXCLAMATIONS = {
    "DAMMIT", "DAMN", "SHIT", "FUCK", "WHAT", "WHAT?!", "WHAT?", "WHAT!",
    "AHH", "AHHH", "AH", "NO", "NO!", "NO...", "NO WAY", "DIE", "DIE!",
    "HUH", "HUH?", "WAIT", "WAIT!", "WHY", "WHY?", "STOP", "STOP!",
    "HEY", "HEY!", "UGH", "UGHH", "GASP", "HEH", "HEHE", "OH", "OH!", "OH?",
    "WOW", "YES", "YES!", "PLEASE", "HELP", "HELP!", "RUN", "RUN!",
    "LOOK", "LOOK!", "LISTEN", "OOF", "OUCH", "HURRY", "SHUT UP", "GET OUT",
    "LET GO", "MY", "YOU", "ME", "HMM", "HM", "TCH", "OUCH", "PHEW", "GULP",
    "KID", "MAN", "BRO", "SIR", "LORD", "MASTER", "WHOA", "WHO", "HOW",
    "WHERE", "WHEN", "OKAY", "OK", "COME ON",
}


def _is_dialogue_exclamation(norm_txt: str) -> bool:
    """Checks if text is a known dialogue exclamation or short speech."""
    if not norm_txt:
        return False
    upper = norm_txt.upper().strip()
    if upper in _DIALOGUE_EXCLAMATIONS:
        return True
    words = [w for w in re.findall(r"[A-Z']+", upper) if w]
    return bool(words and any(w in _DIALOGUE_EXCLAMATIONS for w in words))


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
            # Diyalog ve konuşma ünlemlerinin SFX olarak bypass edilmesini engelle
            if r.type == RegionType.SFX and _is_dialogue_exclamation(norm_txt):
                classified_regions.append(
                    _replace_region_status(
                        r,
                        reg_type=RegionType.DIALOGUE,
                        status=RegionStatus.AUTO,
                        reason="dialogue_exclamation_promoted",
                    )
                )
                continue

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

        # 1d. CJK Script Değerlendirmesi (Çoklu Kanıtlı SFX ve Gürültü Filtreleme)
        if _contains_cjk(txt):
            if _is_credit_metadata_region(r, coords, norm_txt):
                # Kapak veya son sayfa kredi şeridi üzerindeki CJK metni
                classified_regions.append(
                    _replace_region_status(
                        r,
                        reg_type=RegionType.WATERMARK,
                        status=RegionStatus.SKIP,
                        reason="credit_metadata_skip",
                    )
                )
            elif _is_cjk_stylized_sfx(r, txt, norm_txt):
                # Çizim üzerine gömülü geniş alanlı / stilize CJK ses efekti (SFX)
                classified_regions.append(
                    _replace_region_status(
                        r,
                        reg_type=RegionType.SFX,
                        status=RegionStatus.SKIP,
                        reason="cjk_stylized_sfx_skip",
                    )
                )
            elif _is_multi_signal_non_text_noise(r, norm_txt):
                # Birincil OCR boş + zayıf geometri/verifier tekil CJK halüsinasyonu
                classified_regions.append(
                    _replace_region_status(
                        r,
                        reg_type=RegionType.UNKNOWN,
                        status=RegionStatus.SKIP,
                        reason="non_text_noise_skip",
                    )
                )
            else:
                # Belirsiz CJK metni (Hikaye diyalogu olabilecek çoklu kelimeler) -> REVIEW
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
            elif _is_multi_signal_non_text_noise(r, norm_txt):
                # Çoklu kanıt: Birincil OCR boş + zayıf geometri/verifier halüsinasyonu -> SKIP
                classified_regions.append(
                    _replace_region_status(
                        r,
                        reg_type=RegionType.UNKNOWN,
                        status=RegionStatus.SKIP,
                        reason="non_text_noise_skip",
                    )
                )
            elif _is_credit_metadata_region(r, coords, norm_txt):
                # Çoklu kanıt: Kapak/kredi sayfası sınır geometrisi ve künye metni -> SKIP
                classified_regions.append(
                    _replace_region_status(
                        r,
                        reg_type=RegionType.WATERMARK,
                        status=RegionStatus.SKIP,
                        reason="credit_metadata_skip",
                    )
                )
            elif _is_isolated_drawing_sfx(r, norm_txt):
                # Çoklu kanıt: Çizim üzerine gömülü geniş alanlı vokalizasyon/SFX glifi -> SKIP
                classified_regions.append(
                    _replace_region_status(
                        r,
                        reg_type=RegionType.SFX,
                        status=RegionStatus.SKIP,
                        reason="sfx_skip",
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


def _is_credit_metadata_region(region: Region, coords: GlobalCoordinateSystem, norm_txt: str) -> bool:
    """Kapak veya son sayfa kredi kartı üzerindeki hikaye dışı öğeleri çoklu kanıtla saptar."""
    if region.type != RegionType.UNKNOWN:
        return False

    if not coords or not coords.pages or len(coords.pages) < 2:
        return False

    center_y = (region.global_bbox.y1 + region.global_bbox.y2) // 2
    page_idx, page_y = coords.global_to_page(center_y)
    total_pages = len(coords.pages)

    # Sinyal 1: Kapak sayfası (0) veya son sayfa (kredi/künye kartı)
    is_boundary_page = (page_idx == 0 or page_idx == total_pages - 1)
    if not is_boundary_page:
        return False

    # Sinyal 2: Tipik yatay banner / kredi satırı geometrisi (yüksek en-boy oranı veya geniş ve ince blok)
    w = region.global_bbox.width
    h = region.global_bbox.height
    aspect = w / max(1, h)

    return bool(aspect > 3.0 or (w > 200 and h < 120))


_KANA_PATTERN = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")


def _is_cjk_stylized_sfx(region: Region, txt: str, norm_txt: str) -> bool:
    """Çizim katmanına doğrudan çizilmiş geniş alanlı veya stilize CJK ses efektlerini saptar."""
    if region.type not in (RegionType.UNKNOWN, RegionType.SFX):
        return False

    w = region.global_bbox.width
    h = region.global_bbox.height
    area = w * h
    aspect_ratio = max(w, h) / max(1, min(w, h))

    cjk_chars = [c for c in txt if _contains_cjk(c)]
    cjk_len = len(cjk_chars)
    is_kana = bool(_KANA_PATTERN.search(txt))

    # Sinyal 1: Aşırı en-boy oranı (dikey/yatay sfx şeridi)
    if aspect_ratio > 3.0 and cjk_len <= 6:
        return True

    # Sinyal 2: Geniş çizim glifi alanı (alan >= 25,000px² veya her iki boyut >= 150px) ve kısa CJK
    if (area >= 25000 or (w >= 150 and h >= 150)) and cjk_len <= 4:
        return True

    # Sinyal 3: Saf Katakana/Hiragana ses efekti onomatopoeia (alan >= 15,000px² ve kısa metin)
    if is_kana and cjk_len <= 4 and area >= 15000:
        return True

    return False


def _is_isolated_drawing_sfx(region: Region, norm_txt: str) -> bool:
    """Çizim katmanına doğrudan çizilmiş geniş alanlı stilize ses efektlerini saptar."""
    if region.type != RegionType.UNKNOWN:
        return False

    if _is_dialogue_exclamation(norm_txt):
        return False

    w = region.global_bbox.width
    h = region.global_bbox.height
    area = w * h

    # Sinyal 1: Büyük çizim glifi alanı (alan >= 25,000px² veya her iki boyut >= 150px)
    is_large_glyph = (area >= 25000 or (w >= 150 and h >= 150))

    # Sinyal 2: Kısa ses efekti / vokalizasyon metni (<= 2 karakter)
    is_short_vocalization = (len(norm_txt) <= 2)

    return bool(is_large_glyph and is_short_vocalization)


def _is_multi_signal_non_text_noise(region: Region, norm_txt: str) -> bool:
    """Birincil OCR ve geometri kanıtları birlikte zayıf olan sahte çizim tespitlerini saptar.
    
    Yalnızca UNKNOWN tipli ve tek değişkene dayanmayan çoklu kanıt durumunda True döner:
    1. Birincil OCR boştur / alfanümerik metin bulamamıştır (conf == 0.0 veya primary_empty).
    2. İkincil verifier tekil karakter halüsinasyonu üretmiş ve CTD metin geometrisi zayıftır.
    """
    if region.type != RegionType.UNKNOWN:
        return False

    meta = region.metadata if isinstance(region.metadata, dict) else {}
    val = meta.get("region_validity", {}) if isinstance(meta.get("region_validity"), dict) else {}
    verdict = meta.get("ocr_verdict", {}) if isinstance(meta.get("ocr_verdict"), dict) else {}
    repair = meta.get("repair_eligibility", {}) if isinstance(meta.get("repair_eligibility"), dict) else {}

    # Sinyal 1: Birincil OCR boş veya alfanümerik tespit yok
    is_primary_empty = (
        region.ocr_confidence == 0.0
        or val.get("primary_alnum_count", 1) == 0
        or verdict.get("reason") == "primary_empty_verifier_filled"
    )

    # Sinyal 2: Verifier tekil karakter veya zayıf metin geometrisi
    is_verifier_weak_or_hallucinated = (
        repair.get("reason") == "verifier_only_weak_text_geometry"
        or (len(norm_txt) <= 1 and verdict.get("reason") == "primary_empty_verifier_filled")
    )

    return bool(is_primary_empty and is_verifier_weak_or_hallucinated)


def _replace_region_status(
    region: Region,
    reg_type: RegionType,
    status: RegionStatus,
    reason: str,
) -> Region:
    metadata = dict(region.metadata)
    validity = metadata.get("region_validity")
    if isinstance(validity, dict) and validity.get("valid") is False:
        metadata["classification_diagnostic"] = {
            "proposed_type": reg_type.value,
            "proposed_status": status.value,
            "proposed_reason": reason,
        }
        # A strong pre-repair validity rejection outranks the later heuristic
        # classifier.  Only an explicit recovery stage may replace this state.
        validity_reason = str(validity.get("reason") or region.review_reason or "region_validity_rejected")
        return Region(
            id=region.id,
            global_bbox=region.global_bbox,
            type=region.type,
            detection_confidence=region.detection_confidence,
            source_window_ids=region.source_window_ids,
            status=RegionStatus.SKIP,
            text=region.text,
            ocr_confidence=region.ocr_confidence,
            translation=region.translation,
            review_reason=validity_reason,
            metadata=metadata,
        )

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
        metadata=metadata,
    )
