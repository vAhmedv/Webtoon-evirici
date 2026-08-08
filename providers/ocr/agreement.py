"""OCR agreement/disagreement katmanı.

Primary = PaddleOCR-VL-1.6, verifier = PaddleOCR English v5.

Kurallar:
- Safe agreement (yalnız whitespace/newline/case farkı): VL kabul,
  requires_review=False, needs_repair=False.
- Gerçek lexical disagreement: otomatik olarak Paddle veya VL doğru ilan
  EDİLMEZ. provisional_text kullanılır, requires_review=True, needs_repair=True.
- Critical name disagreement: aynı şekilde otomatik seçim YAPILMAZ,
  requires_review=True, needs_repair=True.
- İki engine'in raw/normalized sonucu metadata'da korunur.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from core.ocr_normalizer import normalize_ocr_text
from providers.ocr.base import OCRResult

# CJK karakter aralıkları
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")

# Kritik özel isimler (manhwa bağlamı) — disagreement'da öncelikli
CRITICAL_NAMES = [
    "LUO TIAN",
    "HU SAN",
    "GAO YUAN",
    "CAPTAIN GAO",
    "YOUNG MASTER YU",
    "BLACKWIND RAVINE",
    "BLACKSTONE",
]


@dataclass(frozen=True)
class OCRVerdict:
    """Agreement/disagreement sonucu.

    Attributes:
        accepted_text: Kabul edilen canonical metin (safe agreement'da VL).
            Disagreement durumunda None olabilir; provisional_text kullanılır.
        accepted_raw: Kabul edilen ham metin.
        provisional_text: Disagreement durumunda geçici metin (repair girdisi).
            Otomatik doğru ilan edilmez.
        repaired_text: OCR repair uygulanmış son metin (varsa).
        source: "primary" (VL-1.6) veya "verifier" (Paddle v5).
        requires_review: İnceleme gerekli mi.
        needs_repair: OCR repair gerekli mi.
        reason: İnceleme/repair nedeni.
        primary_raw: VL-1.6 ham metni.
        verifier_raw: Paddle v5 ham metni.
        primary_normalized: VL-1.6 normalized metni.
        verifier_normalized: Paddle v5 normalized metni.
        primary_confidence: VL-1.6 confidence (None olabilir).
        verifier_confidence: Paddle v5 confidence.
    """

    accepted_text: str | None
    accepted_raw: str | None
    source: str
    requires_review: bool = False
    needs_repair: bool = False
    reason: str | None = None
    provisional_text: str | None = None
    repaired_text: str | None = None
    primary_raw: str = ""
    verifier_raw: str = ""
    primary_normalized: str = ""
    verifier_normalized: str = ""
    primary_confidence: float | None = None
    verifier_confidence: float | None = None


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _obvious_repetition(text: str, max_reps: int = 3) -> bool:
    """Aynı token'ın >= max_reps kez tekrarını tespit eder."""
    if not text:
        return False
    tokens = normalize_ocr_text(text).upper().split()
    for tok in set(tokens):
        if len(tok) < 2:
            continue
        if tokens.count(tok) >= max_reps:
            return True
    return False


def _is_gibberish(text: str) -> bool:
    """Anlamsız/gibberish metin tespiti (basit heuristik)."""
    if not text:
        return False
    norm = normalize_ocr_text(text)
    if len(norm) < 3:
        return False
    chars = set(norm.upper())
    if len(chars) <= 2 and len(norm) >= 5:
        return True
    return False


def _critical_name_mismatch(primary: str, verifier: str) -> str | None:
    """Kritik özel isim farkını tespit eder.

    Returns:
        Fark olan isim veya None.
    """
    p_norm = normalize_ocr_text(primary).upper()
    v_norm = normalize_ocr_text(verifier).upper()
    for name in CRITICAL_NAMES:
        p_has = name in p_norm
        v_has = name in v_norm
        if p_has != v_has:
            return name
    return None


def _word_level_similarity(a: str, b: str) -> float:
    """Kelime seti bazlı benzerlik (0.0-1.0)."""
    a_words = set(normalize_ocr_text(a).upper().split())
    b_words = set(normalize_ocr_text(b).upper().split())
    if not a_words and not b_words:
        return 1.0
    if not a_words or not b_words:
        return 0.0
    inter = a_words & b_words
    union = a_words | b_words
    return len(inter) / len(union)


def _format_only_difference(a: str, b: str) -> bool:
    """Sadece case/whitespace/punctuation-format farkı var mı?"""
    if _word_level_similarity(a, b) < 0.95:
        return False
    sim = SequenceMatcher(
        None,
        normalize_ocr_text(a).upper(),
        normalize_ocr_text(b).upper(),
    ).ratio()
    return sim >= 0.90


def _make_verdict(
    *,
    accepted_text: str | None,
    accepted_raw: str | None,
    source: str,
    requires_review: bool,
    needs_repair: bool,
    reason: str | None,
    provisional_text: str | None,
    repaired_text: str | None = None,
    p_raw: str,
    v_raw: str,
    p_conf: float | None,
    v_conf: float | None,
) -> OCRVerdict:
    return OCRVerdict(
        accepted_text=accepted_text,
        accepted_raw=accepted_raw,
        source=source,
        requires_review=requires_review,
        needs_repair=needs_repair,
        reason=reason,
        provisional_text=provisional_text,
        repaired_text=repaired_text,
        primary_raw=p_raw,
        verifier_raw=v_raw,
        primary_normalized=normalize_ocr_text(p_raw),
        verifier_normalized=normalize_ocr_text(v_raw),
        primary_confidence=p_conf,
        verifier_confidence=v_conf,
    )


def decide_ocr_agreement(
    primary: OCRResult,
    verifier: OCRResult,
) -> OCRVerdict:
    """VL-1.6 (primary) ve Paddle v5 (verifier) sonuçlarını karşılaştırır."""
    p_raw = primary.raw_text or ""
    v_raw = verifier.raw_text or ""
    p_text = normalize_ocr_text(p_raw)
    v_text = normalize_ocr_text(v_raw)

    # 1. İkisi de boş → repair gerekli
    if not p_text and not v_text:
        return _make_verdict(
            accepted_text=None,
            accepted_raw=None,
            source="primary",
            requires_review=True,
            needs_repair=True,
            reason="both_empty",
            provisional_text=None,
            p_raw=p_raw,
            v_raw=v_raw,
            p_conf=primary.confidence,
            v_conf=verifier.confidence,
        )

    # 2. Primary boş, verifier dolu → otomatik seçim yok, repair gerekli
    if not p_text and v_text:
        return _make_verdict(
            accepted_text=None,
            accepted_raw=None,
            source="primary",
            requires_review=True,
            needs_repair=True,
            reason="primary_empty_verifier_filled",
            provisional_text=v_text,
            p_raw=p_raw,
            v_raw=v_raw,
            p_conf=primary.confidence,
            v_conf=verifier.confidence,
        )

    # 3. Verifier boş, primary dolu → otomatik seçim yok, repair gerekli
    if p_text and not v_text:
        return _make_verdict(
            accepted_text=None,
            accepted_raw=None,
            source="primary",
            requires_review=True,
            needs_repair=True,
            reason="verifier_empty_primary_filled",
            provisional_text=p_text,
            p_raw=p_raw,
            v_raw=v_raw,
            p_conf=primary.confidence,
            v_conf=verifier.confidence,
        )

    # 4. İkisi de dolu
    # 4a. Birebir aynı → VL kabul, temiz
    if p_text == v_text:
        return _make_verdict(
            accepted_text=p_text,
            accepted_raw=p_raw,
            source="primary",
            requires_review=False,
            needs_repair=False,
            reason=None,
            provisional_text=None,
            p_raw=p_raw,
            v_raw=v_raw,
            p_conf=primary.confidence,
            v_conf=verifier.confidence,
        )

    # 4b. Sadece format farkı (case/whitespace/punctuation) → VL kabul, temiz
    if _format_only_difference(p_text, v_text):
        return _make_verdict(
            accepted_text=p_text,
            accepted_raw=p_raw,
            source="primary",
            requires_review=False,
            needs_repair=False,
            reason=None,
            provisional_text=None,
            p_raw=p_raw,
            v_raw=v_raw,
            p_conf=primary.confidence,
            v_conf=verifier.confidence,
        )

    # 4c. Kritik özel isim farkı → otomatik seçim YOK, repair gerekli
    name_mismatch = _critical_name_mismatch(p_text, v_text)
    if name_mismatch is not None:
        return _make_verdict(
            accepted_text=None,
            accepted_raw=None,
            source="primary",
            requires_review=True,
            needs_repair=True,
            reason=f"critical_name_mismatch:{name_mismatch}",
            provisional_text=p_text,
            p_raw=p_raw,
            v_raw=v_raw,
            p_conf=primary.confidence,
            v_conf=verifier.confidence,
        )

    # 4d. Primary'de CJK / repetition / gibberish → otomatik seçim YOK, repair
    p_problematic = (
        _contains_cjk(p_text)
        or _obvious_repetition(p_text)
        or _is_gibberish(p_text)
    )
    if p_problematic:
        reason_parts = []
        if _contains_cjk(p_text):
            reason_parts.append("cjk")
        if _obvious_repetition(p_text):
            reason_parts.append("repetition")
        if _is_gibberish(p_text):
            reason_parts.append("gibberish")
        return _make_verdict(
            accepted_text=None,
            accepted_raw=None,
            source="primary",
            requires_review=True,
            needs_repair=True,
            reason="primary_" + "_".join(reason_parts),
            provisional_text=v_text,
            p_raw=p_raw,
            v_raw=v_raw,
            p_conf=primary.confidence,
            v_conf=verifier.confidence,
        )

    # 4e. Verifier'da CJK / repetition / gibberish → otomatik seçim YOK, repair
    v_problematic = (
        _contains_cjk(v_text)
        or _obvious_repetition(v_text)
        or _is_gibberish(v_text)
    )
    if v_problematic:
        reason_parts = []
        if _contains_cjk(v_text):
            reason_parts.append("cjk")
        if _obvious_repetition(v_text):
            reason_parts.append("repetition")
        if _is_gibberish(v_text):
            reason_parts.append("gibberish")
        return _make_verdict(
            accepted_text=None,
            accepted_raw=None,
            source="primary",
            requires_review=True,
            needs_repair=True,
            reason="verifier_" + "_".join(reason_parts),
            provisional_text=p_text,
            p_raw=p_raw,
            v_raw=v_raw,
            p_conf=primary.confidence,
            v_conf=verifier.confidence,
        )

    # 4f. Gerçek lexical disagreement → otomatik seçim YOK, repair gerekli
    return _make_verdict(
        accepted_text=None,
        accepted_raw=None,
        source="primary",
        requires_review=True,
        needs_repair=True,
        reason="word_difference",
        provisional_text=p_text,
        p_raw=p_raw,
        v_raw=v_raw,
        p_conf=primary.confidence,
        v_conf=verifier.confidence,
    )