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


# Structural suspicion helper constants for Latin English text
_INVALID_START_CONSONANTS = {
    "BZ", "CD", "FJ", "FGM", "GZ", "HG", "JK", "JV", "JX", "JZ",
    "KG", "KJ", "KP", "KQ", "KV", "KX", "KZ",
    "LB", "LC", "LD", "LF", "LG", "LK", "LM", "LN", "LP", "LQ", "LR", "LS", "LT", "LV", "LX", "LZ",
    "MD", "MG", "MJ", "MK", "ML", "MQ", "MR", "MS", "MT", "MV", "MX", "MZ",
    "PB", "PD", "PF", "PJ", "PK", "PM", "PN", "PQ", "PV", "PX", "PZ",
    "QA", "QC", "QD", "QE", "QF", "QG", "QH", "QI", "QJ", "QK", "QL", "QM", "QN", "QO", "QP", "QR", "QS", "QT", "QV", "QW", "QX", "QY", "QZ",
    "TD", "TG", "TJ", "TK", "TL", "TM", "TN", "TP", "TQ", "TV", "TX", "TZ",
    "VB", "VC", "VD", "VF", "VG", "VH", "VJ", "VK", "VL", "VM", "VN", "VP", "VQ", "VR", "VS", "VT", "VV", "VW", "VX", "VY", "VZ",
    "WB", "WC", "WD", "WF", "WG", "WJ", "WK", "WL", "WM", "WN", "WP", "WQ", "WV", "WX", "WZ",
    "XB", "XC", "XD", "XF", "XG", "XH", "XI", "XJ", "XK", "XL", "XM", "XN", "XO", "XP", "XQ", "XR", "XS", "XT", "XU", "XV", "XW", "XX", "XY", "XZ",
    "ZB", "ZC", "ZD", "ZF", "ZG", "ZH", "ZJ", "ZK", "ZL", "ZM", "ZN", "ZP", "ZQ", "ZR", "ZS", "ZT", "ZV", "ZW", "ZX", "ZY", "ZZ",
}
_INVALID_CLUSTER_RE = re.compile(r"(KTTON|FPT|FKT|GKT|BKT|DKT|PKT|TKT|VKD|ZKD|XKD|QKP|QXK|ZXP)", re.IGNORECASE)
_VOWELS = set("AEIOUYaeiouy")


def _is_structurally_suspicious_latin(text: str) -> tuple[bool, str | None]:
    """Yapısal olarak bozuk Latin İngilizce metin tespiti (genel kurallar)."""
    if not text or not text.strip():
        return False, None

    norm = text.strip()

    if _contains_cjk(norm):
        return True, "cjk"

    if _obvious_repetition(norm):
        return True, "repetition"

    if _is_gibberish(norm):
        return True, "gibberish"

    raw_tokens = norm.split()
    if not raw_tokens:
        return False, None

    for raw in raw_tokens:
        clean_word = re.sub(r"^[^\w]+|[^\w]+$", "", raw)
        if not clean_word:
            continue

        # Fused digit inside alpha word (e.g., "TLUOE4", "WOEOOTAD4", "EXP1S")
        if re.search(r"[A-Za-z]+[0-9]+[A-Za-z]*", clean_word) or re.search(r"[0-9]+[A-Za-z]{3,}", clean_word):
            if not re.match(r"^(1st|2nd|3rd|[0-9]+th)$", clean_word, re.IGNORECASE):
                return True, f"digit_fused_word:{raw}"

        alpha_part = re.sub(r"[^A-Za-z]", "", clean_word)
        if len(alpha_part) >= 3:
            alpha_upper = alpha_part.upper()

            # Invalid start consonants (e.g., "CDANTED")
            if len(alpha_upper) >= 4 and alpha_upper[:2] in _INVALID_START_CONSONANTS:
                return True, f"invalid_start_consonants:{raw}"

            # Invalid internal consonant cluster (e.g., "KTTON" in "APOKTTON")
            if _INVALID_CLUSTER_RE.search(alpha_upper):
                return True, f"invalid_consonant_cluster:{raw}"

            # 4+ letter token with NO vowels
            if len(alpha_upper) >= 4 and not any(c in _VOWELS for c in alpha_upper):
                if not re.match(r"^(MC|TV|PR|DR|MR|MRS|MS|VS|OK|HTML|HTTP)$", alpha_upper):
                    return True, f"no_vowels:{raw}"

            # Unusually long single token without hyphen (spacing corruption, e.g., "CRAPTEDWEAPONS")
            if len(alpha_upper) >= 14 and "-" not in raw and "'" not in raw:
                return True, f"concatenated_token:{raw}"

    return False, None


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


def should_run_verifier(
    primary: OCRResult,
    region: Any = None,
    min_confidence: float = 0.92,
) -> tuple[bool, str]:
    """Koşullu Doğrulama (Gated Verifier) kontrolü.

    Primary OCR sonucu yüksek güvenli (>= 0.92), temiz ve yapısal olarak
    kusursuz ise ikincil ağır verifier model çağrısını güvenle atlar.

    Returns:
        (should_run: bool, reason: str)
    """
    raw = primary.raw_text or primary.text or ""
    text = normalize_ocr_text(raw).strip()

    if not text:
        return True, "primary_empty"

    conf = primary.confidence
    if conf is None or conf < min_confidence:
        conf_str = f"{conf:.3f}" if conf is not None else "none"
        return True, f"low_confidence:{conf_str}"

    if "low_ocr_confidence" in primary.warnings:
        return True, "low_confidence_warning"

    is_susp, susp_reason = _is_structurally_suspicious_latin(text)
    if is_susp:
        return True, f"suspicious:{susp_reason}"

    if not any(c.isalnum() for c in text):
        return True, "no_alphanumeric"

    if region is not None:
        review_reason = getattr(region, "review_reason", None)
        if review_reason == "ambiguous_cjk_review":
            return True, "ambiguous_cjk_review"

    return False, "high_confidence_clean_primary"


def decide_ocr_agreement(
    primary: OCRResult,
    verifier: OCRResult | None = None,
) -> OCRVerdict:
    """Primary ve opsiyonel Verifier OCR sonuçlarını karşılaştırır.

    Verifier `None` ise yalnızca Primary OCR sonucunu yapısal olarak doğrular.
    Confidence skoru tek başına AUTO yapmaz; CJK, gibberish, repetition ve structural suspicion kontrolleri uygulanır.
    """
    p_raw = primary.raw_text or ""
    v_raw = verifier.raw_text or "" if verifier is not None else ""
    p_text = normalize_ocr_text(p_raw)
    v_text = normalize_ocr_text(v_raw) if verifier is not None else ""

    p_suspicious, p_susp_reason = _is_structurally_suspicious_latin(p_text)
    v_suspicious, v_susp_reason = _is_structurally_suspicious_latin(v_text) if verifier is not None else (False, None)

    # Single-pass evaluation when verifier is None
    if verifier is None:
        if not p_text:
            return _make_verdict(
                accepted_text=None,
                accepted_raw=None,
                source="primary",
                requires_review=True,
                needs_repair=True,
                reason="primary_empty",
                provisional_text=None,
                p_raw=p_raw,
                v_raw="",
                p_conf=primary.confidence,
                v_conf=None,
            )

        if p_suspicious:
            return _make_verdict(
                accepted_text=None,
                accepted_raw=None,
                source="primary",
                requires_review=True,
                needs_repair=True,
                reason="primary_" + (p_susp_reason or "suspicious"),
                provisional_text=p_text,
                p_raw=p_raw,
                v_raw="",
                p_conf=primary.confidence,
                v_conf=None,
            )

        if "low_ocr_confidence" in primary.warnings:
            return _make_verdict(
                accepted_text=None,
                accepted_raw=None,
                source="primary",
                requires_review=True,
                needs_repair=True,
                reason="low_ocr_confidence",
                provisional_text=p_text,
                p_raw=p_raw,
                v_raw="",
                p_conf=primary.confidence,
                v_conf=None,
            )

        return _make_verdict(
            accepted_text=p_text,
            accepted_raw=p_raw,
            source="primary",
            requires_review=False,
            needs_repair=False,
            reason=None,
            provisional_text=None,
            p_raw=p_raw,
            v_raw="",
            p_conf=primary.confidence,
            v_conf=None,
        )

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

    # 4d. Primary'de structural suspicion / CJK / repetition / gibberish → otomatik seçim YOK, repair
    if p_suspicious:
        prov_text = v_text if (v_text and not v_suspicious) else p_text
        return _make_verdict(
            accepted_text=None,
            accepted_raw=None,
            source="primary",
            requires_review=True,
            needs_repair=True,
            reason="primary_" + (p_susp_reason or "suspicious"),
            provisional_text=prov_text,
            p_raw=p_raw,
            v_raw=v_raw,
            p_conf=primary.confidence,
            v_conf=verifier.confidence,
        )

    # 4e. Verifier'da structural suspicion / CJK / repetition / gibberish → otomatik seçim YOK, repair
    if v_suspicious:
        prov_text = p_text if (p_text and not p_suspicious) else v_text
        return _make_verdict(
            accepted_text=None,
            accepted_raw=None,
            source="primary",
            requires_review=True,
            needs_repair=True,
            reason="verifier_" + (v_susp_reason or "suspicious"),
            provisional_text=prov_text,
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