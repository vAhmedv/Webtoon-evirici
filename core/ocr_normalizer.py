"""Güvenli OCR metin normalizasyonu.

Yalnızca güvenli whitespace işlemleri yapar:
- newline → space
- fazla whitespace → tek space
- trim
- Unicode whitespace normalization

Kelime düzeltme, spell correction, case değiştirme, isim düzeltme YAPMAZ.
"""

from __future__ import annotations

import re
import unicodedata

# Unicode whitespace karakterleri (space, tab, newline, vb.)
_WS_RE = re.compile(r"\s+")


def normalize_ocr_text(raw: str) -> str:
    """Ham OCR metnini güvenli canonical forma çevirir.

    Args:
        raw: Modelin ürettiği ham metin (newline'lar korunabilir).

    Returns:
        Canonical metin: newline'lar space'e çevrilir, fazla whitespace
        tek space'e indirilir, baş/son boşluk kırpılır.
    """
    if not raw:
        return ""
    # Unicode whitespace normalization (NBSP, vb. → normal space)
    normalized = unicodedata.normalize("NFKC", raw)
    # Tüm whitespace dizilerini tek space'e indir
    collapsed = _WS_RE.sub(" ", normalized)
    return collapsed.strip()