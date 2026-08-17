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
_EDGE_NOISE_RE = re.compile(r"[\|\\\[\]\{\}\<\>_~^]")
_STRAY_SLASHES_RE = re.compile(r"(?:^|\s)[/\\]+(?:\s|$)")


def sanitize_ocr_noise(text: str) -> str:
    """Removes OCR edge artifacts, brackets, stray pipes, and garbage symbols.
    
    Preserves natural alphanumeric words and valid dialogue punctuation (! ? . ... , ' -).
    """
    if not text:
        return ""
    
    t = unicodedata.normalize("NFKC", text)
    
    # Replace bracket/pipe/tilde edge artifacts with spaces
    t = _EDGE_NOISE_RE.sub(" ", t)
    
    # Replace isolated stray slashes (e.g. ' / ' or ' /O/T ' when standalone slashes)
    t = _STRAY_SLASHES_RE.sub(" ", t)
    
    # Remove leading and trailing orphan non-alphanumeric punctuation junk (slashes, pipes, colons, lone quotes)
    t = re.sub(r"^[\s\"'`/\\:\-–—|~*#@+]+", "", t)
    t = re.sub(r"[\s\"'`/\\:\-–—|~*#@+]+$", "", t)
    
    # Collapse multiple whitespace
    t = _WS_RE.sub(" ", t).strip()
    return t


def normalize_ocr_text(raw: str) -> str:
    """Ham OCR metnini güvenli canonical forma çevirir.

    Args:
        raw: Modelin ürettiği ham metin (newline'lar korunabilir).

    Returns:
        Canonical metin: newline'lar space'e çevrilir, fazla whitespace
        tek space'e indirilir, baş/son boşluk kırpılır, kenar çöp karakterleri temizlenir.
    """
    if not raw:
        return ""
    # Unicode whitespace normalization (NBSP, vb. → normal space)
    normalized = unicodedata.normalize("NFKC", raw)
    # Tüm whitespace dizilerini tek space'e indir
    collapsed = _WS_RE.sub(" ", normalized)
    return sanitize_ocr_noise(collapsed)