"""System and Game UI Text Domain Handling Module.

Provides deterministic translation for webtoon game/system text (e.g. system windows,
skill popups, title notifications) while respecting approved glossary overrides.
"""
from __future__ import annotations

import re

SYSTEM_UI_LEXICON: dict[str, str] = {
    "PASSIVE SKILL ACQUIRED": "Kazanılan Pasif Yetenek",
    "PASSIVE SKILL": "Pasif Yetenek",
    "ACTIVE SKILL": "Aktif Yetenek",
    "TITLE ACQUIRED": "Kazanılan Unvan",
    "TITLE": "Unvan",
    "CLASS ADVANCEMENT AVAILABLE": "Sınıf Gelişimi Mevcut",
    "CLASS ADVANCEMENT": "Sınıf Gelişimi",
    "CLASS": "Sınıf",
    "UNIQUE TRAIT": "Benzersiz Özellik",
    "CONDITION FAILED": "Koşul Sağlanmadı",
    "INSUFFICIENT MANA": "Yetersiz Mana",
    "ABILITY COOLDOWN": "Yetenek Bekleme Süresi",
    "SKILL ACTIVATION FAILED": "Yetenek Etkinleştirmesi Başarısız Oldu",
    "SKILL ACTIVATION": "Yetenek Etkinleştirme",
    "COOLDOWN": "Bekleme Süresi",
    "ACQUIRED": "Kazanıldı",
    "AVAILABLE": "Mevcut",
}


def is_system_ui_line(text: str) -> bool:
    """Check if text line matches webtoon system/UI text patterns."""
    clean = text.strip()
    upper = clean.upper()

    if upper in SYSTEM_UI_LEXICON:
        return True

    prefixes = [
        "PASSIVE SKILL ACQUIRED:",
        "PASSIVE SKILL:",
        "ACTIVE SKILL:",
        "TITLE ACQUIRED:",
        "TITLE:",
        "CLASS ADVANCEMENT:",
        "UNIQUE TRAIT:",
        "CONDITION FAILED:",
        "INSUFFICIENT MANA:",
        "ABILITY COOLDOWN:",
        "SKILL ACTIVATION FAILED:",
    ]
    for p in prefixes:
        if upper.startswith(p):
            return True

    if re.search(r"^(ability|skill)?\s*cooldown:\s*\d+\s*(seconds?|sec|s)?\.?$", clean, re.IGNORECASE):
        return True

    return False


def translate_system_ui_line(text: str, approved_terms: dict[str, str] | None = None) -> str | None:
    """Attempt deterministic translation for system/UI text lines.

    Respects approved terminology overrides when translating inner values.
    """
    clean = text.strip()
    upper = clean.upper()
    app_t = approved_terms or {}

    # 1. Exact phrase match
    if upper in SYSTEM_UI_LEXICON:
        return SYSTEM_UI_LEXICON[upper]

    # 2. Cooldown regex pattern e.g. "ABILITY COOLDOWN: 24 SECONDS"
    cooldown_match = re.match(
        r"^(ability|skill)?\s*cooldown:\s*(\d+)\s*(seconds?|sec|s)?\.?$",
        clean,
        re.IGNORECASE,
    )
    if cooldown_match:
        prefix_type = cooldown_match.group(1)
        seconds = cooldown_match.group(2)
        if prefix_type:
            label = "Yetenek Bekleme Süresi"
        else:
            label = "Bekleme Süresi"
        return f"{label}: {seconds} saniye"

    # 3. Key: Value system text patterns e.g. "PASSIVE SKILL ACQUIRED: ECHO SENSE"
    if ":" in clean:
        key_part, val_part = clean.split(":", 1)
        key_upper = key_part.strip().upper()
        val_clean = val_part.strip()
        val_upper = val_clean.upper()

        if key_upper in SYSTEM_UI_LEXICON:
            tr_key = SYSTEM_UI_LEXICON[key_upper]

            # Check if inner value has an explicit approved term
            if val_upper in app_t:
                tr_val = app_t[val_upper]
                return f"{tr_key}: {tr_val}"

            # Check if inner value is in UI lexicon
            if val_upper in SYSTEM_UI_LEXICON:
                tr_val = SYSTEM_UI_LEXICON[val_upper]
                return f"{tr_key}: {tr_val}"

            # Preserve value if unknown/unapproved
            return f"{tr_key}: {val_clean}"

    return None
