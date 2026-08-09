"""System and Game UI Text Domain Handling Module.

Provides deterministic translation for webtoon game/system text (e.g. system windows,
skill popups, title notifications) without altering normal conversational dialogue.
"""
from __future__ import annotations

import re

SYSTEM_UI_LEXICON: dict[str, str] = {
    "PASSIVE SKILL ACQUIRED": "Kazanılan Pasif Beceri",
    "PASSIVE SKILL": "Pasif Beceri",
    "ACTIVE SKILL": "Aktif Beceri",
    "TITLE ACQUIRED": "Kazanılan Unvan",
    "TITLE": "Unvan",
    "CLASS ADVANCEMENT AVAILABLE": "Sınıf Yükseltmesi Mevcut",
    "CLASS ADVANCEMENT": "Sınıf Yükseltmesi",
    "CLASS": "Sınıf",
    "UNIQUE TRAIT": "Benzersiz Özellik",
    "CONDITION FAILED": "Koşul Sağlanmadı",
    "INSUFFICIENT MANA": "Yetersiz Mana",
    "ABILITY COOLDOWN": "Yetenek Bekleme Süresi",
    "SKILL ACTIVATION FAILED": "Beceri Etkinleştirmesi Başarısız Oldu",
    "SKILL ACTIVATION": "Beceri Etkinleştirme",
    "COOLDOWN": "Bekleme Süresi",
    "ACQUIRED": "Kazanıldı",
    "AVAILABLE": "Mevcut",
}


def is_system_ui_line(text: str) -> bool:
    """Check if text line matches webtoon system/UI text patterns."""
    clean = text.strip()
    upper = clean.upper()

    # Exact system notification matches
    if upper in SYSTEM_UI_LEXICON:
        return True

    # Prefix matches like "TITLE ACQUIRED: ...", "CONDITION FAILED: ..."
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

    # System cooldown line regex (e.g. "Ability cooldown: 18 seconds.")
    if re.search(r"^(ability|skill)?\s*cooldown:\s*\d+\s*(seconds?|sec|s)?\.?$", clean, re.IGNORECASE):
        return True

    return False


def translate_system_ui_line(text: str) -> str | None:
    """Attempt deterministic translation for system/UI text lines.

    Returns translated string if text matches UI pattern, otherwise None.
    """
    clean = text.strip()
    upper = clean.upper()

    # 1. Exact phrase match
    if upper in SYSTEM_UI_LEXICON:
        return SYSTEM_UI_LEXICON[upper]

    # 2. Cooldown regex pattern e.g. "Ability cooldown: 18 seconds."
    cooldown_match = re.match(
        r"^(ability|skill)?\s*cooldown:\s*(\d+)\s*(seconds?|sec|s)?\.?$",
        clean,
        re.IGNORECASE,
    )
    if cooldown_match:
        prefix_type = cooldown_match.group(1)
        seconds = cooldown_match.group(2)
        if prefix_type:
            p_upper = prefix_type.upper()
            label = "Yetenek Bekleme Süresi" if p_upper == "ABILITY" else "Beceri Bekleme Süresi"
        else:
            label = "Bekleme Süresi"
        return f"{label}: {seconds} saniye"

    # 3. Key: Value system text patterns e.g. "PASSIVE SKILL ACQUIRED: MANA SENSE"
    if ":" in clean:
        key_part, val_part = clean.split(":", 1)
        key_upper = key_part.strip().upper()
        val_clean = val_part.strip()

        if key_upper in SYSTEM_UI_LEXICON:
            tr_key = SYSTEM_UI_LEXICON[key_upper]

            # Translate inner system value if also in lexicon, e.g. INSUFFICIENT MANA
            if val_clean.upper() in SYSTEM_UI_LEXICON:
                tr_val = SYSTEM_UI_LEXICON[val_clean.upper()]
                return f"{tr_key}: {tr_val}"

            # Return key translated, preserving value for named term protection
            return f"{tr_key}: {val_clean}"

    return None
