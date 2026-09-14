"""Terazi karsilastirma: iki dondurulmus golden dosyayi karsilastirir.

KULLANIM:
    .venv\\Scripts\\python.exe scripts\\compare_golden_v1.py benchmark/golden_baseline_v1.json benchmark/golden_f3_v1.json

Fren kurallari (F6 kilitli): overflow != 0 -> FAIL; yankı-nötr düzeltilmiş
cevrilme-orani ((cevrilmis)/(uygun - yankı)) 5+ puan duserse -> FAIL;
kısa-hikaye-sayaci +3 ve üstü artarsa -> FAIL (hikaye kaybı);
inceleme-orani (review / toplam) 5+ puan artarsa -> FAIL (sel freni).
Yankı bloklar (TR==kaynak) zaten Türkçe olmayacaktı — oran dışıdır;
FAIL varsa çıkış kodu 1, temizse 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# F6 fren eşikleri (oranlar göreli, sayı artışı geri-bakışlı):
# - oranlar: 5 puan (0.05) — bölüm büyüklüğünden bağımsız
# - kısa: +3 — kayıp sayacı, F3/F4 ölçüsünden gelir
RATE_MAX_DROP = 0.05
REVIEW_RATE_MAX_RISE = 0.05
SHORT_MAX_RISE = 3

KEYS = [
    "text_block_count",
    "translation_eligible_blocks_count",
    "translated_blocks_count",
    "successfully_inpainted_blocks_count",
    "actually_rendered_blocks_count",
    "overflow_blocks_count",
    "short_dialogue_untranslated_count",
    "short_unprinted_count",
    "final_auto_regions",
    "final_review_regions",
    "final_skip_regions",
]


def _rate(ch: dict, num: str, den: str) -> float | None:
    d = ch.get(den) or 0
    if not d:
        return None
    return round((ch.get(num) or 0) / d, 3)


def _adj_rate(ch: dict) -> float | None:
    """Yankı-nötr çevrilme oranı: yankılar zaten Türkçe olmayacaktı."""
    den = (ch.get("translation_eligible_blocks_count") or 0) - (ch.get("echo_preserved_blocks_count") or 0)
    if den <= 0:
        return None
    return round((ch.get("translated_blocks_count") or 0) / den, 3)


def _review_rate(ch: dict) -> float | None:
    """İnceleme-oranı: review / (auto+review+skip). Bölüm boyundan bağımsız."""
    auto = ch.get("final_auto_regions") or 0
    rev = ch.get("final_review_regions") or 0
    skip = ch.get("final_skip_regions") or 0
    tot = auto + rev + skip
    if tot <= 0:
        return None
    return round(rev / tot, 3)


def compare_chapters(old: dict, new: dict) -> tuple[list[str], list[str]]:
    """Test edilebilir çekirdek: (alarmlar, kisa-artışları). Saf fonksiyon."""
    alarms: list[str] = []
    short_rises: list[str] = []
    for tag, nch in new.get("chapters", {}).items():
        och = old.get("chapters", {}).get(tag)
        if och is None or nch.get("error") or och.get("error"):
            continue
        if (nch.get("overflow_blocks_count") or 0) != 0:
            alarms.append(f"{tag}: OVERFLOW!")
        o_rate, n_rate = _adj_rate(och), _adj_rate(nch)
        if o_rate is not None and n_rate is not None and n_rate < o_rate - RATE_MAX_DROP:
            alarms.append(f"{tag}: cevrilme-orani {o_rate}->{n_rate} (5+ puan dusus)")
        o_short = och.get("short_dialogue_untranslated_count") or 0
        n_short = nch.get("short_dialogue_untranslated_count") or 0
        if n_short - o_short >= SHORT_MAX_RISE:
            short_rises.append(f"{tag}: kisa-hikaye {o_short}->{n_short}")
        o_rev, n_rev = _review_rate(och), _review_rate(nch)
        if o_rev is not None and n_rev is not None and n_rev - o_rev > REVIEW_RATE_MAX_RISE:
            alarms.append(f"{tag}: inceleme-orani {o_rev}->{n_rev} (5+ puan artis, sel)")
    return alarms, short_rises


def main() -> int:
    old = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    new = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    print(f"# {old.get('name')} -> {new.get('name')}")
    alarms: list[str] = []
    for tag, nch in new["chapters"].items():
        och = old["chapters"].get(tag)
        if och is None:
            print(f"\n## {tag}: ESKI OL CUM YOK (karsilastirilamadi)")
            continue
        if nch.get("error") or och.get("error"):
            print(f"\n## {tag}: HATALI OL CUM (atlandi)")
            continue
        print(f"\n## {tag}")
        for k in KEYS:
            o, n = och.get(k), nch.get(k)
            if o is None or n is None:
                continue
            flag = ""
            if o != n:
                flag = f"  <-- {n - o:+d}" if isinstance(n, int) and isinstance(o, int) else "  <-- degisti"
            print(f"  {k}: {o} -> {n}{flag}")
        o_rate = _adj_rate(och)
        n_rate = _adj_rate(nch)
        print(f"  cevrilme-orani (yanki-notr): {o_rate} -> {n_rate}")
        o_rev, n_rev = _review_rate(och), _review_rate(nch)
        print(f"  inceleme-orani: {o_rev} -> {n_rev}")
        if (nch.get("overflow_blocks_count") or 0) != 0:
            alarms.append(f"{tag}: OVERFLOW!")
        if o_rate is not None and n_rate is not None and n_rate < o_rate - RATE_MAX_DROP:
            alarms.append(f"{tag}: cevrilme-orani {o_rate}->{n_rate} (5+ puan dusus)")
        o_short = och.get("short_dialogue_untranslated_count") or 0
        n_short = nch.get("short_dialogue_untranslated_count") or 0
        if n_short - o_short >= SHORT_MAX_RISE:
            msg = f"{tag}: kisa-hikaye {o_short}->{n_short} (temizlenemeyen kisalar artti)"
            print(f"  FAIL: {msg}")
            alarms.append(msg)
        if o_rev is not None and n_rev is not None and n_rev - o_rev > REVIEW_RATE_MAX_RISE:
            alarms.append(f"{tag}: inceleme-orani {o_rev}->{n_rev} (5+ puan artis, sel)")
    print()
    if alarms:
        print("FAIL:")
        for a in alarms:
            print(f"  !! {a}")
        return 1
    print("Fren: temiz (overflow yok, cevrilme-orani 5+ puan dusmedi, sel yok).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
