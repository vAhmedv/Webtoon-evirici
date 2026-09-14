"""Terazi karsilastirma: iki dondurulmus golden dosyayi karsilastirir.

KULLANIM:
    .venv\\Scripts\\python.exe scripts\\compare_golden_v1.py benchmark/golden_baseline_v1.json benchmark/golden_f3_v1.json

Fren kurallari (F4-b): overflow != 0 -> ALARM; yankı-nötr düzeltilmiş
cevrilme-orani ((cevrilmis)/(uygun - yankı)) 5+ puan duserse -> UYARI;
kısa-hikaye-sayaci +3 ve üstü artarsa -> UYARI (hikaye kaybı).
Yankı bloklar (TR==kaynak) zaten Türkçe olmayacaktı — oran dışıdır;
review artisi tek basina alarm degildir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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


def main() -> None:
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
        if (nch.get("overflow_blocks_count") or 0) != 0:
            alarms.append(f"{tag}: OVERFLOW!")
        if o_rate is not None and n_rate is not None and n_rate < o_rate - 0.05:
            alarms.append(f"{tag}: cevrilme-orani {o_rate}->{n_rate} (5+ puan dusus)")
        o_short = och.get("short_dialogue_untranslated_count") or 0
        n_short = nch.get("short_dialogue_untranslated_count") or 0
        if n_short - o_short >= 3:
            print(f"  UYARI: kisa-hikaye {o_short}->{n_short} (temizlenemeyen kisalar artti)")
    print()
    if alarms:
        print("ALARM:")
        for a in alarms:
            print(f"  !! {a}")
    else:
        print("Fren: temiz (overflow yok, cevrilme-orani 5+ puan dusmedi).")


if __name__ == "__main__":
    main()
