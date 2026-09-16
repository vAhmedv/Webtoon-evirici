"""IS 4.1 — Kör çeviri setini dondur (bir kez yazılır, bir daha değişmez).

KAYNAK: audit_output/real_chapter1_e2e/analysis/regions.json (üretim çıktısı).
SEÇİM: çevrilmiş bloklardan tabakalı 30 balon:
  short (≤25 karakter + !/?/…), long (≥120 karakter),
  term (TAM-BÜYÜK token), multi (çok üyeli grup).
ÇIKTI: benchmark/translation_ab_v1.json [{block_id, source, strata}].
Deterministik (blok-id sıralı); mevcut dosya varsa --force olmadan çıkılır.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUDIT_REGIONS = ROOT / "audit_output" / "real_chapter1_e2e" / "analysis" / "regions.json"
OUT_PATH = ROOT / "benchmark" / "translation_ab_v1.json"

N_SHORT, N_LONG, N_TERM = 8, 8, 8
TOTAL = 30
_TERM_RE = re.compile(r"\b[A-Z][A-Z0-9&'\-]{3,}\b")


def _strata(source: str, member_count: int) -> list[str]:
    # Not: bölüm gerçeği (ch1: maks 108 karakter, çok-üyeli çevrilmiş blok
    # yok) — "long" = bölümün en uzunları, "multi" = tanımsal, doldurmada
    # kullanılmaz. Eşikler mutlak değil, bölüme göreli.
    s: list[str] = []
    if len(source) <= 25 and re.search(r"[!?…]|[.]{3}", source):
        s.append("short")
    if _TERM_RE.search(source):
        s.append("term")
    if member_count > 1:
        s.append("multi")
    return s or ["other"]


def main() -> None:
    force = "--force" in sys.argv[1:]
    if OUT_PATH.exists() and not force:
        print(f"[AB-FREEZE] mevcut (değişmez): {OUT_PATH} — çıkılıyor (--force ile ezilir)")
        return
    assert AUDIT_REGIONS.is_file(), f"Audit regions.json yok: {AUDIT_REGIONS}"
    data = json.loads(AUDIT_REGIONS.read_text(encoding="utf-8"))
    blocks = [
        b for b in data.get("text_blocks", [])
        if (b.get("source_text") or "").strip() and (b.get("translation") or "").strip()
    ]
    assert len(blocks) >= TOTAL, f"Yetersiz çevrilmiş blok: {len(blocks)}"
    enriched = [
        {
            "block_id": b["id"],
            "source": " ".join((b["source_text"] or "").split()),
            "strata": _strata((b["source_text"] or "").strip(), len(b.get("member_ids", []))),
        }
        for b in sorted(blocks, key=lambda x: x["id"])
    ]
    picked: list[dict] = []
    used: set[int] = set()

    def _take(cat: str, n: int) -> None:
        for e in enriched:
            if len([p for p in picked if cat in p["strata"]]) >= n:
                break
            if e["block_id"] in used or cat not in e["strata"]:
                continue
            picked.append(e)
            used.add(e["block_id"])

    _take("short", N_SHORT)
    _take("term", N_TERM)
    # "long": bölümün en uzunlarından (mutlak eşik yok — ch1'de 120+ yok).
    for e in sorted(
        (x for x in enriched if x["block_id"] not in used),
        key=lambda x: (-len(x["source"]), x["block_id"]),
    ):
        if len([p for p in picked if "long" in p["strata"]]) >= N_LONG:
            break
        e["strata"].append("long")
        picked.append(e)
        used.add(e["block_id"])
    for e in enriched:  # eksik kalırsa blok-id sırasıyla doldur
        if len(picked) >= TOTAL:
            break
        if e["block_id"] not in used:
            picked.append(e)
            used.add(e["block_id"])
    picked.sort(key=lambda e: e["block_id"])
    payload = {
        "name": "translation_ab_v1",
        "created": date.today().isoformat(),
        "source_audit": str(AUDIT_REGIONS.relative_to(ROOT)),
        "source_blocks_total": len(blocks),
        "items": picked,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    cats: dict[str, int] = {}
    for e in picked:
        for c in e["strata"]:
            cats[c] = cats.get(c, 0) + 1
    print(f"[AB-FREEZE] {len(picked)} balon -> {OUT_PATH} | tabaka: {cats}")


if __name__ == "__main__":
    main()
