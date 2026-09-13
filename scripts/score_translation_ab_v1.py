"""IS 4.3 — Kör sıralama paketi: otomatik skorlar + insan pusulası.

KULLANIM:
    .venv\\Scripts\\python.exe scripts/score_translation_ab_v1.py <sonuc1.json> <sonuc2.json>

OTOMATİK (4.3a): boş-oran, yankı-oranı (kaynakla aynı), TR karakter varlığı.
İNSAN (4.3b): modeller A/B olarak karıştırılır (tohumlu, anahtar ayrı dosyada),
  `ballot_<stamp>.md` içinde 30 çift + oy kutuları üretilir.
KARAR (4.4): oylar sayılınca ROADMAP Faz 6 satırına işlenir.
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmark" / "results" / "translation_ab_v1"


def _auto(results: list[dict]) -> dict:
    n = max(1, len(results))
    empty = sum(1 for r in results if not (r.get("translation") or "").strip())
    echo = sum(
        1
        for r in results
        if (r.get("translation") or "").strip().casefold() == (r.get("source") or "").strip().casefold()
    )
    tr_chars = sum(1 for r in results if any(c in (r.get("translation") or "") for c in "ğĞşıŞçÇöÖüÜ"))
    avg_ratio = sum(len(r.get("translation") or "") / max(1, len(r.get("source") or "")) for r in results) / n
    return {"n": n, "empty": empty, "echo": echo, "tr_chars": tr_chars, "avg_len_ratio": round(avg_ratio, 2)}


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Kullanım: score_translation_ab_v1.py <sonuc1.json> <sonuc2.json>")
    a = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    b = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    assert [r["block_id"] for r in a["results"]] == [r["block_id"] for r in b["results"]], "Kör setler eşleşmiyor!"

    print(f"[AB-SCORE] {a['model']}: {_auto(a['results'])}")
    print(f"[AB-SCORE] {b['model']}: {_auto(b['results'])}")

    rng = random.Random(20260912)
    swap = rng.random() < 0.5
    label_a, label_b = ("B", "A") if swap else ("A", "B")
    key = {"A": b["model"] if swap else a["model"], "B": a["model"] if swap else b["model"]}
    by_model = {a["model"]: a["results"], b["model"]: b["results"]}
    res_a, res_b = by_model[key["A"]], by_model[key["B"]]

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    ballot = [f"# Kör çeviri oylaması — translation_ab_v1 ({stamp})", "",
              "Her çiftte daha iyi Türkçeyi seçin (anlam + doğallık + terim). "
              "Kimlikler gizli; anahtar `ballot key` dosyasında.", ""]
    for ra, rb in zip(res_a, res_b):
        ballot += [f"## Blok {ra['block_id']} — EN: {ra['source']}",
                   f"- **A:** {ra['translation']}", f"- **B:** {rb['translation']}",
                   "- [ ] A  /  [ ] B  /  [ ] Berabere", ""]
    (OUT_DIR / f"ballot_{stamp}.md").write_text("\n".join(ballot), encoding="utf-8")
    key_path = OUT_DIR / f"ballot_{stamp}.key.json"
    key_path.write_text(json.dumps({"key": key, "files": [sys.argv[1], sys.argv[2]]}, indent=2), encoding="utf-8")
    print(f"[AB-BALLOT] pusula -> {OUT_DIR / f'ballot_{stamp}.md'} (A={key['A']}, B={key['B']} anahtarı ayrı dosyada)")


if __name__ == "__main__":
    main()
