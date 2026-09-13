"""IS 5.3 — Her koşuda otomatik kusur raporu (insan REVIEW turu kör olmaz).

GİRDİ: <output>/analysis/{regions,summary,glossary}.json (üretim artefaktları).
ÇIKTI: review_output/defect_report_<stamp>.md — şüpheli blok listeleri:
  REVIEW bölgeler (sebep + kaynak metin), inpaint-REVIEW blokları,
  overflow blokları, çevrilmeyen kısa balonlar, kilitsiz tekrar terimler.
Model koşusu YOK; salt rapor. Tarih damgalı, her koşuda yeniden üretilir.

KULLANIM:
    .venv\\Scripts\\python.exe scripts/write_defect_report.py audit_output/real_chapter1_e2e
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_e2e_real_chapter1 import count_short_dialogue_untranslated


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "audit_output" / "real_chapter1_e2e"
    analysis = output / "analysis"
    regions = json.loads((analysis / "regions.json").read_text(encoding="utf-8"))
    summary = json.loads((analysis / "summary.json").read_text(encoding="utf-8"))
    glossary_path = analysis / "glossary.json"
    glossary = json.loads(glossary_path.read_text(encoding="utf-8")) if glossary_path.is_file() else {}

    raw_regions: list[dict] = regions.get("regions", [])
    review_regs = [r for r in raw_regions if r.get("status") == "review"]
    inpaint_review_ids = regions.get("review_block_ids", [])
    blocks = {b["id"]: b for b in regions.get("text_blocks", [])}

    lines = [
        f"# Kusur raporu — {output.name} ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"Blok: {regions.get('text_blocks_count')} | çevrilen: {summary.get('translated_blocks_count')} | "
        f"inpaint: {summary.get('inpainted_blocks_count')} ok / {summary.get('inpaint_review_blocks_count')} review | "
        f"render: {summary.get('rendered_blocks_count')} | overflow: {summary.get('overflow_blocks_count')} | "
        f"kısa-çevrilmeyen: {count_short_dialogue_untranslated(raw_regions)}",
        "",
        f"## REVIEW bölgeler ({len(review_regs)})",
    ]
    for r in sorted(review_regs, key=lambda x: x.get("id", 0))[:100]:
        lines.append(
            f"- r{r.get('id')} [{r.get('type')}/{r.get('review_reason')}]: {(r.get('text') or '').strip()[:80]}"
        )
    if len(review_regs) > 100:
        lines.append(f"- … +{len(review_regs) - 100} bölge daha (regions.json'a bakın)")
    lines += ["", f"## Inpaint-REVIEW blokları ({len(inpaint_review_ids)})"]
    for bid in inpaint_review_ids[:50]:
        b = blocks.get(bid, {})
        lines.append(f"- b{bid}: {(b.get('source_text') or '').strip()[:80]}")
    lines += ["", "## Kilitli terimler"]
    for t in (glossary.get("locked_terms", []) or [])[:30]:
        lines.append(
            f"- {t.get('source')}->{t.get('target')} (x{t.get('occurrences')}, {t.get('method')})"
        )
    if not glossary.get("locked_terms"):
        lines.append("- (kilit yok)")
    lines += ["", "## Gözlenen (kilitsiz) terimler"]
    for t in (glossary.get("observed_terms", []) or [])[:20]:
        lines.append(f"- {t.get('source')} (x{t.get('occurrences')})")

    out_dir = ROOT / "review_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_path = out_dir / f"defect_report_{stamp}.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DEFECT-REPORT] {len(review_regs)} review, {len(inpaint_review_ids)} inpaint-review -> {out_path}")


if __name__ == "__main__":
    main()
