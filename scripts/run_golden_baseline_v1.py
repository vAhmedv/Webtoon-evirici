"""F1 altın-set terazi ölçümü (kod yok, sadece koşu).

Aynı denetimi (audit_e2e_real_chapter1.py, --strict YOK — terazi ham ölçer)
4 Asura Ch1 bölümünde sırayla koşturur (VRAM çakışmasın diye paralel YOK),
her birinin e2e_audit_metrics.json özetini benchmark/golden_baseline_v1.json
dosyasında dondurur. Sonraki fix'ler bu sayılarla karşılaştırılır.

KULLANIM:
    .venv\\Scripts\\python.exe scripts\\run_golden_baseline_v1.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = Path(r"C:\Users\Ahmed\AppData\Local\Tachidesk\downloads\mangas\Asura Scans (EN)")
CHAPTERS = [
    ("dungeon_odyssey_ch1", BASE / "Dungeon Odyssey" / "Chapter 1"),
    ("omniscient_reader_prologue", BASE / "Omniscient Reader\u2019s Viewpoint" / "Chapter 0 - Prologue"),
    ("swordmaster_ch1", BASE / "Swordmaster\u2019s Youngest Son" / "Chapter 1"),
    ("estate_developer_ch1", BASE / "The Greatest Estate Developer" / "Chapter 1"),
]

SUMMARY_KEYS = [
    "source_page_count",
    "output_page_count",
    "text_block_count",
    "translation_eligible_blocks_count",
    "translated_blocks_count",
    "successfully_inpainted_blocks_count",
    "review_inpaint_blocks_count",
    "actually_rendered_blocks_count",
    "overflow_blocks_count",
    "short_dialogue_untranslated_count",
    "final_auto_regions",
    "final_review_regions",
    "final_skip_regions",
    "ocr_elapsed_seconds",
    "translation_elapsed_seconds",
    "inpainting_rendering_elapsed_seconds",
    "elapsed_seconds",
    "gates",
]


def main() -> None:
    frozen: dict = {
        "name": "golden_baseline_v1",
        "created": time.strftime("%Y-%m-%d"),
        "code": "F1 olcumu (mevcut kod, fix yok)",
        "chapters": {},
    }
    for tag, src in CHAPTERS:
        try:
            if not src.is_dir():
                raise AssertionError(f"Kaynak yok: {src}")
            out = ROOT / "audit_output" / "golden" / tag
            metrics_path = out / "e2e_audit_metrics.json"
            if metrics_path.is_file():
                # Aynı kod + aynı girdi ile önceki ölçüm geçerli — tekrar koşma.
                print(f"[GOLDEN] atlaniyor (ölçüm var): {tag}", flush=True)
                full = json.loads(metrics_path.read_text(encoding="utf-8"))
                frozen["chapters"][tag] = {
                    "source": str(src),
                    "output": str(out),
                    "run_seconds": None,
                    **{k: full.get(k) for k in SUMMARY_KEYS},
                }
                continue
            env = dict(os.environ)
            env["AUDIT_SOURCE_CHAPTER"] = str(src)
            env["AUDIT_OUTPUT_DIR"] = str(out)
            print(f"[GOLDEN] basliyor: {tag} ({src})", flush=True)
            t0 = time.time()
            proc = subprocess.run(
                [sys.executable, "scripts/audit_e2e_real_chapter1.py"],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                errors="replace",
            )
            dt = round(time.time() - t0, 1)
            out.mkdir(parents=True, exist_ok=True)
            (out / "audit_stdout.log").write_text(proc.stdout[-6000:], encoding="utf-8")
            (out / "audit_stderr.log").write_text(proc.stderr[-6000:], encoding="utf-8")
            print(f"[GOLDEN] bitti: {tag} ({dt} sn, exit={proc.returncode})", flush=True)
            metrics_path = out / "e2e_audit_metrics.json"
            if proc.returncode != 0 or not metrics_path.is_file():
                print(f"[GOLDEN-HATA] {tag}: metrik yok, kosu loguna bak: {out}", flush=True)
                frozen["chapters"][tag] = {"error": True, "run_seconds": dt, "output": str(out)}
                continue
            full = json.loads(metrics_path.read_text(encoding="utf-8"))
            frozen["chapters"][tag] = {
                "source": str(src),
                "output": str(out),
                "run_seconds": dt,
                **{k: full.get(k) for k in SUMMARY_KEYS},
            }
        except Exception as exc:
            print(f"[GOLDEN-HATA] {tag}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            frozen["chapters"][tag] = {"error": True, "run_seconds": None, "output": ""}
            continue

    dest = ROOT / "benchmark" / "golden_baseline_v1.json"
    dest.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[GOLDEN-OK] terazi donduruldu -> {dest}", flush=True)
    for tag, ch in frozen["chapters"].items():
        if ch.get("error"):
            print(f"  {tag}: HATA")
        else:
            print(
                f"  {tag}: blok={ch.get('text_block_count')} cevrilmis={ch.get('translated_blocks_count')} "
                f"basılmıs={ch.get('actually_rendered_blocks_count')} overflow={ch.get('overflow_blocks_count')} "
                f"kisacevrilmemis={ch.get('short_dialogue_untranslated_count')} review={ch.get('final_review_regions')} "
                f"sure={ch.get('elapsed_seconds')}sn"
            )


if __name__ == "__main__":
    main()
