r"""V1 kör-nokta avı: cümleleri Hy-MT2'den geçir, otomatik eleklerle süz.

KULLANIM (arka plan):
    .venv\Scripts\python.exe scripts/mine_blind_spots_v1.py --limit 2500
ÇIKTI: scratch/mine_v1/candidates.jsonl (artımlı, kaldığı yerden devam eder)
       + summary.json (sonda). Pipeline'a DOKUNMAZ (salt okuma + çeviri).

Elekler (hepsi genel, dile-özel; bölüme-özel sayı YOK):
  J_uydurma: TR sözcük hunspell-tr'de YOK + kaynakta YOK (AMBİLGO sınıfı)
  J_sorry:   kaynakta SORRY + TR'de tek-başına özürüm/özürsün
  J_kinship: find_kinship_mismatch (akraba-anlam kayması)
  J_embed:   --embed ile LaBSE kaynak-çeviri benzerliği < 0.55
             (kalibrasyon: 703 çift ort=0.748 p5=0.560; kaba-anlam
             kayması + boş çeviri yakalar, ince-rol hatasını YAKALAMAZ —
             o kapalı listenin işi; GPU ister, boru-hattına girmez)
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.translation.protection import find_kinship_mismatch

_TR_WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]{4,}")
_SORRY_SRC_RE = re.compile(r"(?<![A-Za-z])SORRY(?![A-Za-z])", re.IGNORECASE)
_SORRY_TR_RE = re.compile(r"(?<![A-Za-zÇĞİÖŞÜçğıöşü])ÖZ[ÜU]R[ÜU]M(?![A-Za-zÇĞİÖŞÜçğıöşü])|(?<![A-Za-zÇĞİÖŞÜçğıöşü])ÖZ[ÜU]RS[ÜU]N(?![A-Za-zÇĞİÖŞÜçğıöşü])")

# Türkçe-duyarlı küçük-harf: I->ı, İ->i ÖNCE (casefold I/İ'yi bozar:
# İ->i+birleşen-nokta, sözlük eşleşmesi ölür). ASCII I sorunu da kapanır.
_TR_LOWER_MAP = str.maketrans({"I": "ı", "İ": "i"})


def tr_lower(s: str) -> str:
    return s.translate(_TR_LOWER_MAP).lower()


def load_spell():
    try:
        from core.translation.chapter_glossary import _spell_dictionary

        spell = _spell_dictionary()
        if spell is None:
            raise RuntimeError("spell dictionary unavailable")
        spell.lookup("test")
        return spell
    except Exception as exc:
        print(f"[MINE] sozluk yok (J_uydurma atlanir): {exc}", flush=True)
        return None


def judge_unknown(tr: str, src_words: set[str], spell) -> list[str]:
    if spell is None:
        return []
    hits = []
    for w in set(_TR_WORD_RE.findall(tr)):
        wl = tr_lower(w)
        if wl in src_words:
            continue
        # Özel-ad + Türkçe eki sınıfı (ODYSSEYİ, HYUNJInin): kaynak sözcüğün
        # devamıysa bayrak yok — genel, liste yok.
        if any(len(s) >= 4 and wl.startswith(s) for s in src_words):
            continue
        try:
            if spell.lookup(wl):
                continue
        except Exception:
            continue
        hits.append(w)
    return sorted(hits)


def judge_sorry(src: str, tr: str) -> bool:
    return bool(_SORRY_SRC_RE.search(src)) and bool(_SORRY_TR_RE.search(tr))


# LaBSE kaba-anlam eşiği (kalibrasyon: 703 çift ort=0.748 p5=0.560).
# Altı = boş/kaba-kayma çeviri adayı. İnce-rol hatasını YAKALAMAZ.
EMBED_SIM_THRESHOLD = 0.55


def judge_embed(sim: float) -> bool:
    return sim < EMBED_SIM_THRESHOLD


def collect_corpus(seeds_path: Path) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for f in sorted(glob.glob(str(ROOT / "audit_output" / "golden" / "*" / "analysis" / "regions.json"))):
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        for b in d.get("text_blocks", []) or []:
            s = (b.get("source_text") or "").strip()
            # Tek-kelimelik kaynaklar av-dışı (bağlamsız tek sözcükte model
            # saçmalar: THE->MİRAS sınıfı; boru-hattı blok çevirir).
            if s and s not in seen and 2 <= len(s.split()) <= 30:
                seen.add(s)
                out.append(s)
    for f in sorted(glob.glob(str(ROOT / "audit_output" / "dungeon_ch2" / "analysis" / "regions.json"))):
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        for b in d.get("text_blocks", []) or []:
            s = (b.get("source_text") or "").strip()
            if s and s not in seen and 2 <= len(s.split()) <= 30:
                seen.add(s)
                out.append(s)
    if seeds_path.is_file():
        for line in seeds_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            # Tohumlar özenli listedir: tek kelime de olur (DAMMIT...! sınıfı).
            if s and not s.startswith("#") and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2500)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--seeds", default="benchmark/mining_seeds_v1.txt")
    ap.add_argument("--outdir", default="scratch/mine_v1")
    # --embed: LaBSE benzerlik eleği (GPU ister, boru-hattıyla yan yana
    # gelmez; requirements-mining.txt). Kapalıysa sıfır ek bağımlılık.
    ap.add_argument("--embed", action="store_true")
    args = ap.parse_args()

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cand_path = outdir / "candidates.jsonl"
    done: set[str] = set()
    if cand_path.is_file():
        for line in cand_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["source"])
            except Exception:
                pass
    print(f"[MINE] onceki calismadan {len(done)} cumle atlanacak", flush=True)

    corpus = collect_corpus(ROOT / args.seeds)
    todo = [s for s in corpus if s not in done][: args.limit]
    print(f"[MINE] derlem={len(corpus)} yapilacak={len(todo)} chunk={args.chunk}", flush=True)
    if not todo:
        print("[MINE] yapilacak is yok", flush=True)
        return

    from providers.translation.base import TranslationInput, TranslationItem
    from providers.translation.hy_mt2_gguf_translation import HyMT2GGUFTranslationProvider

    spell = load_spell()
    tr_provider = HyMT2GGUFTranslationProvider()
    tr_provider.load()
    model_name = getattr(getattr(tr_provider, "metrics", None), "translation_model", "unknown")
    print(f"[MINE] model={model_name}", flush=True)

    embedder = None
    if args.embed:
        import torch
        from sentence_transformers import SentenceTransformer, util

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = (SentenceTransformer("sentence-transformers/LaBSE", device=device), util)
        print(f"[MINE] embed=acik ({device}, esik={EMBED_SIM_THRESHOLD})", flush=True)

    counts = {"n": 0, "J_uydurma": 0, "J_sorry": 0, "J_kinship": 0, "J_embed_low": 0, "sec": 0.0}
    t0 = time.time()
    try:
        with cand_path.open("a", encoding="utf-8") as fh:
            for i in range(0, len(todo), args.chunk):
                batch = todo[i : i + args.chunk]
                t1 = time.time()
                out = tr_provider.translate(
                    TranslationInput(
                        items=[TranslationItem(region_id=j + 1, source=s) for j, s in enumerate(batch)],
                        glossary=[],
                    )
                )
                tr_map = {it.region_id: (it.translation or "") for it in out.results}
                counts["sec"] += time.time() - t1
                sim_map: dict[int, float] = {}
                if embedder is not None:
                    model_e, util_e = embedder
                    e = model_e.encode(batch + [tr_map.get(j + 1, "") for j in range(len(batch))],
                                       batch_size=32, show_progress_bar=False)
                    for j in range(len(batch)):
                        sim_map[j + 1] = float(util_e.cos_sim(e[j], e[len(batch) + j]).item())
                for j, s in enumerate(batch):
                    tr = tr_map.get(j + 1, "")
                    src_words = set(re.findall(r"[A-Za-z]+", s.casefold()))
                    flags: dict = {}
                    unk = judge_unknown(tr, src_words, spell)
                    if unk:
                        flags["J_uydurma"] = unk
                        counts["J_uydurma"] += 1
                    if judge_sorry(s, tr):
                        flags["J_sorry"] = True
                        counts["J_sorry"] += 1
                    if find_kinship_mismatch(s, tr):
                        flags["J_kinship"] = True
                        counts["J_kinship"] += 1
                    if (j + 1) in sim_map and judge_embed(sim_map[j + 1]):
                        flags["J_embed_low"] = round(sim_map[j + 1], 4)
                        counts["J_embed_low"] += 1
                    rec = {"source": s, "tr": tr, "flags": flags, "model": model_name}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counts["n"] += 1
                fh.flush()
                if counts["n"] % 50 == 0 or i + args.chunk >= len(todo):
                    el = round(time.time() - t0, 1)
                    print(f"[MINE] {counts['n']}/{len(todo)} uydurma={counts['J_uydurma']} sorry={counts['J_sorry']} kin={counts['J_kinship']} embed={counts['J_embed_low']} {el}sn", flush=True)
    finally:
        try:
            tr_provider.unload()
        except Exception:
            pass
    summary = {**counts, "model": model_name, "total_sec": round(time.time() - t0, 1)}
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[MINE] BITTI: {summary}", flush=True)


if __name__ == "__main__":
    main()
