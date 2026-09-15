"""LaBSE kalibrasyonu: av derlemindeki (kaynak, çeviri) benzerlik dağılımı.

KULLANIM: .venv\\Scripts\\python.exe scripts/score_mining_embed.py
GİRDİ: scratch/mine_v1/candidates.jsonl
ÇIKTI: scratch/mine_v1/embed_scores.jsonl (source, tr, sim, flags)
GPU ister (CPU kutuda kullanılamaz-yavaş); hat boru-hattıyla yan yana gelmez.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    import torch
    from sentence_transformers import SentenceTransformer, util

    cand_path = ROOT / "scratch" / "mine_v1" / "candidates.jsonl"
    out_path = ROOT / "scratch" / "mine_v1" / "embed_scores.jsonl"
    rows = [json.loads(l) for l in cand_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    srcs = [r["source"] for r in rows]
    trs = [r.get("tr") or "" for r in rows]
    print(f"[EMBED] {len(rows)} çift", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/LaBSE", device=device)
    t0 = time.time()
    embs = model.encode(srcs + trs, batch_size=32, show_progress_bar=False)
    n = len(rows)
    with out_path.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            sim = float(util.cos_sim(embs[i], embs[n + i]).item())
            fh.write(json.dumps({"source": r["source"], "tr": r["tr"], "sim": round(sim, 4),
                                 "flags": r.get("flags", {})}, ensure_ascii=False) + "\n")
    el = round(time.time() - t0, 1)
    sims = sorted(
        float(util.cos_sim(embs[i], embs[n + i]).item()) for i in range(n)
    )
    import statistics

    print(f"[EMBED] bitti {el}sn ort={statistics.mean(sims):.3f} "
          f"p5={sims[int(0.05*n)]:.3f} p10={sims[int(0.10*n)]:.3f} min={sims[0]:.3f}", flush=True)
    print("[EMBED] en-düşük 12:", flush=True)
    order = sorted(range(n), key=lambda i: float(util.cos_sim(embs[i], embs[n + i]).item()))
    for i in order[:12]:
        print(f"  {float(util.cos_sim(embs[i], embs[n+i]).item()):.3f} {srcs[i][:55]!r} -> {trs[i][:55]!r} {rows[i].get('flags', {})}")


if __name__ == "__main__":
    main()
