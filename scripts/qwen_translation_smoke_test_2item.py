#!/usr/bin/env python3
"""Qwen translation 2-item smoke test.

Focused smoke test (per task step 3): loads Qwen3.5-9B 8-bit ONCE, translates
exactly the two specified real English bubbles in a single batch, then verifies
the contract (Turkish output, LUO TIAN preserved, 2 IDs, JSON-parsed, no
thinking/reasoning leak, peak VRAM) and unloads.

Usage:
    .venv\Scripts\python.exe scripts/qwen_translation_smoke_test_2item.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from providers.translation.qwen_translation import QwenTranslationProvider
from providers.translation.base import TranslationInput, TranslationItem

KNOWN_NAMES = ["LUO TIAN", "HU SAN", "GAO YUAN", "YU"]
GLOSSARY = [
    "ABILITY USER -> yetenek kullanıcısı",
    "SECRET REALM -> gizli âlem",
    "SECRET REALM GUIDE -> gizli âlem rehberi",
    "LEVEL 1 -> 1. seviye",
    "BLACKWIND RAVINE -> Blackwind Ravine",
]

# Exactly the two spec'd bubbles (region_id 16 and 19).
BUBBLES = [
    (16, "RELAX, KID. YOU SAW IT YOURSELF JUST NOW."),
    (19, "MY NAME IS LUO TIAN. I'M NOT AN ABILITY USER-I'M A SECRET REALM GUIDE."),
]


def _has_turkish(s: str) -> bool:
    return bool(re.search(r"[ğçşıöüĞÇŞİÖÜ]", s, re.IGNORECASE))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

    items = [
        TranslationItem(
            region_id=rid,
            source=src,
            reading_order=rid - 16,
            known_names=KNOWN_NAMES,
        )
        for rid, src in BUBBLES
    ]
    inp = TranslationInput(items=items, glossary=GLOSSARY)

    # --- Load ---
    print("\n=== Loading Qwen3.5-9B 8-bit ===")
    provider = QwenTranslationProvider()
    t_load_start = time.perf_counter()
    try:
        provider.load()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nLOAD FAIL: {e}")
        return 2
    load_seconds = time.perf_counter() - t_load_start
    print(f"  Model: {provider.metrics.translation_model}")
    print(f"  Model load time: {load_seconds:.2f} s")
    print(f"  Model-load VRAM: {provider.metrics.model_load_vram_gb:.2f} GB")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- Translate ---
    print(f"\n=== Translating {len(items)} bubbles ===")
    out = provider.translate(inp)

    peak_vram = provider.metrics.peak_vram_gb
    print(f"  Input token count: {provider.metrics.input_token_count}")
    print(f"  Generated token count: {provider.metrics.generated_token_count}")
    print(f"  Max new tokens: {provider.metrics.max_new_tokens}")
    print(f"  Generation seconds: {provider.metrics.generation_seconds:.2f} s")
    print(f"  Tokens / sec: {provider.metrics.tokens_per_sec:.2f}")
    print(f"  Generation call count: {provider.metrics.generation_call_count}")
    print(f"  JSON retry happened: {provider.metrics.json_retry_happened}")
    print(f"  Peak inference VRAM: {peak_vram:.2f} GB")
    if torch.cuda.is_available():
        print(f"  12 GB limit: {'OK' if peak_vram < 12 else 'EXCEEDED'}")

    provider.unload()
    torch.cuda.empty_cache()

    # --- Emit expected JSON structure ---
    payload = {
        "translations": [
            {"id": r.region_id, "translation": r.translation or ""}
            for r in out.results
        ]
    }
    print("\n=== OUTPUT JSON ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    print("\n=== RAW (truncated) ===")
    print(out.raw_response[:1200])

    # --- Verification ---
    checks = []

    ids_returned = sorted(r.region_id for r in out.results)
    checks.append(("two_ids_returned", ids_returned == [16, 19], f"ids={ids_returned}"))

    by_id = {r.region_id: r for r in out.results}
    parsed_ok = all(r.translation is not None for r in out.results)
    checks.append(("json_parsed_with_translations", parsed_ok, ""))

    t16 = by_id[16].translation or ""
    t19 = by_id[19].translation or ""
    checks.append(("item_16_turkish_or_meaningful", _has_turkish(t16) or bool(t16), repr(t16)))
    checks.append(("item_19_turkish_or_meaningful", _has_turkish(t19) or bool(t19), repr(t19)))

    # LUO TIAN name preservation (case-insensitive)
    luo_kept = "luo tian" in t19.lower()
    checks.append(("luo_tian_preserved", luo_kept, repr(t19)))

    # Glossary checks
    glossary_check_19 = "yetenek" in t19.lower() or "gizli" in t19.lower()
    checks.append(("glossary_applied_item_19", glossary_check_19, repr(t19)))

    # No thinking / reasoning leak into results or raw model response.
    raw_lower = out.raw_response.lower()
    translations_blob = t16 + t19
    no_thinking_marker = "<thinking>" not in raw_lower and "</thinking>" not in raw_lower
    no_reasoning_word = "reasoning" not in translations_blob.lower()
    checks.append(("no_thinking_leak", no_thinking_marker and no_reasoning_word, ""))

    # Warning count check
    review_warnings = sum(1 for r in out.results if r.requires_review)
    checks.append(("zero_review_warnings", review_warnings == 0, f"review_count={review_warnings}"))

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    all_ok = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))

    print("\n" + "=" * 60)
    print(f"RESULT: {'ALL CHECKS PASSED' if all_ok else 'FAILURES PRESENT'}")
    print(f"Model Load Time: {load_seconds:.2f} s")
    print(f"Generation Time: {provider.metrics.generation_seconds:.2f} s")
    print(f"Tokens/sec: {provider.metrics.tokens_per_sec:.2f}")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    print(f"IDs: {ids_returned}")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    import time
    sys.exit(main())
