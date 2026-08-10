#!/usr/bin/env python3
"""Qwen translation smoke test.

Loads Qwen3.5-9B 8-bit once, translates 8-10 real English bubble sources
in a single batch, measures VRAM, then unloads.

Usage:
  .venv\Scripts\python.exe scripts/qwen_translation_smoke_test.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from providers.translation.qwen_translation import QwenTranslationProvider
from providers.translation.base import TranslationInput, TranslationItem

KNOWN_NAMES = [
    "LUO TIAN",
    "HU SAN",
    "GAO YUAN",
    "YU",
]

GLOSSARY = [
    "ABILITY USER -> yetenek kullanıcısı",
    "SECRET REALM -> gizli âlem",
    "SECRET REALM GUIDE -> gizli âlem rehberi",
    "LEVEL 1 -> 1. seviye",
    "BLACKWIND RAVINE -> Blackwind Ravine",
]

CHAPTER_CONTEXT = (
    "A young man with spatial abilities (Luo Tian) is exploring a dangerous area "
    "with his team including Captain Gao Yuan and Hu San. They encounter beast "
    "creatures and discuss level 1 ability users, secret realms, and guides."
)

# 10 real English bubble sources from the chapter (reading order)
BUBBLES = [
    (10, "JUDGING BY LUO TIAN'S PERFORMANCE JUST NOW, HE'S ALMOST ON PAR WITH A LEVEL 1 ABILITY USER WHO SPECIALIZES IN ARCHERY."),
    (11, "YOUNG MASTER YU, CAPTAIN GAO, WE NEED TO BE CAREFUL FROM HERE ON."),
    (12, "THESE GRAY WOLF BEASTS ARE SUPPOSED TO BE ACTIVE IN BLACKWIND RAVINE AHEAD OF US."),
    (13, "HU SAN, YOU'RE THE FASTEST. GO SCOUT THE PATH AHEAD."),
    (14, "THE FACT THAT THEY'VE APPEARED HERE IS PROBABLY NOT A GOOD SIGN."),
    (15, "CAPTAIN GAO YUAN IS A PEAK LEVEL 1 ABILITY USER, AND THE REST OF THE TEAM ARE NO PUSHOVERS EITHER."),
    (16, "RELAX, KID. YOU SAW IT YOURSELF JUST NOW."),
    (17, "I'M USED TO IT."),
    (18, "COUNTLESS SPATIAL SECRET REALMS HAVE FORMED ALL AROUND THE WORLD."),
    (19, "MY NAME IS LUO TIAN. I'M NOT AN ABILITY USER-I'M A SECRET REALM GUIDE."),
]


def main():
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
            reading_order=rid - 10,
            known_names=KNOWN_NAMES,
        )
        for rid, src in BUBBLES
    ]

    inp = TranslationInput(
        items=items,
        glossary=GLOSSARY,
        chapter_context=CHAPTER_CONTEXT,
    )

    # Load model
    print("\n=== Loading Qwen3.5-9B 8-bit ===")
    provider = QwenTranslationProvider()
    t0 = time.perf_counter()
    provider.load()
    load_time = time.perf_counter() - t0
    print(f"  Load time: {load_time:.2f}s")
    print(f"  Model: {provider.metrics.translation_model}")
    print(f"  Model-load VRAM: {provider.metrics.model_load_vram_gb:.2f} GB")

    if torch.cuda.is_available():
        peak_after_load = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak VRAM after load: {peak_after_load:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    # Translate
    print(f"\n=== Translating {len(items)} bubbles ===")
    t0 = time.perf_counter()
    out = provider.translate(inp)
    translate_time = time.perf_counter() - t0
    print(f"  Translate time: {translate_time:.2f}s")
    print(f"  Input token count: {provider.metrics.input_token_count}")
    print(f"  Generated token count: {provider.metrics.generated_token_count}")
    print(f"  Max new tokens: {provider.metrics.max_new_tokens}")
    print(f"  Tokens / sec: {provider.metrics.tokens_per_sec:.2f}")
    print(f"  Generation call count: {provider.metrics.generation_call_count}")
    print(f"  JSON retry happened: {provider.metrics.json_retry_happened}")

    if torch.cuda.is_available():
        peak_vram = provider.metrics.peak_vram_gb
        print(f"  Peak inference VRAM: {peak_vram:.2f} GB")
        print(f"  12 GB limit: {'OK' if peak_vram < 12 else 'EXCEEDED'}")

    # Results
    print("\n" + "=" * 72)
    print("TRANSLATION RESULTS")
    print("=" * 72)

    review_count = 0
    for item, result in zip(items, out.results):
        print(f"\n[{item.reading_order}] id={item.region_id}")
        print(f"  EN:  {item.source}")
        print(f"  TR:  {result.translation or '(unresolved)'}")
        if result.validation_warnings:
            print(f"  warnings: {result.validation_warnings}")
            review_count += 1
        if result.requires_review:
            print(f"  requires_review: True")

    print("\n" + "=" * 72)
    print(f"Bubbles translated: {len(out.results)}")
    print(f"Results requiring review: {review_count}")
    print(f"Raw response length: {len(out.raw_response)} chars")
    print(f"Model: {provider.metrics.translation_model}")
    print(f"Model-load time: {load_time:.2f}s")
    print(f"Generation time: {translate_time:.2f}s")
    print(f"Input tokens: {provider.metrics.input_token_count}")
    print(f"Generated tokens: {provider.metrics.generated_token_count}")
    print(f"Tokens/sec: {provider.metrics.tokens_per_sec:.2f}")
    print(f"Generation calls: {provider.metrics.generation_call_count}")
    print(f"JSON retry: {provider.metrics.json_retry_happened}")
    print(f"Model-load VRAM: {provider.metrics.model_load_vram_gb:.2f} GB")
    if torch.cuda.is_available():
        print(f"Peak inference VRAM: {provider.metrics.peak_vram_gb:.2f} GB")
    print("=" * 72)

    # Unload
    provider.unload()
    torch.cuda.empty_cache()
    print("\nModel unloaded.")

    # Print raw response (truncated)
    print(f"\nRaw model response (first 1000 chars):")
    print(out.raw_response[:1000])


if __name__ == "__main__":
    main()
