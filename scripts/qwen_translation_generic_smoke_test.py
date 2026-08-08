#!/usr/bin/env python3
"""Qwen generic multi-series translation smoke test.

Runs two test cases with Qwen3.5-9B 8-bit loaded ONCE:
  Test A: Empty Profile (generic translator works with empty profile)
  Test B: Koharu Test Profile (names & glossary loaded from test fixture)

Usage:
    .venv\\Scripts\\python.exe scripts/qwen_translation_generic_smoke_test.py
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from core.translation.series_profile import SeriesProfile
from providers.translation.base import TranslationInput, TranslationItem
from providers.translation.qwen_translation import QwenTranslationProvider

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

    # Load model ONCE
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
    print(f"  Model load time: {load_seconds:.2f} s")
    print(f"  Model-load VRAM: {provider.metrics.model_load_vram_gb:.2f} GB")

    items = [
        TranslationItem(
            region_id=rid,
            source=src,
            reading_order=rid - 16,
        )
        for rid, src in BUBBLES
    ]

    checks = []

    # --- TEST A: Empty Profile ---
    print("\n" + "=" * 60)
    print("TEST A: Empty Profile (Generic Translator)")
    print("=" * 60)
    empty_profile = SeriesProfile(series_id="empty_test")
    inp_a = TranslationInput(items=items, profile=empty_profile)

    t0 = time.perf_counter()
    out_a = provider.translate(inp_a)
    gen_time_a = time.perf_counter() - t0

    print(f"  Translate time: {gen_time_a:.2f}s")
    for r in out_a.results:
        print(f"  [{r.region_id}] {r.source} -> {r.translation}")

    parsed_a = all(r.translation is not None for r in out_a.results)
    no_reviews_a = sum(1 for r in out_a.results if r.requires_review) == 0
    checks.append(("test_a_parsed_ok", parsed_a, ""))
    checks.append(("test_a_zero_review_warnings", no_reviews_a, ""))

    # --- TEST B: Koharu Test Profile Fixture ---
    print("\n" + "=" * 60)
    print("TEST B: Koharu Test Profile Fixture")
    print("=" * 60)
    koharu_fixture_path = Path("test_data/series_profiles/koharu_test.json")
    koharu_profile = SeriesProfile.load_from_json(koharu_fixture_path)
    print(f"  Loaded profile: {koharu_profile.series_id}")
    print(f"  Known names: {koharu_profile.known_names}")
    print(f"  Glossary: {koharu_profile.glossary}")

    inp_b = TranslationInput(items=items, profile=koharu_profile)

    t0 = time.perf_counter()
    out_b = provider.translate(inp_b)
    gen_time_b = time.perf_counter() - t0

    print(f"  Translate time: {gen_time_b:.2f}s")
    for r in out_b.results:
        print(f"  [{r.region_id}] {r.source} -> {r.translation}")

    parsed_b = all(r.translation is not None for r in out_b.results)
    no_reviews_b = sum(1 for r in out_b.results if r.requires_review) == 0
    t19_b = next((r.translation or "" for r in out_b.results if r.region_id == 19), "")
    luo_kept_b = "luo tian" in t19_b.lower()
    glossary_b = "yetenek" in t19_b.lower() or "gizli" in t19_b.lower()

    checks.append(("test_b_parsed_ok", parsed_b, ""))
    checks.append(("test_b_zero_review_warnings", no_reviews_b, ""))
    checks.append(("test_b_luo_tian_preserved", luo_kept_b, repr(t19_b)))
    checks.append(("test_b_glossary_applied", glossary_b, repr(t19_b)))

    # --- Unload ---
    provider.unload()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("SUMMARY VERIFICATION")
    print("=" * 60)
    all_ok = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))

    print("\n" + "=" * 60)
    print(f"RESULT: {'ALL CHECKS PASSED' if all_ok else 'FAILURES PRESENT'}")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
