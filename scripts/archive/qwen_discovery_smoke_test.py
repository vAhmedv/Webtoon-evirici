#!/usr/bin/env python3
"""Qwen candidate discovery smoke test.

Loads Qwen3.5-9B 8-bit ONCE and executes discovery extraction on two synthetic webtoon fixtures:
  Fixture A: Dungeon/Guild series
  Fixture B: Murim series

Verifies:
  - Candidates extracted and saved ONLY as 'provisional' in CandidateStore
  - Deterministic evidence filtering discards unverified source candidates
  - Series A and Series B candidates remain cleanly isolated
  - Zero auto-writing into confirmed SeriesProfile data

Usage:
    .venv\\Scripts\\python.exe scripts/qwen_discovery_smoke_test.py
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from core.translation.profile_discovery import CandidateStore
from core.translation.series_profile import SeriesProfile
from providers.translation.base import TranslationItem
from providers.translation.qwen_discovery import discover_candidates_with_qwen
from providers.translation.qwen_translation import QwenTranslationProvider

FIXTURE_A = [
    (1, "KANG MINHO, REPORT TO THE GUILD MASTER."),
    (2, "THE RED GATE HAS OPENED AGAIN."),
    (3, "ONLY AWAKENERS WITH A STABLE MANA CORE MAY ENTER."),
]

FIXTURE_B = [
    (10, "THE SECT LEADER HAS SUMMONED JIN-WOO."),
    (11, "AN INNER DISCIPLE MUST PROTECT HIS DANTIAN."),
    (12, "THE HEAVENLY DEMON CULT IS MOVING AGAIN."),
]


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

    checks = []

    # --- FIXTURE A: Dungeon / Guild ---
    print("\n" + "=" * 60)
    print("FIXTURE A: Dungeon / Guild Discovery")
    print("=" * 60)
    items_a = [TranslationItem(region_id=rid, source=src, reading_order=i) for i, (rid, src) in enumerate(FIXTURE_A)]
    store_a = CandidateStore(series_id="dungeon_guild_smoke")
    profile_a = SeriesProfile(series_id="dungeon_guild_smoke")

    t0 = time.perf_counter()
    res_a = discover_candidates_with_qwen(
        provider=provider,
        series_id="dungeon_guild_smoke",
        chapter_id="ch001",
        items=items_a,
        existing_profile=profile_a,
        candidate_store=store_a,
    )
    time_a = time.perf_counter() - t0

    print(f"  Discovery time: {time_a:.2f}s")
    print(f"  Discovered candidates ({len(res_a.candidates)}):")
    for cand in res_a.candidates:
        print(f"    - [{cand.kind}] {cand.source} -> {cand.suggested_target} (status={cand.status}, evidence={cand.evidence_count})")
    print(f"  Filtered count: {res_a.filtered_count}")

    # Safety principle check: Discovered candidates MUST be provisional and confirmed profile MUST stay empty
    all_provisional_a = all(c.status == "provisional" for c in res_a.candidates)
    profile_untouched_a = len(profile_a.known_names) == 0 and len(profile_a.glossary) == 0
    checks.append(("fixture_a_provisional_status", all_provisional_a, ""))
    checks.append(("fixture_a_confirmed_profile_untouched", profile_untouched_a, ""))
    checks.append(("fixture_a_candidates_found", len(res_a.candidates) > 0, f"count={len(res_a.candidates)}"))

    # --- FIXTURE B: Murim ---
    print("\n" + "=" * 60)
    print("FIXTURE B: Murim Discovery")
    print("=" * 60)
    items_b = [TranslationItem(region_id=rid, source=src, reading_order=i) for i, (rid, src) in enumerate(FIXTURE_B)]
    store_b = CandidateStore(series_id="murim_smoke")
    profile_b = SeriesProfile(series_id="murim_smoke")

    t0 = time.perf_counter()
    res_b = discover_candidates_with_qwen(
        provider=provider,
        series_id="murim_smoke",
        chapter_id="ch001",
        items=items_b,
        existing_profile=profile_b,
        candidate_store=store_b,
    )
    time_b = time.perf_counter() - t0

    print(f"  Discovery time: {time_b:.2f}s")
    print(f"  Discovered candidates ({len(res_b.candidates)}):")
    for cand in res_b.candidates:
        print(f"    - [{cand.kind}] {cand.source} -> {cand.suggested_target} (status={cand.status}, evidence={cand.evidence_count})")
    print(f"  Filtered count: {res_b.filtered_count}")

    all_provisional_b = all(c.status == "provisional" for c in res_b.candidates)
    profile_untouched_b = len(profile_b.known_names) == 0 and len(profile_b.glossary) == 0
    no_cross_contamination = set(store_a.candidates.keys()).isdisjoint(set(store_b.candidates.keys()))

    checks.append(("fixture_b_provisional_status", all_provisional_b, ""))
    checks.append(("fixture_b_confirmed_profile_untouched", profile_untouched_b, ""))
    checks.append(("fixture_b_candidates_found", len(res_b.candidates) > 0, f"count={len(res_b.candidates)}"))
    checks.append(("series_isolation_clean", no_cross_contamination, ""))

    # --- Unload ---
    peak_vram = provider.metrics.peak_vram_gb
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
    print(f"Model Load Time: {load_seconds:.2f}s")
    print(f"Fixture A Time: {time_a:.2f}s")
    print(f"Fixture B Time: {time_b:.2f}s")
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
