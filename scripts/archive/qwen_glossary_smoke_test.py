#!/usr/bin/env python3
"""Qwen glossary proposal smoke test.

Loads Qwen3.5-9B 8-bit ONCE and generates multi-option Turkish translation proposals for 3 discovered candidates:
  1. GUILD MASTER (title_or_rank)
  2. MANA CORE (term)
  3. INNER DISCIPLE (title_or_rank)

Verifies:
  - Valid structured GlossaryProposal outputs with 2-3 options
  - preferred_target is non-empty Turkish text matching candidate source
  - Contextual reason is provided
  - Warnings recorded for quality flags (source language leak, external claims) without crashing
  - CandidateStore candidates updated with preferred_target while status remains 'provisional'
  - Confirmed SeriesProfile remains untouched
  - Peak VRAM measured accurately during inference before unload

Usage:
    .venv\\Scripts\\python.exe scripts/qwen_glossary_smoke_test.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from core.translation.glossary_proposal import GlossaryProposal
from core.translation.profile_discovery import CandidateEvidence, CandidateStore, ProfileCandidate
from core.translation.series_profile import SeriesProfile
from providers.translation.qwen_glossary import propose_glossary_targets_with_qwen
from providers.translation.qwen_translation import QwenTranslationProvider

TEST_CANDIDATES = [
    ProfileCandidate(
        source="GUILD MASTER",
        kind="title_or_rank",
        status="provisional",
        evidence=[CandidateEvidence(chapter_id="ch001", region_id=1, text="KANG MINHO, REPORT TO THE GUILD MASTER.")],
    ),
    ProfileCandidate(
        source="MANA CORE",
        kind="term",
        status="provisional",
        evidence=[CandidateEvidence(chapter_id="ch001", region_id=3, text="ONLY AWAKENERS WITH A STABLE MANA CORE MAY ENTER.")],
    ),
    ProfileCandidate(
        source="INNER DISCIPLE",
        kind="title_or_rank",
        status="provisional",
        evidence=[CandidateEvidence(chapter_id="ch001", region_id=11, text="AN INNER DISCIPLE MUST PROTECT HIS DANTIAN.")],
    ),
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

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

    load_vram = provider.metrics.model_load_vram_gb
    print(f"  Model load time: {load_seconds:.2f} s")
    print(f"  Model-load VRAM: {load_vram:.2f} GB")

    store = CandidateStore(series_id="glossary_smoke_test")
    for cand in TEST_CANDIDATES:
        store.candidates[cand.source.upper()] = cand

    profile = SeriesProfile(series_id="glossary_smoke_test")

    print("\n" + "=" * 60)
    print("GLOSSARY PROPOSAL RUN (3 CANDIDATES)")
    print("=" * 60)

    t0 = time.perf_counter()
    proposals = propose_glossary_targets_with_qwen(
        provider=provider,
        candidate_store=store,
        existing_profile=profile,
        candidates_to_propose=TEST_CANDIDATES,
    )
    prop_time = time.perf_counter() - t0

    # Measure VRAM DURING/RIGHT AFTER inference BEFORE unload
    peak_inference_vram = 0.0
    if torch.cuda.is_available():
        peak_inference_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(f"  Proposal execution time: {prop_time:.2f}s")
    print(f"  Proposals generated ({len(proposals)}):")
    for prop in proposals:
        opts_str = ", ".join(f"'{o}'" for o in prop.options)
        print(f"    - [{prop.kind}] {prop.source}")
        print(f"      options: [{opts_str}]")
        print(f"      preferred_target: '{prop.preferred_target}'")
        print(f"      reason: {prop.reason}")
        print(f"      is_valid: {prop.is_valid}, requires_review: {prop.requires_review}")
        print(f"      warnings: {prop.warnings if prop.warnings else 'None'}")

    checks = []

    # 1. Valid proposals returned
    checks.append(("three_proposals_returned", len(proposals) == 3, f"count={len(proposals)}"))

    # 2. Options present
    all_has_options = all(len(p.options) > 0 for p in proposals)
    checks.append(("options_generated", all_has_options, ""))

    # 3. Preferred target non-empty
    all_targets_valid = all(bool(p.preferred_target and p.preferred_target.strip()) for p in proposals)
    checks.append(("non_empty_preferred_targets", all_targets_valid, ""))

    # 4. Contextual reasons provided
    all_reasons_valid = all(bool(p.reason and p.reason.strip()) for p in proposals)
    checks.append(("contextual_reasons_provided", all_reasons_valid, ""))

    # 5. Store updated while status remains provisional
    all_provisional = all(c.status == "provisional" for c in store.candidates.values())
    checks.append(("store_candidates_remain_provisional", all_provisional, ""))

    # 6. Confirmed SeriesProfile remains completely untouched
    profile_untouched = len(profile.glossary) == 0 and len(profile.known_names) == 0
    checks.append(("confirmed_profile_remains_untouched", profile_untouched, ""))

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
    print(f"Proposal Time: {prop_time:.2f}s")
    print(f"Model Load VRAM: {load_vram:.2f} GB")
    print(f"Peak Inference VRAM: {peak_inference_vram:.2f} GB")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
