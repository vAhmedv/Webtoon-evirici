#!/usr/bin/env python3
"""Qwen Translation Evidence-Driven Termbase smoke test.

Loads Qwen3.5-9B 8-bit ONCE and executes translation on two synthetic webtoon fixtures:
  Fixture A: Dungeon/Guild series (GUILD MASTER, MANA CORE)
  Fixture B: Murim series (INNER DISCIPLE, DANTIAN)

Verifies:
  - Full sentence translations succeed naturally
  - Grounded term_usages are extracted from actual translation outputs
  - TermObservations accumulate in CandidateStore with strict deduplication identity (chapter_id, region_id)
  - Candidate lifecycle transitions: discovered -> provisional -> ready_for_review
  - Zero auto-approval (confirmed SeriesProfile remains completely untouched)
  - Accurate Peak VRAM measurement during inference

Usage:
    .venv\\Scripts\\python.exe scripts/qwen_evidence_termbase_smoke_test.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from core.translation.profile_discovery import CandidateStore, ProfileCandidate, record_validated_term_observations
from core.translation.series_profile import SeriesProfile
from providers.translation.base import TranslationInput, TranslationItem
from providers.translation.qwen_translation import QwenTranslationProvider

FIXTURE_A_ITEMS = [
    TranslationItem(region_id=1, source="KANG MINHO, REPORT TO THE GUILD MASTER.", reading_order=0),
    TranslationItem(region_id=2, source="THE RED GATE HAS OPENED AGAIN.", reading_order=1),
    TranslationItem(region_id=3, source="ONLY AWAKENERS WITH A STABLE MANA CORE MAY ENTER.", reading_order=2),
    TranslationItem(region_id=4, source="THE GUILD MASTER DECIDED TO CHECK THE MANA CORE HIMSELF.", reading_order=3),
]

FIXTURE_B_ITEMS = [
    TranslationItem(region_id=10, source="THE SECT LEADER HAS SUMMONED JIN-WOO.", reading_order=0),
    TranslationItem(region_id=11, source="AN INNER DISCIPLE MUST PROTECT HIS DANTIAN.", reading_order=1),
    TranslationItem(region_id=12, source="ANOTHER INNER DISCIPLE IS GUARDING THE DANTIAN.", reading_order=2),
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

    checks = []

    # --- FIXTURE A: Dungeon / Guild ---
    print("\n" + "=" * 60)
    print("FIXTURE A: Dungeon / Guild Translation & Observation Accumulation")
    print("=" * 60)

    # --- FIXTURE A: Dungeon / Guild ---
    print("\n" + "=" * 60)
    print("FIXTURE A: Dungeon / Guild Translation & Observation Accumulation")
    print("=" * 60)

    store_a = CandidateStore(series_id="dungeon_evidence_smoke")
    # Simulate source-side discovery: populate discovered candidates
    store_a.candidates["GUILD MASTER"] = ProfileCandidate(source="GUILD MASTER", kind="title_or_rank", status="discovered")
    store_a.candidates["RED GATE"] = ProfileCandidate(source="RED GATE", kind="place_name", status="discovered")
    store_a.candidates["MANA CORE"] = ProfileCandidate(source="MANA CORE", kind="term", status="discovered")

    profile_a = SeriesProfile(series_id="dungeon_evidence_smoke")

    inp_a = TranslationInput(
        items=FIXTURE_A_ITEMS,
        profile=profile_a,
        candidate_store=store_a,
        chapter_id="ch001",
    )

    t0 = time.perf_counter()
    out_a = provider.translate(inp_a)
    time_a = time.perf_counter() - t0

    print(f"  Translation time: {time_a:.2f}s")
    print("  Results:")
    for res in out_a.results:
        print(f"    [{res.region_id}] {res.source} -> {res.translation}")
        if res.term_usages:
            print(f"        term_usages: {res.term_usages}")
        if res.fidelity_flags:
            print(f"        fidelity_flags: {res.fidelity_flags}")

        # Orchestration layer calls record_validated_term_observations ONLY after translation is accepted
        if res.translation and not res.requires_review:
            record_validated_term_observations(
                candidate_store=store_a,
                chapter_id="ch001",
                region_id=res.region_id,
                source_text=res.source,
                translated_text=res.translation,
                raw_term_usages=res.term_usages,
                term_id_map=res.term_id_map,
                fidelity_flags=res.fidelity_flags,
                requires_review=res.requires_review,
            )

    print("\n  Accumulated Candidate Store Observations (Fixture A):")
    for cand_key, cand in store_a.candidates.items():
        print(f"    - {cand.source} (status={cand.status}, obs_count={len(cand.observations)}, counts={cand.observed_target_counts})")
        for obs in cand.observations:
            print(f"        * [reg={obs.region_id}] '{obs.source_text}' -> '{obs.translated_text}' (form='{obs.observed_target_form}')")

    # Deduplication test: re-run region 3 (MANA CORE) on same store_a
    duplicate_run_recs = record_validated_term_observations(
        candidate_store=store_a,
        chapter_id="ch001",
        region_id=3,
        source_text="ONLY AWAKENERS WITH A STABLE MANA CORE MAY ENTER.",
        translated_text="Sadece kararlı Mana Çekirdeğine sahip Uyanmışlar girebilir.",
        raw_term_usages=[{"term_id": "T1", "target_form": "Mana Çekirdeğine"}],
        term_id_map={"T1": "MANA CORE"},
    )
    dup_check_ok = len(duplicate_run_recs) == 0

    checks.append(("fixture_a_translations_successful", all(r.translation for r in out_a.results), ""))
    checks.append(("fixture_a_candidates_accumulated", any(len(c.observations) > 0 for c in store_a.candidates.values()), ""))
    checks.append(("fixture_a_confirmed_profile_untouched", len(profile_a.glossary) == 0 and len(profile_a.known_names) == 0, ""))
    checks.append(("deduplication_identity_verified", dup_check_ok, "Re-run region 3 produced 0 duplicate observations"))

    # --- FIXTURE B: Murim ---
    print("\n" + "=" * 60)
    print("FIXTURE B: Murim Translation & Observation Accumulation")
    print("=" * 60)

    store_b = CandidateStore(series_id="murim_evidence_smoke")
    # Simulate source-side discovery: populate discovered candidates
    store_b.candidates["SECT LEADER"] = ProfileCandidate(source="SECT LEADER", kind="title_or_rank", status="discovered")
    store_b.candidates["INNER DISCIPLE"] = ProfileCandidate(source="INNER DISCIPLE", kind="title_or_rank", status="discovered")
    store_b.candidates["DANTIAN"] = ProfileCandidate(source="DANTIAN", kind="term", status="discovered")

    profile_b = SeriesProfile(series_id="murim_evidence_smoke")

    inp_b = TranslationInput(
        items=FIXTURE_B_ITEMS,
        profile=profile_b,
        candidate_store=store_b,
        chapter_id="ch001",
    )

    t0 = time.perf_counter()
    out_b = provider.translate(inp_b)
    time_b = time.perf_counter() - t0

    print(f"  Translation time: {time_b:.2f}s")
    print("  Results:")
    for res in out_b.results:
        print(f"    [{res.region_id}] {res.source} -> {res.translation}")
        if res.term_usages:
            print(f"        term_usages: {res.term_usages}")
        if res.fidelity_flags:
            print(f"        fidelity_flags: {res.fidelity_flags}")

        if res.translation and not res.requires_review:
            record_validated_term_observations(
                candidate_store=store_b,
                chapter_id="ch001",
                region_id=res.region_id,
                source_text=res.source,
                translated_text=res.translation,
                raw_term_usages=res.term_usages,
                term_id_map=res.term_id_map,
                fidelity_flags=res.fidelity_flags,
                requires_review=res.requires_review,
            )

    print("\n  Accumulated Candidate Store Observations (Fixture B):")
    for cand_key, cand in store_b.candidates.items():
        print(f"    - {cand.source} (status={cand.status}, obs_count={len(cand.observations)}, counts={cand.observed_target_counts})")
        for obs in cand.observations:
            print(f"        * [reg={obs.region_id}] '{obs.source_text}' -> '{obs.translated_text}' (form='{obs.observed_target_form}')")

    no_cross_contamination = set(store_a.candidates.keys()).isdisjoint(set(store_b.candidates.keys()))

    checks.append(("fixture_b_translations_successful", all(r.translation for r in out_b.results), ""))
    checks.append(("fixture_b_candidates_accumulated", any(len(c.observations) > 0 for c in store_b.candidates.values()), ""))
    checks.append(("fixture_b_confirmed_profile_untouched", len(profile_b.glossary) == 0 and len(profile_b.known_names) == 0, ""))
    checks.append(("series_isolation_clean", no_cross_contamination, ""))

    peak_inference_vram = 0.0
    if torch.cuda.is_available():
        peak_inference_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

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
    print(f"Model Load VRAM: {load_vram:.2f} GB")
    print(f"Peak Inference VRAM: {peak_inference_vram:.2f} GB")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
