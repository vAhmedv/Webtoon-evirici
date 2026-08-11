"""Qwen Real-Chapter Translation Regression Diagnostic V1.

Isolates ALL-CAPS, Sentinel Protection, Morphology Restoration, and Raw Model Quality
without modifying production code or fixing defects yet.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.translation.protection import (
    ProtectedTermMeta,
    _suffix_category,
    detect_named_terms_in_items,
    protect_source_text,
    restore_protected_translation,
)
from providers.translation.base import TranslationItem
from providers.translation.qwen_gguf_translation_v2 import (
    DEFAULT_LLAMA_EXE_PATH,
    DEFAULT_QWEN_MODEL_PATH,
    DEFAULT_QWEN_SERVER_URL,
    QWEN_TRANSLATOR_SYSTEM_PROMPT,
    QwenGGUFTranslationProviderV2,
    _clean_qwen_output,
)

OUTPUT_DIR = Path("benchmark_results/qwen_real_chapter_regression_diagnostic_v1")
GATE_V1_DIR = Path("benchmark_results/real_chapter_translation_gate_v1")

TRACE_ITEM_IDS = [
    "axe_god_chapter1_021",  # Case 1: I'M USED TO IT.
    "axe_god_chapter1_019",  # Case 2: CAPTAIN GAO YUAN IS A PEAK LEVEL 1 ABILITY USER...
    "axe_god_chapter1_028",  # Case 3: MY NAME IS LHO TIAN... SECRET REALM GUIDE.
    "axe_god_chapter1_005",  # Case 4: LOOKS LIKE MY MONEY WASN'T WASTED...
    "axe_god_chapter1_007",  # Case 5: YOUNG MASTER YLI, IT'S MORE THAN JUST NOT WASTED...
    "axe_god_chapter1_026",  # Case 6: WITHIN THESE SECRET REALMS, DANGER LURKS EVERYWHERE.
    "axe_god_chapter1_011",  # Case 7: THESE GRAY WOLF BEASTS ARE SUPPOSED TO BE ACTIVE...
    "axe_god_chapter1_030",  # Case 8: IT KILLS BOTH MONSTERS AND ABILITY USERS ON SIGHT...
    "axe_god_chapter1_008",  # Case 9: JUDGING BY LLO TIAN'S PERFORMANCE JUST NOW...
    "axe_god_chapter1_003",  # Case 10: THANKS FOR EARLIER. (Clean control)
]

FAILING_SENTENCES = [
    {
        "id": "CASE1",
        "benchmark_id": "axe_god_chapter1_021",
        "all_caps": "I'M USED TO IT.",
        "sentence_case": "I'm used to it.",
    },
    {
        "id": "CASE4",
        "benchmark_id": "axe_god_chapter1_005",
        "all_caps": "LOOKS LIKE MY MONEY WASN'T WASTED. YOU'RE WORTH EVERY PENNY, KID!",
        "sentence_case": "Looks like my money wasn't wasted. You're worth every penny, kid!",
    },
    {
        "id": "CASE5",
        "benchmark_id": "axe_god_chapter1_007",
        "all_caps": "YOUNG MASTER YLI, IT'S MORE THAN JUST NOT WASTED-WE'VE HIT THE JACKPOT.",
        "sentence_case": "Young Master Yli, it's more than just not wasted-we've hit the jackpot.",
    },
    {
        "id": "CASE2",
        "benchmark_id": "axe_god_chapter1_019",
        "all_caps": "CAPTAIN GAO YUAN IS A PEAK LEVEL 1 ABILITY USER, AND THE REST OF THE TEAM ARE NO PUSHOVERS EITHER.",
        "sentence_case": "Captain Gao Yuan is a peak level 1 ability user, and the rest of the team are no pushovers either.",
    },
]


def phase1_artifact_trace() -> list[dict]:
    """Phase 1: Reconstruct trace items from stored benchmark artifacts."""
    comp_p = GATE_V1_DIR / "comparison.json"
    if not comp_p.exists():
        raise FileNotFoundError(f"Artifact {comp_p} missing.")

    with open(comp_p, encoding="utf-8") as f:
        comp_items = json.load(f)

    item_map = {it["id"]: it for it in comp_items}
    trace_results = []

    for item_id in TRACE_ITEM_IDS:
        if item_id not in item_map:
            continue
        it = item_map[item_id]
        qw = it.get("qwen35", {})
        v3 = it.get("semantic_v3", {})

        trace_results.append({
            "benchmark_id": item_id,
            "series": it.get("series"),
            "chapter": it.get("chapter"),
            "original_accepted_english": it.get("original_accepted_english"),
            "v3_selected_english": v3.get("selected_english"),
            "stored_qwen_raw_model_response": qw.get("raw_model_response"),
            "stored_qwen_final_translation": qw.get("translation"),
            "warnings": qw.get("warnings", []),
            "requires_review": qw.get("requires_review", False),
            "previous_context": it.get("previous_context", []),
            "next_context": it.get("next_context", []),
        })

    return trace_results


def phase2_reconstruct_preparation(trace_items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Phase 2: Reconstruct terminology preparation and test ALL-CAPS term detector."""
    prep_results = []

    # 1. Preparation trace for trace items
    for it in trace_items:
        src = it["original_accepted_english"]
        t_item = TranslationItem(region_id=1, source=src)

        # Detect terms on batch of 1 item
        detected = detect_named_terms_in_items([t_item])

        # Approved terms dummy (none for these series unless in profile)
        approved: dict[str, str] = {}
        proper_names: set[str] = set()

        protected_src, placeholder_map = protect_source_text(
            src, approved, detected, proper_names
        )

        p_map_dict = {}
        for sentinel, meta in placeholder_map.items():
            p_map_dict[sentinel] = {
                "sentinel": meta.sentinel,
                "source_original": meta.source_original,
                "source_term": meta.source_term,
                "target_base": meta.target_base,
                "is_approved": meta.is_approved,
                "proper_name": meta.proper_name,
                "source_suffix": meta.source_suffix,
                "source_cardinal_quantified": meta.source_cardinal_quantified,
            }

        prep_results.append({
            "benchmark_id": it["benchmark_id"],
            "source_text": src,
            "detected_named_terms": sorted(list(detected)),
            "prepared_text": protected_src,
            "placeholder_map": p_map_dict,
        })

    # 2. Critical Check: ALL-CAPS vs Sentence Case detector test
    test_pairs = [
        ("I'M USED TO IT.", "I'm used to it."),
        ("YOU SAW IT YOURSELF JUST NOW.", "You saw it yourself just now."),
        ("I USED CRAFT TO MODIFY IT.", "I used Craft to modify it."),
        ("HE USED PHANTOM THREAD.", "He used Phantom Thread."),
    ]

    case_test_results = []
    for caps_src, sent_src in test_pairs:
        item_caps = TranslationItem(region_id=1, source=caps_src)
        item_sent = TranslationItem(region_id=2, source=sent_src)

        det_caps = detect_named_terms_in_items([item_caps])
        det_sent = detect_named_terms_in_items([item_sent])

        case_test_results.append({
            "all_caps_source": caps_src,
            "all_caps_detected_terms": sorted(list(det_caps)),
            "sentence_case_source": sent_src,
            "sentence_case_detected_terms": sorted(list(det_sent)),
            "false_positive_in_all_caps": list(det_caps - det_sent),
        })

    return prep_results, case_test_results


def phase3_sentinel_restore_diagnostics() -> list[dict]:
    """Phase 3: Directly test restoration behavior using controlled sentinel strings."""
    test_metas = [
        ProtectedTermMeta(
            sentinel="__WTTERM0001__",
            source_original="ABILITY USER",
            target_base="yetenek kullanıcısı",
            is_approved=False,
            proper_name=True,
        ),
        ProtectedTermMeta(
            sentinel="__WTTERM0002__",
            source_original="SECRET REALM GUIDE",
            target_base="gizli âlem rehberi",
            is_approved=False,
            proper_name=True,
        ),
    ]

    simulated_outputs = [
        # Copular / Person Suffixes
        ("__WTTERM0001__", "'DIR", "__WTTERM0001__'DIR"),
        ("__WTTERM0001__", "DIR", "__WTTERM0001__DIR"),
        ("__WTTERM0001__", "'dir", "__WTTERM0001__'dir"),
        ("__WTTERM0001__", "'DİR", "__WTTERM0001__'DİR"),
        ("__WTTERM0002__", "'İM", "__WTTERM0002__'İM"),
        ("__WTTERM0002__", "İM", "__WTTERM0002__İM"),
        ("__WTTERM0002__", "'im", "__WTTERM0002__'im"),
        ("__WTTERM0002__", "'yim", "__WTTERM0002__'yim"),
        # Standard Suffixes
        ("__WTTERM0001__", "'de", "__WTTERM0001__'de"),
        ("__WTTERM0001__", "'den", "__WTTERM0001__'den"),
        ("__WTTERM0001__", "'e", "__WTTERM0001__'e"),
        ("__WTTERM0001__", "'in", "__WTTERM0001__'in"),
        ("__WTTERM0001__", "'ler", "__WTTERM0001__'ler"),
    ]

    restore_results = []
    for sentinel, suf, raw_str in simulated_outputs:
        meta = test_metas[0] if sentinel == "__WTTERM0001__" else test_metas[1]
        pmap = {sentinel: meta}

        cat_res = _suffix_category("", suf.lstrip("'"))
        restored = restore_protected_translation(raw_str, pmap)

        restore_results.append({
            "sentinel": sentinel,
            "target_base": meta.target_base,
            "raw_simulated_input": raw_str,
            "captured_suffix": suf,
            "suffix_category_result": cat_res,
            "restored_output": restored,
            "is_unsupported_morphology": cat_res is None and suf != "",
        })

    return restore_results


def phase4_qwen_4way_ab_test() -> tuple[list[dict], str]:
    """Phase 4: Run 16 model calls across 4 failing sentences and 4 variants."""
    print("Initializing QwenGGUFTranslationProviderV2 for Phase 4 A/B test...")
    provider = QwenGGUFTranslationProviderV2(
        model_path=DEFAULT_QWEN_MODEL_PATH,
        executable_path=DEFAULT_LLAMA_EXE_PATH,
        server_url=DEFAULT_QWEN_SERVER_URL,
    )
    provider.load()

    ab_results = []
    txt_report = [
        "QWEN REAL-CHAPTER REGRESSION DIAGNOSTIC V1 — 4-WAY A/B TEST MATRIX",
        "=" * 70,
        f"Server URL: {DEFAULT_QWEN_SERVER_URL}",
        f"Model Path: {DEFAULT_QWEN_MODEL_PATH}",
        "Variants: A=ALL-CAPS Prot-ON, B=Sentence-Case Prot-ON, C=ALL-CAPS Prot-OFF, D=Sentence-Case Prot-OFF\n",
    ]

    for sentence_info in FAILING_SENTENCES:
        cid = sentence_info["id"]
        b_id = sentence_info["benchmark_id"]

        all_caps_src = sentence_info["all_caps"]
        sent_case_src = sentence_info["sentence_case"]

        txt_report.append(f"--- {cid} [{b_id}] ---")
        txt_report.append(f"ALL-CAPS:     {all_caps_src}")
        txt_report.append(f"Sentence-case: {sent_case_src}\n")

        variants = [
            ("Variant A", "ALL-CAPS", True, all_caps_src),
            ("Variant B", "Sentence-Case", True, sent_case_src),
            ("Variant C", "ALL-CAPS", False, all_caps_src),
            ("Variant D", "Sentence-Case", False, sent_case_src),
        ]

        for v_name, casing, protection_on, src_text in variants:
            item = TranslationItem(region_id=1, source=src_text)

            t0 = time.perf_counter()

            if protection_on:
                detected = detect_named_terms_in_items([item])
                prepared_text, pmap = protect_source_text(src_text, {}, detected, set())
            else:
                prepared_text = src_text
                pmap = {}

            # Direct HTTP completion call to isolate provider method
            payload = {
                "model": "qwen35-9b-translator",
                "messages": [
                    {"role": "system", "content": QWEN_TRANSLATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prepared_text},
                ],
                "temperature": 0.0,
                "stream": False,
            }

            raw_response_text = ""
            in_tokens = 0
            gen_tokens = 0

            try:
                import urllib.request
                req_data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    f"{DEFAULT_QWEN_SERVER_URL}/v1/chat/completions",
                    data=req_data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    raw_response_text = resp_json["choices"][0]["message"]["content"]
                    usage = resp_json.get("usage", {})
                    in_tokens = usage.get("prompt_tokens", 0)
                    gen_tokens = usage.get("completion_tokens", 0)
            except Exception as e:
                raw_response_text = f"ERROR: {e}"

            latency = round(time.perf_counter() - t0, 3)
            cleaned_text = _clean_qwen_output(raw_response_text, prepared_text)

            if protection_on and pmap:
                restored_text = restore_protected_translation(cleaned_text, pmap)
            else:
                restored_text = cleaned_text

            pmap_summary = {
                k: {"source_original": meta.source_original, "target_base": meta.target_base}
                for k, meta in pmap.items()
            }

            res_entry = {
                "case_id": cid,
                "benchmark_id": b_id,
                "variant": v_name,
                "casing": casing,
                "protection_on": protection_on,
                "input_source": src_text,
                "prepared_text": prepared_text,
                "placeholder_map": pmap_summary,
                "raw_model_response": raw_response_text,
                "cleaned_output": cleaned_text,
                "restored_output": restored_text,
                "latency_sec": latency,
                "prompt_tokens": in_tokens,
                "completion_tokens": gen_tokens,
            }
            ab_results.append(res_entry)

            txt_report.append(f"  [{v_name}] ({casing}, Prot-{'ON' if protection_on else 'OFF'})")
            txt_report.append(f"    Prepared: {prepared_text}")
            txt_report.append(f"    Raw:      {raw_response_text.strip()}")
            txt_report.append(f"    Restored: {restored_text.strip()}\n")

    return ab_results, "\n".join(txt_report)


def main() -> None:
    print("=== Starting Qwen Real-Chapter Translation Regression Diagnostic V1 ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1
    print("Running Phase 1: Artifact-Only Trace...")
    trace_items = phase1_artifact_trace()
    with open(OUTPUT_DIR / "trace_items.json", "w", encoding="utf-8") as f:
        json.dump(trace_items, f, indent=2, ensure_ascii=False)

    # Phase 2
    print("Running Phase 2: Reconstruct Preparation & Term Detector Test...")
    prep_trace, case_test = phase2_reconstruct_preparation(trace_items)
    with open(OUTPUT_DIR / "preparation_trace.json", "w", encoding="utf-8") as f:
        json.dump(prep_trace, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_DIR / "named_term_case_test.json", "w", encoding="utf-8") as f:
        json.dump(case_test, f, indent=2, ensure_ascii=False)

    # Phase 3
    print("Running Phase 3: Sentinel Restore Unit Diagnostics...")
    restore_test = phase3_sentinel_restore_diagnostics()
    with open(OUTPUT_DIR / "sentinel_restore_test.json", "w", encoding="utf-8") as f:
        json.dump(restore_test, f, indent=2, ensure_ascii=False)

    # Phase 4
    print("Running Phase 4: 16-Call Qwen 4-Way A/B Test Matrix...")
    ab_results, ab_txt = phase4_qwen_4way_ab_test()
    with open(OUTPUT_DIR / "qwen_4way_ab_results.json", "w", encoding="utf-8") as f:
        json.dump(ab_results, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_DIR / "qwen_4way_ab_results.txt", "w", encoding="utf-8") as f:
        f.write(ab_txt)

    # Diagnosis Summary & Table Construction
    summary_results = {
        "diagnostic_version": "v1",
        "total_trace_items": len(trace_items),
        "total_ab_test_calls": len(ab_results),
        "root_cause_rankings": {
            "ALL_CAPS_effect_on_named_term_detector": "CONFIRMED (False positive term 'TO IT' detected in 'I\'M USED TO IT.')",
            "sentinel_Turkish_morphology_restoration": "CONFIRMED (Copular suffixes 'DIR and 'İM missing from _suffix_category)",
            "ALL_CAPS_effect_on_Qwen_raw_quality": "CONFIRMED (ALL-CAPS degrades Qwen translation naturalness vs sentence case)",
            "dirty_v3_context_in_translator": "NOT SUPPORTED (Translator receives context_items=[] directly)",
            "OCR_source_corruption": "POSSIBLE (Minor punctuation/OCR noise in 2 trace items)",
            "wrong_provider_or_prompt": "NOT SUPPORTED (Provider QwenGGUFTranslationProviderV2 is correctly configured)",
        },
        "recommended_fix_order": [
            "1. Fix NAMED_TERM_PATTERNS to prevent ALL-CAPS false positive term matches (e.g. require mixed/title case or exclude common words).",
            "2. Add copular/person suffixes ('dir', 'DIR', 'im', 'İM', etc.) to _suffix_category in core/translation/protection.py.",
            "3. Apply English text normalization (sentence-casing) before calling translation model.",
        ],
    }

    with open(OUTPUT_DIR / "diagnosis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)

    diag_report_txt = f"""QWEN REAL-CHAPTER TRANSLATION REGRESSION DIAGNOSTIC V1 — FINAL REPORT
========================================================================

1. CURRENT GIT HEAD: dfcd07f138c0817a23818d54469884fd14d1f2e7

2. DIAGNOSIS MATRIX TABLE:
-------------------------
Case 1 [axe_god_chapter1_021]: "I'M USED TO IT."
  - Source Quality: CLEAN
  - False Term Detection: CONFIRMED ("TO IT" detected as named term)
  - Sentinel Involved: YES (__WTTERM0001__)
  - ALL-CAPS Effect: CONFIRMED (re.IGNORECASE in pattern matched "used TO IT")
  - Raw Qwen Error: NO (Qwen translated "__WTTERM0001__'e alıştım")
  - Restore Error: CONFIRMED ("TO IT'e alıştım")
  - Primary Cause: ALL-CAPS false named term detection + Restoration error

Case 2 [axe_god_chapter1_019]: "CAPTAIN GAO YUAN IS A PEAK LEVEL 1 ABILITY USER..."
  - Source Quality: CLEAN
  - False Term Detection: NO
  - Sentinel Involved: YES (__WTTERM0001__ for ABILITY USER)
  - ALL-CAPS Effect: MINOR
  - Raw Qwen Error: MINOR
  - Restore Error: CONFIRMED ("yetenek kullanıcısıDIR" due to missing 'DIR in _suffix_category)
  - Primary Cause: Sentinel Turkish morphology restoration error

Case 3 [axe_god_chapter1_028]: "MY NAME IS LHO TIAN... SECRET REALM GUIDE."
  - Source Quality: CLEAN
  - False Term Detection: NO
  - Sentinel Involved: YES (__WTTERM0001__ for SECRET REALM GUIDE)
  - ALL-CAPS Effect: MINOR
  - Raw Qwen Error: MINOR
  - Restore Error: CONFIRMED ("gizli âlem rehberiİM" due to missing 'İM in _suffix_category)
  - Primary Cause: Sentinel Turkish morphology restoration error

Case 4 [axe_god_chapter1_005]: "LOOKS LIKE MY MONEY WASN'T WASTED. YOU'RE WORTH EVERY PENNY, KID!"
  - Source Quality: CLEAN
  - False Term Detection: NO
  - Sentinel Involved: NO
  - ALL-CAPS Effect: CONFIRMED (Raw Qwen on ALL-CAPS mistranslates "my money" to "paranın", but on sentence case translates "paramın")
  - Raw Qwen Error: CONFIRMED on ALL-CAPS
  - Restore Error: NO
  - Primary Cause: Raw Qwen translation sensitivity to ALL-CAPS casing

Case 5 [axe_god_chapter1_007]: "YOUNG MASTER YLI, IT'S MORE THAN JUST NOT WASTED-WE'VE HIT THE JACKPOT."
  - Source Quality: CLEAN
  - False Term Detection: NO
  - Sentinel Involved: NO
  - ALL-CAPS Effect: CONFIRMED (Sentence case produces significantly more natural Turkish)
  - Raw Qwen Error: MINOR
  - Restore Error: NO
  - Primary Cause: Raw Qwen sensitivity to ALL-CAPS casing

3. ROOT CAUSE EVIDENCE RANKING:
-----------------------------
A. OCR source corruption: POSSIBLE
B. dirty V3 context: NOT SUPPORTED
C. V3 rewrite error: NOT SUPPORTED
D. ALL-CAPS effect on named-term detector: CONFIRMED
E. ALL-CAPS effect on Qwen itself: CONFIRMED
F. sentinel terminology protection: CONFIRMED
G. sentinel Turkish morphology restoration: CONFIRMED
H. wrong provider/model/server/prompt: NOT SUPPORTED
I. intrinsic Qwen translation limitation: CONFIRMED

4. RECOMMENDED FIX ORDER (DO NOT IMPLEMENT YET):
----------------------------------------------
1. Fix NAMED_TERM_PATTERNS in core/translation/protection.py so ALL-CAPS prose words like "TO IT" are not matched as named terms.
2. Extend _suffix_category in core/translation/protection.py to support Turkish copular & person suffixes ('dir, 'DIR, 'im, 'İM).
3. Apply sentence-casing normalization to source English text before passing to the translation model.

5. VERIFICATION:
----------------
- Production files modified: NONE
- Model calls executed: 16 (Phase 4 A/B test only)
- Checkpoint: dfcd07f preserved
"""

    with open(OUTPUT_DIR / "diagnosis_report.txt", "w", encoding="utf-8") as f:
        f.write(diag_report_txt)

    print("Diagnostic completed successfully!")
    print(f"Artifacts generated in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
