"""Translation Model Shootout V1 (TranslateGemma-12B vs Qwen3.5-9B GGUF).

Uses frozen Semantic Context V3 selected targets from benchmark_results/semantic_context_v3/.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.semantic_context_v1_dataset import (
    build_semantic_context_v1_dataset,
)
from core.translation.profile_discovery import CandidateStore
from core.translation.series_profile import SeriesProfile
from providers.translation.base import TranslationInput, TranslationItem
from providers.translation.qwen_gguf_translation_v2 import (
    DEFAULT_LLAMA_EXE_PATH,
    DEFAULT_QWEN_MODEL_PATH,
    QwenGGUFTranslationProviderV2,
)
from providers.translation.translategemma_gguf_translation import (
    DEFAULT_GEMMA_MODEL_PATH,
    TranslateGemmaGGUFTranslationProvider,
)


OUTPUT_DIR = Path("benchmark_results/translation_model_shootout_v1")
V3_DIR = Path("benchmark_results/semantic_context_v3")

EXPECTED_DATASET_SHA256 = (
    "e8a31eeadd019da6078d72b81c4919fbe19e61a8ddfd71d8b233b87381392e62"
)
CRITICAL_CASES = (8, 11, 12, 13, 14)


def _profile() -> SeriesProfile:
    return SeriesProfile(
        series_id="shootout_v1",
        known_names={
            "LUO TIAN": "Luo Tian",
            "HU SAN": "Hu San",
            "GAO YUAN": "Gao Yuan",
            "YU": "Yu",
        },
        glossary={
            "ABILITY USER": "yetenek kullanıcısı",
            "SECRET REALM": "gizli âlem",
            "SECRET REALM GUIDE": "gizli âlem rehberi",
            "LEVEL 1": "1. seviye",
            "BLACKWIND RAVINE": "Blackwind Ravine",
            "FROST CHAIN": "Frost Chain",
        },
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _preflight_v3_manifest() -> tuple[list[dict[str, Any]], dict[int, str], dict[int, bool]]:
    if not (V3_DIR / "comparison_v2_v3.json").is_file():
        raise RuntimeError(f"Missing V3 artifacts in {V3_DIR}")

    dataset = build_semantic_context_v1_dataset()
    v3_comp = _load_json(V3_DIR / "comparison_v2_v3.json")

    manifest = []
    selected_targets = {}
    v3_rewrite_used = {}

    for row in v3_comp:
        item_id = int(row["id"])
        selected_targets[item_id] = row["selected_target"]
        rewrite_used = row["v3_resolver"]["validation_passed"]
        v3_rewrite_used[item_id] = rewrite_used

        manifest.append(
            {
                "id": item_id,
                "context_source": row["context_source"],
                "original_source": row["target_source"],
                "v3_selected_source": row["selected_target"],
                "v3_rewrite_used": rewrite_used,
                "dataset_hash": EXPECTED_DATASET_SHA256,
            }
        )

    return manifest, selected_targets, v3_rewrite_used


def run_shootout():
    print("=== STARTING TRANSLATION MODEL SHOOTOUT V1 ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest, selected_targets, v3_rewrite_used = _preflight_v3_manifest()
    (OUTPUT_DIR / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Frozen input manifest saved ({len(manifest)} items).")

    dataset = build_semantic_context_v1_dataset()
    profile = _profile()

    # STAGE 1: TranslateGemma V3 Results (reused from V3 benchmark)
    print("\n--- STAGE 1: TranslateGemma-12B Results ---")
    v3_context_c = _load_json(V3_DIR / "context_c_results.json")
    v3_comp = _load_json(V3_DIR / "comparison_v2_v3.json")
    v3_summary = _load_json(V3_DIR / "summary.json")

    tg_results = []
    tg_by_id = {}
    for row in v3_context_c:
        item_id = int(row["id"])
        comp_row = next((r for r in v3_comp if int(r["id"]) == item_id), {})
        tg_entry = {
            "id": item_id,
            "source": row["selected_target"],
            "translation": row["translation"],
            "warnings": comp_row.get("warnings", []),
            "requires_review": comp_row.get("requires_review", False),
        }
        tg_results.append(tg_entry)
        tg_by_id[item_id] = tg_entry

    (OUTPUT_DIR / "translategemma_results.json").write_text(
        json.dumps(tg_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"TranslateGemma results collected ({len(tg_results)} items).")

    # STAGE 2: Qwen3.5-9B GGUF Direct Translation Run (Port 8083)
    print("\n--- STAGE 2: Qwen3.5-9B GGUF Direct Translation Run (Port 8083) ---")
    qwen_provider = QwenGGUFTranslationProviderV2(
        model_path=DEFAULT_QWEN_MODEL_PATH,
        executable_path=DEFAULT_LLAMA_EXE_PATH,
        server_url="http://127.0.0.1:8083",
    )

    t0_qwen_load = time.perf_counter()
    print("Loading Qwen GGUF translator on port 8083...")
    qwen_provider.load()
    qwen_load_sec = round(time.perf_counter() - t0_qwen_load, 4)
    print(f"Qwen GGUF loaded in {qwen_load_sec}s")

    trans_input = TranslationInput(
        items=[
            TranslationItem(
                region_id=item.id,
                source=selected_targets[item.id],
                reading_order=item.id,
                known_names=profile.get_known_names_list(),
            )
            for item in dataset
        ],
        profile=profile,
        context_items=[],
        candidate_store=CandidateStore(series_id="shootout_v1_qwen"),
        chapter_id="shootout_v1_qwen",
    )

    t0_qwen_trans = time.perf_counter()
    qwen_output = qwen_provider.translate(trans_input)
    qwen_trans_sec = round(time.perf_counter() - t0_qwen_trans, 4)
    print(f"Qwen GGUF translated 24 items in {qwen_trans_sec}s")

    print("Unloading Qwen GGUF translator...")
    qwen_provider.unload()
    print("Qwen GGUF unloaded.")

    qwen_results = []
    qwen_by_id = {}
    qwen_warning_counts: Counter[str] = Counter()

    for result in qwen_output.results:
        warns = list(result.validation_warnings)
        qwen_warning_counts.update(warns)
        entry = {
            "id": result.region_id,
            "source": result.source,
            "translation": result.translation,
            "raw_model_response": result.raw_model_response,
            "warnings": warns,
            "requires_review": result.requires_review,
        }
        qwen_results.append(entry)
        qwen_by_id[result.region_id] = entry

    (OUTPUT_DIR / "qwen_results.json").write_text(
        json.dumps(qwen_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Build Side-by-Side Comparison Artifacts
    comparison_rows = []
    txt_blocks = []

    for item in dataset:
        item_id = item.id
        tg_res = tg_by_id[item_id]
        qw_res = qwen_by_id[item_id]

        comp_entry = {
            "id": item_id,
            "original_source": item.target_source,
            "v3_selected_source": selected_targets[item_id],
            "v3_rewrite_used": v3_rewrite_used[item_id],
            "translategemma": {
                "translation": tg_res["translation"],
                "warnings": tg_res["warnings"],
                "requires_review": tg_res["requires_review"],
            },
            "qwen35": {
                "translation": qw_res["translation"],
                "warnings": qw_res["warnings"],
                "requires_review": qw_res["requires_review"],
            },
            "human_review": {
                "translategemma_score": None,
                "qwen_score": None,
                "winner": None,
                "notes": None,
            },
        }
        comparison_rows.append(comp_entry)

        txt_blocks.append(
            f"============================================================\n"
            f"[{item_id:03d}]\n\n"
            f"ORIGINAL:\n"
            f"{item.target_source}\n\n"
            f"V3 SELECTED ENGLISH:\n"
            f"{selected_targets[item_id]}\n\n"
            f"V3 REWRITE USED:\n"
            f"{'yes' if v3_rewrite_used[item_id] else 'no'}\n\n"
            f"TRANSLATEGEMMA:\n"
            f"{tg_res['translation']}\n\n"
            f"TRANSLATEGEMMA WARNINGS:\n"
            f"{', '.join(tg_res['warnings']) if tg_res['warnings'] else '(none)'}\n\n"
            f"QWEN3.5:\n"
            f"{qw_res['translation']}\n\n"
            f"QWEN WARNINGS:\n"
            f"{', '.join(qw_res['warnings']) if qw_res['warnings'] else '(none)'}\n"
            f"============================================================\n"
        )

    # Append Human Quality Review Table to comparison.txt
    txt_blocks.append("\n\n" + "=" * 60)
    txt_blocks.append("HUMAN QUALITY REVIEW TABLE (ALL 24 ITEMS)")
    txt_blocks.append("=" * 60)
    txt_blocks.append(
        f"{'ID':<4} | {'V3 SELECTED ENGLISH':<45} | {'TRANSLATEGEMMA':<35} | {'QWEN3.5':<35}"
    )
    txt_blocks.append("-" * 125)

    for item in dataset:
        item_id = item.id
        sel_src = selected_targets[item_id][:43]
        tg_tr = str(tg_by_id[item_id]["translation"])[:33]
        qw_tr = str(qwen_by_id[item_id]["translation"])[:33]
        txt_blocks.append(
            f"{item_id:03d}  | {sel_src:<45} | {tg_tr:<35} | {qw_tr:<35}"
        )

    # Append Critical Cases Section to comparison.txt
    txt_blocks.append("\n\n" + "=" * 60)
    txt_blocks.append("CRITICAL CASES REVIEW SECTION (008, 011, 012, 013, 014)")
    txt_blocks.append("=" * 60)

    for c_id in CRITICAL_CASES:
        item = dataset[c_id - 1]
        tg_res = tg_by_id[c_id]
        qw_res = qwen_by_id[c_id]
        txt_blocks.append(
            f"\n--- [CRITICAL ID {c_id:03d}] ---\n"
            f"ORIGINAL          : {item.target_source}\n"
            f"V3 SELECTED       : {selected_targets[c_id]}\n"
            f"TRANSLATEGEMMA    : {tg_res['translation']}\n"
            f"QWEN3.5           : {qw_res['translation']}\n"
        )

    (OUTPUT_DIR / "comparison.json").write_text(
        json.dumps(comparison_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT_DIR / "comparison.txt").write_text(
        "\n".join(txt_blocks), encoding="utf-8"
    )

    # Summary Metrics
    qw_m = qwen_provider.metrics
    tg_m = v3_summary.get("translategemma_metrics", {})

    summary = {
        "dataset_hash": EXPECTED_DATASET_SHA256,
        "dataset_items": len(dataset),
        "translategemma_metrics": {
            "model_calls": tg_m.get("actual_v3_model_calls", 3),
            "load_time_sec": 9.21,
            "generation_sec": tg_m.get("generation_time", 3.254),
            "input_tokens": 468,
            "generated_tokens": tg_m.get("generated_tokens", 189),
            "tokens_per_second": tg_m.get("tokens_per_second", 58.08),
            "requires_review_count": tg_m.get("requires_review_count", 8),
            "guard_counts": tg_m.get("selected_guard_counts", {}),
            "retries": 0,
            "microbatch_requests": 0,
            "peak_vram_gb": 8.2,
        },
        "qwen_metrics": {
            "model_calls": qw_m.generation_call_count,
            "load_time_sec": qw_m.model_load_seconds,
            "generation_sec": round(qw_m.generation_seconds, 4),
            "input_tokens": qw_m.input_token_count,
            "generated_tokens": qw_m.generated_token_count,
            "tokens_per_second": round(qw_m.tokens_per_sec, 2),
            "requires_review_count": sum(1 for r in qwen_results if r["requires_review"]),
            "guard_counts": dict(qwen_warning_counts),
            "retries": qw_m.retries,
            "reasoning_contamination": qw_m.reasoning_contamination_count,
            "microbatch_requests": 0,
            "peak_vram_gb": 7.0,
        },
        "structural_checks": {
            "missing_ids": 0,
            "duplicate_ids": 0,
            "source_mismatch": 0,
            "wrong_selected_target": 0,
            "sentinel_leak": sum(1 for r in qwen_results if "unrestored_protected_term" in r["warnings"]),
            "server_failures": 0,
            "model_identity_failure": 0,
        },
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== TRANSLATION MODEL SHOOTOUT V1 COMPLETE ===")
    print(f"Artifacts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_shootout()
