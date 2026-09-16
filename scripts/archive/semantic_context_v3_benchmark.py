"""One-shot Semantic Context V3 controlled-English bridge benchmark."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmark.semantic_context_v1_dataset import (
    ContextualTestItem,
    build_semantic_context_v1_dataset,
)
from core.translation.profile_discovery import CandidateStore
from core.translation.protection import contains_unrestored_protected_term
from core.translation.semantic_context import (
    DEFAULT_SEMANTIC_RESOLVER_MIN_CONFIDENCE,
    ControlledBridgeOutcome,
    SemanticContextRequest,
    render_controlled_english_bridge_prompt,
    resolve_controlled_bridge_with_fallback,
    validate_controlled_target,
)
from core.translation.series_profile import SeriesProfile
from providers.translation.base import TranslationInput, TranslationItem, TranslationOutput
from providers.translation.qwen_semantic_resolver import (
    DEFAULT_QWEN_SEMANTIC_CONTEXT_SIZE,
    DEFAULT_QWEN_SEMANTIC_GPU_LAYERS,
    DEFAULT_QWEN_SEMANTIC_LLAMA_EXE_PATH,
    DEFAULT_QWEN_SEMANTIC_MODEL_PATH,
    DEFAULT_QWEN_SEMANTIC_SERVER_URL,
    DEFAULT_QWEN_SEMANTIC_TEMPERATURE,
    QwenSemanticResolverProvider,
)
from providers.translation.translategemma_gguf_translation import (
    DEFAULT_GEMMA_MODEL_PATH,
    TranslateGemmaGGUFTranslationProvider,
    _PreparedTranslationItem,
    contains_segment_marker,
)


OUTPUT_DIR = Path("benchmark_results/semantic_context_v3")
V1_DIR = Path("benchmark_results/semantic_context_v1")
V2_DIR = Path("benchmark_results/semantic_context_v2")

EXPECTED_DATASET_SHA256 = (
    "e8a31eeadd019da6078d72b81c4919fbe19e61a8ddfd71d8b233b87381392e62"
)
EXPECTED_QWEN_MODEL_PATH = (
    r"C:\AI\Models\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q5_K_M.gguf"
)
EXPECTED_QWEN_LLAMA_EXE_PATH = r"C:\AI\llama-cpp-cuda\llama.exe"
EXPECTED_TRANSLATEGEMMA_MODEL_PATH = (
    r"C:\AI\Models\translategemma-12b-it-q5_k_m.gguf"
)
MIN_CONFIDENCE = DEFAULT_SEMANTIC_RESOLVER_MIN_CONFIDENCE
SPECIAL_REVIEW_IDS = (8, 11, 12, 13, 14, 19, 22, 23)
CONTROL_CASES = (15, 16, 17, 18, 20, 21, 22, 23, 24)


class _CapturingTranslateGemmaProvider(TranslateGemmaGGUFTranslationProvider):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.prepared_sources: dict[int, str] = {}

    def _translate_single_prepared(
        self,
        prepared: _PreparedTranslationItem,
        **kwargs: Any,
    ):
        self.prepared_sources.setdefault(
            prepared.item.region_id,
            prepared.prepared_text,
        )
        return super()._translate_single_prepared(prepared, **kwargs)


def _profile() -> SeriesProfile:
    return SeriesProfile(
        series_id="semantic_context_v3",
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


def _make_translation_input(
    dataset: list[ContextualTestItem],
    selected_targets: dict[int, str],
) -> TranslationInput:
    profile = _profile()
    return TranslationInput(
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
        candidate_store=CandidateStore(series_id="semantic_context_v3_context_c"),
        chapter_id="semantic_context_v3_context_c",
    )


def _bypass_type(raw_model_response: str | None) -> str | None:
    if raw_model_response == "[SYSTEM_UI_BYPASS]":
        return "system_ui"
    if raw_model_response == "[TERM_ONLY_BYPASS]":
        return "term_only"
    return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _preflight_alignment() -> tuple[
    bytes,
    list[ContextualTestItem],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    dataset_bytes = (V1_DIR / "dataset.json").read_bytes()
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "Frozen dataset hash mismatch: "
            f"expected {EXPECTED_DATASET_SHA256}, got {dataset_hash}"
        )
    dataset = build_semantic_context_v1_dataset()
    baseline_rows = _load_json(V1_DIR / "baseline_c_results.json")
    v2_rows = _load_json(V2_DIR / "comparison_v1_v2.json") if (V2_DIR / "comparison_v1_v2.json").is_file() else []
    v2_resolver_rows = _load_json(V2_DIR / "resolver_outputs.json") if (V2_DIR / "resolver_outputs.json").is_file() else []
    return dataset_bytes, dataset, baseline_rows, v2_rows


def _resolution_view(outcome: ControlledBridgeOutcome) -> dict[str, Any]:
    resolution = outcome.resolution
    return {
        "raw_response": outcome.raw_response,
        "rewrite_needed": resolution.rewrite_needed if resolution else None,
        "confidence": resolution.confidence if resolution else None,
        "risk_types": list(resolution.risk_types) if resolution else [],
        "semantic_notes": (
            [
                {
                    "span": note.span,
                    "resolved_meaning": note.resolved_meaning,
                    "evidence": note.evidence,
                }
                for note in resolution.semantic_notes
            ]
            if resolution
            else []
        ),
        "question_word": resolution.question_word if resolution else None,
        "tense_aspect": resolution.tense_aspect if resolution else None,
        "referents": list(resolution.referents) if resolution else [],
        "controlled_target": resolution.controlled_target if resolution else None,
    }


def run_benchmark():
    print("=== STARTING SEMANTIC CONTEXT V3 BENCHMARK ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_bytes, dataset, baseline_rows, v2_rows = _preflight_alignment()

    v1_baseline_by_id = {int(row["id"]): row for row in baseline_rows}
    v2_by_id = {int(row["id"]): row for row in v2_rows}

    # STEP 1: Qwen Controlled-English Resolver Run
    print("\n--- STAGE 1: Qwen GGUF Controlled-English Resolver ---")
    resolver = QwenSemanticResolverProvider(
        server_url=DEFAULT_QWEN_SEMANTIC_SERVER_URL,
        prompt_renderer=render_controlled_english_bridge_prompt,
    )

    print("Loading Qwen GGUF model...")
    t_start_load = time.perf_counter()
    resolver.load()
    t_qwen_load = round(time.perf_counter() - t_start_load, 4)
    print(f"Qwen GGUF loaded in {t_qwen_load}s")

    resolver_outcomes: dict[int, ControlledBridgeOutcome] = {}

    for item in dataset:
        req = SemanticContextRequest(
            previous_context=item.previous_context,
            target_source=item.target_source,
            next_context=item.next_context,
            named_terms=item.named_terms,
        )
        print(f"Resolving [{item.id:03d}] {item.target_source[:40]}...")
        outcome = resolve_controlled_bridge_with_fallback(req, resolver.resolve)
        resolver_outcomes[item.id] = outcome

    qwen_metrics = resolver.metrics
    print("\nUnloading Qwen GGUF...")
    resolver.unload()
    print("Qwen GGUF unloaded.")

    # Process Resolver Outputs
    resolver_rows = []
    selected_targets = {}
    accepted_rewrites = 0
    rejected_rewrites = 0
    uncertain_fallbacks = 0
    low_confidence_fallbacks = 0
    unchanged_targets = 0
    validation_reason_counts: Counter[str] = Counter()

    for item in dataset:
        outcome = resolver_outcomes[item.id]
        res = outcome.resolution
        dec = outcome.decision

        selected_targets[item.id] = dec.selected_target

        if dec.rewrite_used:
            accepted_rewrites += 1
        else:
            if dec.rejection_reason == "controlled_validation_failed":
                rejected_rewrites += 1
                for f in dec.validation_failures:
                    validation_reason_counts[f] += 1
                if dec.validator_uncertain:
                    uncertain_fallbacks += 1
            elif dec.rejection_reason == "low_confidence":
                low_confidence_fallbacks += 1
            elif dec.rejection_reason in ("rewrite_not_needed", "unchanged_target"):
                unchanged_targets += 1

        resolver_rows.append(
            {
                "id": item.id,
                "target_source": item.target_source,
                "selected_target": dec.selected_target,
                "rewrite_used": dec.rewrite_used,
                "rejection_reason": dec.rejection_reason,
                "validation_failures": list(dec.validation_failures),
                "validator_uncertain": dec.validator_uncertain,
                "resolution": _resolution_view(outcome),
            }
        )

    (OUTPUT_DIR / "dataset.json").write_bytes(dataset_bytes)
    (OUTPUT_DIR / "resolver_outputs.json").write_text(
        json.dumps(resolver_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    selected_rows = [
        {"id": item_id, "target_source": dataset[item_id - 1].target_source, "selected_target": target}
        for item_id, target in selected_targets.items()
    ]
    (OUTPUT_DIR / "selected_targets.json").write_text(
        json.dumps(selected_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # STAGE 2: TranslateGemma Run on Accepted Rewrites
    print("\n--- STAGE 2: TranslateGemma Variant C Execution ---")
    gemma_provider = _CapturingTranslateGemmaProvider(
        model_path=EXPECTED_TRANSLATEGEMMA_MODEL_PATH,
        executable_path=EXPECTED_QWEN_LLAMA_EXE_PATH,
    )

    items_to_translate = [
        item for item in dataset if resolver_outcomes[item.id].decision.rewrite_used
    ]
    print(f"Items requiring TranslateGemma execution: {len(items_to_translate)} / 24")

    tg_results_by_id: dict[int, str] = {}
    tg_wall_time = 0.0

    if items_to_translate:
        trans_input = _make_translation_input(items_to_translate, selected_targets)
        t_start_tg = time.perf_counter()
        print("Loading TranslateGemma...")
        gemma_provider.load()
        output = gemma_provider.translate(trans_input)
        gemma_provider.unload()
        tg_wall_time = time.perf_counter() - t_start_tg
        print(f"TranslateGemma finished in {tg_wall_time:.2f}s")

        for res in output.results:
            tg_results_by_id[res.region_id] = res.translation

    # Combine with baseline for non-rewritten targets
    final_translations: dict[int, str] = {}
    baseline_reused_count = 0

    for item in dataset:
        if item.id in tg_results_by_id:
            final_translations[item.id] = tg_results_by_id[item.id]
        else:
            final_translations[item.id] = v1_baseline_by_id[item.id]["translation"]
            baseline_reused_count += 1

    # Write context_c_results.json
    context_c_rows = []
    for item in dataset:
        res_outcome = resolver_outcomes[item.id]
        dec = res_outcome.decision
        context_c_rows.append(
            {
                "id": item.id,
                "source": item.target_source,
                "selected_target": dec.selected_target,
                "rewrite_used": dec.rewrite_used,
                "translation": final_translations[item.id],
            }
        )
    (OUTPUT_DIR / "context_c_results.json").write_text(
        json.dumps(context_c_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Build per-item V2 vs V3 comparison artifacts
    comparison_rows = []
    txt_blocks = []

    rewritten_correct_controls = 0

    for item in dataset:
        outcome = resolver_outcomes[item.id]
        res = outcome.resolution
        dec = outcome.decision
        b_row = v1_baseline_by_id[item.id]
        v2_item = v2_by_id.get(item.id, {})
        v2_res = v2_item.get("v2_resolver", {})
        v2_target = v2_res.get("clarified_target") or item.target_source

        orig_len = len(item.target_source)
        ctrl_len = len(dec.selected_target)
        ratio = round(ctrl_len / orig_len, 2) if orig_len else 1.0

        if item.id in CONTROL_CASES and dec.rewrite_used:
            rewritten_correct_controls += 1

        comp_entry = {
            "id": item.id,
            "context_source": "SYNTHETIC" if item.synthetic_context else "REAL",
            "previous_context": list(item.previous_context),
            "target_source": item.target_source,
            "next_context": list(item.next_context),
            "v1_baseline_c": {
                "translation": b_row["translation"],
            },
            "v2": {
                "rewrite_used": v2_item.get("v2_decision", {}).get("rewrite_used", False),
                "clarified_target": v2_target,
                "translation": v2_item.get("v2_translation", b_row["translation"]),
            },
            "v3_resolver": {
                "rewrite_needed": res.rewrite_needed if res else False,
                "confidence": res.confidence if res else 0.0,
                "risk_types": list(res.risk_types) if res else [],
                "semantic_notes": (
                    [
                        {
                            "span": n.span,
                            "resolved_meaning": n.resolved_meaning,
                            "evidence": n.evidence,
                        }
                        for n in res.semantic_notes
                    ]
                    if res
                    else []
                ),
                "question_word": res.question_word if res else None,
                "referents": list(res.referents) if res else [],
                "tense_aspect": res.tense_aspect if res else None,
                "controlled_target": res.controlled_target if res else item.target_source,
                "validation_passed": dec.rewrite_used,
                "validation_reasons": list(dec.validation_failures),
                "validator_uncertain": dec.validator_uncertain,
            },
            "selected_target": dec.selected_target,
            "translation": final_translations[item.id],
            "warnings": [],
            "requires_review": item.id in SPECIAL_REVIEW_IDS,
            "original_length": orig_len,
            "controlled_length": ctrl_len,
            "length_ratio": ratio,
        }
        comparison_rows.append(comp_entry)

        # Build TXT comparison entry
        txt_blocks.append(
            f"============================================================\n"
            f"[{item.id:03d}]\n\n"
            f"PREVIOUS:\n"
            f"{' / '.join(item.previous_context) if item.previous_context else '(none)'}\n\n"
            f"ORIGINAL TARGET:\n"
            f"{item.target_source}\n\n"
            f"V2 TARGET:\n"
            f"{v2_target}\n\n"
            f"V3 CONTROLLED TARGET:\n"
            f"{dec.selected_target}\n\n"
            f"V3 RESOLVER:\n"
            f"rewrite_needed: {res.rewrite_needed if res else False}\n"
            f"confidence: {res.confidence if res else 0.0}\n"
            f"risk_types: {', '.join(res.risk_types) if res and res.risk_types else '(none)'}\n\n"
            f"VALIDATION:\n"
            f"{'PASS' if dec.rewrite_used else 'REJECT/FALLBACK (' + str(dec.rejection_reason) + ')'}\n\n"
            f"BASELINE C:\n"
            f"{b_row['translation']}\n\n"
            f"V2:\n"
            f"{v2_item.get('v2_translation', b_row['translation'])}\n\n"
            f"V3:\n"
            f"{final_translations[item.id]}\n\n"
            f"LENGTH RATIO: {ratio} ({orig_len} -> {ctrl_len})\n"
            f"============================================================\n"
        )

    (OUTPUT_DIR / "comparison_v2_v3.json").write_text(
        json.dumps(comparison_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT_DIR / "comparison_v2_v3.txt").write_text(
        "\n".join(txt_blocks), encoding="utf-8"
    )

    # Summary Metrics
    tg_metrics = gemma_provider.metrics
    summary = {
        "dataset_hash": EXPECTED_DATASET_SHA256,
        "dataset_items": len(dataset),
        "qwen_metrics": {
            "resolver_calls": qwen_metrics.resolver_calls,
            "provider_failures": qwen_metrics.resolver_failures,
            "malformed_json": sum(1 for o in resolver_outcomes.values() if o.malformed_json),
            "rewrite_needed_true": sum(
                1 for o in resolver_outcomes.values() if o.resolution and o.resolution.rewrite_needed
            ),
            "rewrite_needed_false": sum(
                1 for o in resolver_outcomes.values() if o.resolution and not o.resolution.rewrite_needed
            ),
            "accepted_rewrites": accepted_rewrites,
            "rejected_rewrites": rejected_rewrites,
            "uncertain_fallbacks": uncertain_fallbacks,
            "low_confidence_fallbacks": low_confidence_fallbacks,
            "unchanged_targets": unchanged_targets,
            "validation_reasons": dict(validation_reason_counts),
            "average_sec_per_item": round(qwen_metrics.average_resolver_seconds, 4),
            "total_generation_sec": round(qwen_metrics.generation_seconds, 4),
            "input_tokens": qwen_metrics.input_token_count,
            "generated_tokens": qwen_metrics.generated_token_count,
            "tokens_per_second": round(qwen_metrics.tokens_per_second, 2),
            "model_load_sec": round(qwen_metrics.model_load_seconds, 4),
            "peak_vram_gb": round(qwen_metrics.peak_vram_gb, 4),
        },
        "translategemma_metrics": {
            "actual_v3_model_calls": tg_metrics.generation_call_count,
            "baseline_reused_count": baseline_reused_count,
            "generation_time": round(tg_metrics.generation_seconds, 4),
            "generated_tokens": tg_metrics.generated_token_count,
            "tokens_per_second": round(tg_metrics.tokens_per_sec, 2),
            "requires_review_count": sum(1 for c in comparison_rows if c["requires_review"]),
            "micro_batch_requests": tg_metrics.micro_batch_requests,
        },
        "selectivity_metrics": {
            "rewritten_correct_controls": rewritten_correct_controls,
            "total_control_cases": len(CONTROL_CASES),
        },
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== SEMANTIC CONTEXT V3 BENCHMARK COMPLETE ===")
    print(f"Artifacts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_benchmark()
