"""One-shot Qwen semantic-context resolver → TranslateGemma Variant C benchmark."""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmark.semantic_context_v1_dataset import (
    REAL_CHAPTER_SEQUENCE,
    ContextualTestItem,
    build_semantic_context_v1_dataset,
)
from core.translation.profile_discovery import CandidateStore
from core.translation.protection import contains_unrestored_protected_term
from core.translation.semantic_context import (
    DEFAULT_SEMANTIC_RESOLVER_MIN_CONFIDENCE,
    ResolverOutcome,
    SemanticContextRequest,
    TranslationExperimentMode,
    render_semantic_resolver_prompt,
    resolve_with_fallback,
    validate_clarified_target,
)
from core.translation.series_profile import SeriesProfile
from providers.translation.base import TranslationInput, TranslationItem, TranslationOutput
from providers.translation.qwen_semantic_resolver import (
    DEFAULT_QWEN_SEMANTIC_MODEL_PATH,
    QwenSemanticResolverProvider,
)
from providers.translation.translategemma_gguf_translation import (
    DEFAULT_GEMMA_MODEL_PATH,
    TranslateGemmaGGUFTranslationProvider,
    _PreparedTranslationItem,
    contains_segment_marker,
)


OUTPUT_DIR = Path("benchmark_results/semantic_context_v1")
EXPECTED_QWEN_MODEL_PATH = r"C:\AI\Models\Qwen3.5-9B"
EXPECTED_TRANSLATEGEMMA_MODEL_PATH = (
    r"C:\AI\Models\translategemma-12b-it-q5_k_m.gguf"
)
MIN_CONFIDENCE = DEFAULT_SEMANTIC_RESOLVER_MIN_CONFIDENCE


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
        series_id="semantic_context_v1",
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
    mode: TranslationExperimentMode,
) -> TranslationInput:
    profile = _profile()
    items = [
        TranslationItem(
            region_id=item.id,
            source=selected_targets[item.id],
            reading_order=item.id,
            known_names=profile.get_known_names_list(),
        )
        for item in dataset
    ]
    return TranslationInput(
        items=items,
        profile=profile,
        context_items=[],
        candidate_store=CandidateStore(series_id=f"semantic_context_v1_{mode.value}"),
        chapter_id=f"semantic_context_v1_{mode.value}",
    )


def _bypass_type(raw_model_response: str) -> str | None:
    if raw_model_response == "[System UI Lexicon]":
        return "SYSTEM_UI"
    if raw_model_response == "[Term-Only Bypass]":
        return "TERM_ONLY"
    return None


def _collect_translation(
    provider: _CapturingTranslateGemmaProvider,
    output: TranslationOutput,
    wall_time: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()
    for result in output.results:
        warnings = list(result.validation_warnings)
        warning_counts.update(warnings)
        bypass = _bypass_type(result.raw_model_response)
        rows.append(
            {
                "id": result.region_id,
                "source": result.source,
                "prepared_source": provider.prepared_sources.get(result.region_id),
                "translation": result.translation,
                "raw_model_response": result.raw_model_response,
                "warnings": warnings,
                "requires_review": result.requires_review,
                "model_called": bypass is None,
                "bypass_type": bypass,
                "micro_batch_id": result.micro_batch_id,
                "micro_batch_region_ids": list(result.micro_batch_region_ids),
            }
        )

    metrics = provider.metrics
    summary = {
        "items": len(rows),
        "model_calls": metrics.generation_call_count,
        "system_ui_bypass": metrics.system_ui_bypass_count,
        "term_only_bypass": metrics.term_only_bypass_count,
        "requires_review": sum(row["requires_review"] for row in rows),
        "guard_counts": dict(sorted(warning_counts.items())),
        "generation_seconds": round(metrics.generation_seconds, 4),
        "wall_time": round(wall_time, 4),
        "input_tokens": metrics.input_token_count,
        "generated_tokens": metrics.generated_token_count,
        "tokens_per_second": metrics.tokens_per_sec,
        "retries": metrics.retries,
        "micro_batch_requests": metrics.micro_batch_requests,
    }
    return rows, summary


def _resolution_view(
    item: ContextualTestItem,
    outcome: ResolverOutcome,
) -> dict[str, Any]:
    resolution = outcome.resolution
    if resolution is None:
        parsed = {
            "ambiguous": None,
            "confidence": None,
            "semantic_notes": [],
            "question_type": None,
            "tense_aspect": None,
            "referents": [],
            "register_hint": None,
            "clarified_target": item.target_source,
        }
    else:
        parsed = {
            "ambiguous": resolution.ambiguous,
            "confidence": resolution.confidence,
            "semantic_notes": [asdict(note) for note in resolution.semantic_notes],
            "question_type": resolution.question_type,
            "tense_aspect": resolution.tense_aspect,
            "referents": list(resolution.referents),
            "register_hint": resolution.register_hint,
            "clarified_target": resolution.clarified_target,
        }
    return {
        **parsed,
        "selected_target": outcome.decision.selected_target,
        "clarification_used": outcome.decision.clarification_used,
        "rejection_reason": outcome.decision.rejection_reason,
        "validation_failures": list(outcome.decision.validation_failures),
        "resolver_failed": outcome.resolver_failed,
        "malformed_json": outcome.malformed_json,
        "raw_response": outcome.raw_response,
        "rendered_prompt": render_semantic_resolver_prompt(outcome.request),
    }


def _duplicates(rows: list[dict[str, Any]]) -> list[int]:
    ids = [row["id"] for row in rows]
    return sorted({region_id for region_id in ids if ids.count(region_id) > 1})


def _format_warnings(warnings: list[str]) -> str:
    return ", ".join(warnings) if warnings else "none"


def _comparison_text(rows: list[dict[str, Any]]) -> str:
    separator = "=" * 60
    lines: list[str] = []
    for row in rows:
        resolver = row["resolver"]
        notes = resolver["semantic_notes"]
        if notes:
            note_lines = [
                f"{note['span']} -> {note['intended_sense']} ({note['evidence']})"
                for note in notes
            ]
        else:
            note_lines = ["none"]
        lines.extend(
            [
                separator,
                f"[{row['id']:03d}]",
                f"CONTEXT SOURCE: {'SYNTHETIC' if row['synthetic_context'] else 'REAL'}",
                f"PROVENANCE: {row['source_origin']}",
                "",
                "PREVIOUS CONTEXT:",
            ]
        )
        lines.extend(
            [f"{index}. {source}" for index, source in enumerate(row["previous_context"], 1)]
            or ["(none)"]
        )
        lines.extend(["", "TARGET:", row["target_source"], "", "NEXT CONTEXT:"])
        lines.extend(
            [f"{index}. {source}" for index, source in enumerate(row["next_context"], 1)]
            or ["(none)"]
        )
        lines.extend(
            [
                "",
                "QWEN:",
                f"ambiguous: {resolver['ambiguous']}",
                f"confidence: {resolver['confidence']}",
                "",
                "SEMANTIC NOTES:",
                *note_lines,
                "",
                "CLARIFIED TARGET:",
                resolver["clarified_target"],
                "",
                "SELECTED TARGET:",
                resolver["selected_target"],
                "",
                "CLARIFICATION USED:",
                "yes" if resolver["clarification_used"] else "no",
                "",
                "REJECTION/FALLBACK REASON:",
                resolver["rejection_reason"] or "none",
                "",
                "BASELINE C:",
                row["baseline"]["translation"] or "<FAILED>",
                "",
                "CONTEXT C:",
                row["contextual"]["translation"] or "<FAILED>",
                "",
                "BASELINE WARNINGS:",
                _format_warnings(row["baseline"]["warnings"]),
                "",
                "CONTEXT WARNINGS:",
                _format_warnings(row["contextual"]["warnings"]),
                separator,
                "",
            ]
        )
    return "\n".join(lines)


def _real_context_order_mismatches(dataset: list[ContextualTestItem]) -> list[int]:
    real_sources = [source for _, source in REAL_CHAPTER_SEQUENCE]
    real_items = [item for item in dataset if not item.synthetic_context]
    mismatches: list[int] = []
    for index, item in enumerate(real_items):
        expected_previous = tuple(real_sources[max(0, index - 3) : index])
        expected_next = tuple(real_sources[index + 1 : index + 2])
        if (
            item.target_source != real_sources[index]
            or item.previous_context != expected_previous
            or item.next_context != expected_next
        ):
            mismatches.append(item.id)
    return mismatches


def run_semantic_context_v1() -> None:
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"Fresh-run guard: output directory already exists: {OUTPUT_DIR}")
    dataset = build_semantic_context_v1_dataset()
    expected_ids = [item.id for item in dataset]
    original_targets = {item.id: item.target_source for item in dataset}

    qwen = QwenSemanticResolverProvider()
    if qwen.model_path != EXPECTED_QWEN_MODEL_PATH:
        raise RuntimeError(f"Unexpected Qwen model path: {qwen.model_path}")

    outcomes: dict[int, ResolverOutcome] = {}
    qwen.load()
    try:
        for item in dataset:
            request = SemanticContextRequest(
                previous_context=item.previous_context,
                target_source=item.target_source,
                next_context=item.next_context,
                named_terms=item.named_terms,
            )
            outcomes[item.id] = resolve_with_fallback(
                request,
                qwen.resolve,
                min_confidence=MIN_CONFIDENCE,
            )
    finally:
        qwen.unload()

    selected_targets = {
        region_id: outcome.decision.selected_target
        for region_id, outcome in outcomes.items()
    }
    baseline_input = _make_translation_input(
        dataset,
        original_targets,
        TranslationExperimentMode.BASELINE_C,
    )
    contextual_input = _make_translation_input(
        dataset,
        selected_targets,
        TranslationExperimentMode.SEMANTIC_CONTEXT_C,
    )

    baseline_provider = _CapturingTranslateGemmaProvider(
        managed=True,
        micro_batch_enabled=False,
        prompt_variant="minimal_faithful",
    )
    contextual_provider = _CapturingTranslateGemmaProvider(
        managed=False,
        micro_batch_enabled=False,
        prompt_variant="minimal_faithful",
    )
    if baseline_provider.model_path != EXPECTED_TRANSLATEGEMMA_MODEL_PATH:
        raise RuntimeError(
            f"Unexpected TranslateGemma model path: {baseline_provider.model_path}"
        )

    baseline_provider.load()
    try:
        baseline_started = time.perf_counter()
        baseline_output = baseline_provider.translate(baseline_input)
        baseline_wall = time.perf_counter() - baseline_started

        contextual_started = time.perf_counter()
        contextual_output = contextual_provider.translate(contextual_input)
        contextual_wall = time.perf_counter() - contextual_started
    finally:
        baseline_provider.unload()

    baseline_rows, baseline_summary = _collect_translation(
        baseline_provider,
        baseline_output,
        baseline_wall,
    )
    contextual_rows, contextual_summary = _collect_translation(
        contextual_provider,
        contextual_output,
        contextual_wall,
    )
    by_id_baseline = {row["id"]: row for row in baseline_rows}
    by_id_contextual = {row["id"]: row for row in contextual_rows}

    resolver_rows = [
        {
            "id": item.id,
            "synthetic_context": item.synthetic_context,
            "source_origin": item.source_origin,
            "previous_context": list(item.previous_context),
            "target_source": item.target_source,
            "next_context": list(item.next_context),
            "named_terms": list(item.named_terms),
            **_resolution_view(item, outcomes[item.id]),
        }
        for item in dataset
    ]
    by_id_resolver = {row["id"]: row for row in resolver_rows}

    comparison_rows: list[dict[str, Any]] = []
    for item in dataset:
        baseline = by_id_baseline[item.id]
        contextual = by_id_contextual[item.id]
        resolver = by_id_resolver[item.id]
        comparison_rows.append(
            {
                "id": item.id,
                "synthetic_context": item.synthetic_context,
                "source_origin": item.source_origin,
                "previous_context": list(item.previous_context),
                "target_source": item.target_source,
                "next_context": list(item.next_context),
                "baseline": {
                    "translation": baseline["translation"],
                    "warnings": baseline["warnings"],
                    "requires_review": baseline["requires_review"],
                    "prepared_source": baseline["prepared_source"],
                },
                "resolver": {
                    key: resolver[key]
                    for key in (
                        "ambiguous",
                        "confidence",
                        "semantic_notes",
                        "question_type",
                        "tense_aspect",
                        "referents",
                        "register_hint",
                        "clarified_target",
                        "selected_target",
                        "clarification_used",
                        "rejection_reason",
                        "validation_failures",
                        "resolver_failed",
                        "malformed_json",
                    )
                },
                "contextual": {
                    "translation": contextual["translation"],
                    "warnings": contextual["warnings"],
                    "requires_review": contextual["requires_review"],
                    "prepared_source": contextual["prepared_source"],
                },
            }
        )

    validation_failure_ids: dict[str, list[int]] = {
        "named_term_loss": [],
        "number_changed": [],
        "polarity_changed": [],
        "question_type_changed": [],
        "clarified_target_not_english": [],
        "empty_clarified_target": [],
        "unsupported_named_entity": [],
    }
    failed_validation_ids: list[int] = []
    selected_target_safety_failure_ids: list[int] = []
    for item in dataset:
        failures = outcomes[item.id].decision.validation_failures
        if failures:
            failed_validation_ids.append(item.id)
        for failure in failures:
            validation_failure_ids.setdefault(failure, []).append(item.id)
        selected_failures = validate_clarified_target(
            item.target_source,
            selected_targets[item.id],
            named_terms=item.named_terms,
            context_sources=(*item.previous_context, *item.next_context),
        )
        if selected_failures:
            selected_target_safety_failure_ids.append(item.id)

    def leak_ids(rows: list[dict[str, Any]], predicate) -> list[int]:
        return [
            row["id"]
            for row in rows
            if row["translation"] and predicate(row["translation"])
        ]

    baseline_server_errors = [
        row["id"]
        for row in baseline_rows
        if "translation_server_error" in row["warnings"]
    ]
    contextual_server_errors = [
        row["id"]
        for row in contextual_rows
        if "translation_server_error" in row["warnings"]
    ]
    structural = {
        "missing_baseline_ids": sorted(set(expected_ids) - set(by_id_baseline)),
        "missing_contextual_ids": sorted(set(expected_ids) - set(by_id_contextual)),
        "duplicate_baseline_ids": _duplicates(baseline_rows),
        "duplicate_contextual_ids": _duplicates(contextual_rows),
        "context_order_mismatch_ids": _real_context_order_mismatches(dataset),
        "resolver_proposal_named_term_loss_ids": validation_failure_ids[
            "named_term_loss"
        ],
        "resolver_proposal_number_change_ids": validation_failure_ids[
            "number_changed"
        ],
        "resolver_proposal_polarity_change_ids": validation_failure_ids[
            "polarity_changed"
        ],
        "resolver_proposal_question_type_change_ids": validation_failure_ids[
            "question_type_changed"
        ],
        "failed_clarification_validation_ids": failed_validation_ids,
        "selected_target_safety_failure_ids": selected_target_safety_failure_ids,
        "baseline_sentinel_leak_ids": leak_ids(
            baseline_rows, contains_unrestored_protected_term
        ),
        "contextual_sentinel_leak_ids": leak_ids(
            contextual_rows, contains_unrestored_protected_term
        ),
        "baseline_segment_marker_leak_ids": leak_ids(
            baseline_rows, contains_segment_marker
        ),
        "contextual_segment_marker_leak_ids": leak_ids(
            contextual_rows, contains_segment_marker
        ),
        "baseline_server_error_ids": baseline_server_errors,
        "contextual_server_error_ids": contextual_server_errors,
        "baseline_micro_batch_requests": baseline_provider.metrics.micro_batch_requests,
        "contextual_micro_batch_requests": contextual_provider.metrics.micro_batch_requests,
    }
    final_path_keys = (
        "missing_baseline_ids",
        "missing_contextual_ids",
        "duplicate_baseline_ids",
        "duplicate_contextual_ids",
        "context_order_mismatch_ids",
        "selected_target_safety_failure_ids",
        "baseline_sentinel_leak_ids",
        "contextual_sentinel_leak_ids",
        "baseline_segment_marker_leak_ids",
        "contextual_segment_marker_leak_ids",
        "baseline_server_error_ids",
        "contextual_server_error_ids",
        "baseline_micro_batch_requests",
        "contextual_micro_batch_requests",
    )
    structural["final_path_clean"] = not any(structural[key] for key in final_path_keys)

    if baseline_provider.metrics.micro_batch_requests:
        raise RuntimeError("Baseline C issued a forbidden micro-batch request")
    if contextual_provider.metrics.micro_batch_requests:
        raise RuntimeError("Semantic Context C issued a forbidden micro-batch request")
    if selected_target_safety_failure_ids:
        raise RuntimeError(
            "Unsafe selected targets reached the translation boundary: "
            f"{selected_target_safety_failure_ids}"
        )

    rejection_counts = Counter(
        outcome.decision.rejection_reason
        for outcome in outcomes.values()
        if outcome.decision.rejection_reason
    )
    qwen_summary = {
        "model": qwen.name,
        "model_path": DEFAULT_QWEN_SEMANTIC_MODEL_PATH,
        "quantization": "8-bit",
        "resolver_calls": qwen.metrics.resolver_calls,
        "resolver_failures": qwen.metrics.resolver_failures,
        "malformed_json": sum(outcome.malformed_json for outcome in outcomes.values()),
        "low_confidence_count": rejection_counts["low_confidence"],
        "clarification_accepted": sum(
            outcome.decision.clarification_used for outcome in outcomes.values()
        ),
        "clarification_rejected": rejection_counts[
            "clarification_validation_failed"
        ],
        "unchanged_targets": (
            rejection_counts["unchanged_target"] + rejection_counts["not_ambiguous"]
        ),
        "fallback_reason_counts": dict(sorted(rejection_counts.items())),
        "average_resolver_seconds": round(
            qwen.metrics.average_resolver_seconds,
            4,
        ),
        "generation_seconds": round(qwen.metrics.generation_seconds, 4),
        "input_tokens": qwen.metrics.input_token_count,
        "generated_tokens": qwen.metrics.generated_token_count,
        "model_load_seconds": qwen.metrics.model_load_seconds,
        "model_load_vram_gb": qwen.metrics.model_load_vram_gb,
        "peak_vram_gb": round(qwen.metrics.peak_vram_gb, 4),
    }
    summary = {
        "dataset": {
            "items": len(dataset),
            "real_items": sum(not item.synthetic_context for item in dataset),
            "synthetic_items": sum(item.synthetic_context for item in dataset),
            "real_source": "scripts/qwen_translation_smoke_test.py::BUBBLES",
            "synthetic_source": "benchmark/semantic_context_v1_dataset.py::SYNTHETIC_ITEMS",
        },
        "experiment_modes": [
            TranslationExperimentMode.BASELINE_C.value,
            TranslationExperimentMode.SEMANTIC_CONTEXT_C.value,
        ],
        "semantic_resolver_min_confidence": MIN_CONFIDENCE,
        "model_sequence": [
            "load_qwen_resolver",
            "resolve_all_targets",
            "unload_qwen_resolver",
            "load_translategemma",
            "run_baseline_c_single_item",
            "run_semantic_context_c_single_item",
            "unload_translategemma",
        ],
        "qwen": qwen_summary,
        "translategemma_baseline_c": baseline_summary,
        "translategemma_semantic_context_c": contextual_summary,
        "structural": structural,
        "quality_winner": None,
        "quality_winner_note": (
            "Manual human review required; no semantic heuristic or auto-ranking performed."
        ),
        "production_default": "legacy",
    }

    contextual_result_rows = [
        {
            **row,
            "original_target_source": original_targets[row["id"]],
            "clarification_used": outcomes[row["id"]].decision.clarification_used,
            "rejection_reason": outcomes[row["id"]].decision.rejection_reason,
        }
        for row in contextual_rows
    ]
    serialized = {
        "dataset.json": json.dumps(
            [item.to_dict() for item in dataset], ensure_ascii=False, indent=2
        ),
        "baseline_c_results.json": json.dumps(
            baseline_rows, ensure_ascii=False, indent=2
        ),
        "semantic_context_c_results.json": json.dumps(
            contextual_result_rows, ensure_ascii=False, indent=2
        ),
        "comparison.txt": _comparison_text(comparison_rows),
        "comparison.json": json.dumps(
            comparison_rows, ensure_ascii=False, indent=2
        ),
        "summary.json": json.dumps(summary, ensure_ascii=False, indent=2),
        "resolver_outputs.json": json.dumps(
            resolver_rows, ensure_ascii=False, indent=2
        ),
    }
    OUTPUT_DIR.mkdir(parents=True)
    for filename, content in serialized.items():
        (OUTPUT_DIR / filename).write_text(content, encoding="utf-8")

    print("=== SEMANTIC CONTEXT V1 COMPLETED ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Human comparison: {OUTPUT_DIR / 'comparison.txt'}")


if __name__ == "__main__":
    run_semantic_context_v1()
