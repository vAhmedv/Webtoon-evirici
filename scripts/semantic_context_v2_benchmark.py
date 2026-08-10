"""One-shot Semantic Context V2 translation-risk benchmark."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from benchmark.semantic_context_v1_dataset import (
    ContextualTestItem,
    build_semantic_context_v1_dataset,
)
from core.translation.profile_discovery import CandidateStore
from core.translation.protection import contains_unrestored_protected_term
from core.translation.semantic_context import (
    DEFAULT_SEMANTIC_RESOLVER_MIN_CONFIDENCE,
    SemanticContextRequest,
    TranslationRiskOutcome,
    render_translation_risk_resolver_prompt,
    resolve_translation_risk_with_fallback,
    validate_clarified_target,
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


OUTPUT_DIR = Path("benchmark_results/semantic_context_v2")
V1_DIR = Path("benchmark_results/semantic_context_v1")
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
SPECIAL_REVIEW_IDS = (8, 11, 12, 13, 14, 19, 23)


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
        series_id="semantic_context_v2",
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
        candidate_store=CandidateStore(series_id="semantic_context_v2_context_c"),
        chapter_id="semantic_context_v2_context_c",
    )


def _bypass_type(raw_model_response: str | None) -> str | None:
    if raw_model_response == "[SYSTEM_UI_BYPASS]":
        return "system_ui"
    if raw_model_response == "[TERM_ONLY_BYPASS]":
        return "term_only"
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
    return rows, {
        "selected_items": len(rows),
        "experimental_model_calls": metrics.generation_call_count,
        "system_ui_bypass": metrics.system_ui_bypass_count,
        "term_only_bypass": metrics.term_only_bypass_count,
        "selected_requires_review": sum(row["requires_review"] for row in rows),
        "selected_guard_counts": dict(sorted(warning_counts.items())),
        "generation_seconds": round(metrics.generation_seconds, 4),
        "wall_time": round(wall_time, 4),
        "input_tokens": metrics.input_token_count,
        "generated_tokens": metrics.generated_token_count,
        "tokens_per_second": metrics.tokens_per_sec,
        "retries": metrics.retries,
        "micro_batch_requests": metrics.micro_batch_requests,
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _duplicates(rows: list[dict[str, Any]]) -> list[int]:
    counts = Counter(int(row["id"]) for row in rows)
    return sorted(item_id for item_id, count in counts.items() if count > 1)


def _preflight_v1_alignment() -> tuple[
    bytes,
    list[ContextualTestItem],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    required = (
        "dataset.json",
        "baseline_c_results.json",
        "comparison.json",
        "resolver_outputs.json",
    )
    missing = [name for name in required if not (V1_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing required V1 artifacts: {missing}")

    dataset_bytes = (V1_DIR / "dataset.json").read_bytes()
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            "Frozen V1 dataset hash mismatch: "
            f"expected {EXPECTED_DATASET_SHA256}, got {dataset_hash}"
        )
    artifact_dataset = json.loads(dataset_bytes.decode("utf-8"))
    dataset = build_semantic_context_v1_dataset()
    code_dataset = [item.to_dict() for item in dataset]
    if artifact_dataset != code_dataset:
        raise RuntimeError("Frozen V1 dataset artifact no longer matches dataset code")

    expected_ids = [item.id for item in dataset]
    baseline_rows = _load_json(V1_DIR / "baseline_c_results.json")
    comparison_rows = _load_json(V1_DIR / "comparison.json")
    resolver_rows = _load_json(V1_DIR / "resolver_outputs.json")
    for label, rows in (
        ("baseline", baseline_rows),
        ("comparison", comparison_rows),
        ("resolver", resolver_rows),
    ):
        ids = [int(row["id"]) for row in rows]
        if ids != expected_ids or len(set(ids)) != len(expected_ids):
            raise RuntimeError(f"V1 {label} IDs are not aligned with the frozen dataset")

    by_id_dataset = {item.id: item for item in dataset}
    for row in baseline_rows:
        if row["source"] != by_id_dataset[row["id"]].target_source:
            raise RuntimeError(f"V1 baseline source mismatch at ID {row['id']}")
    for row in comparison_rows:
        item = by_id_dataset[row["id"]]
        if (
            row["target_source"] != item.target_source
            or tuple(row["previous_context"]) != item.previous_context
            or tuple(row["next_context"]) != item.next_context
        ):
            raise RuntimeError(f"V1 comparison context mismatch at ID {row['id']}")
    for row in resolver_rows:
        if row["target_source"] != by_id_dataset[row["id"]].target_source:
            raise RuntimeError(f"V1 resolver source mismatch at ID {row['id']}")

    return dataset_bytes, dataset, baseline_rows, comparison_rows, resolver_rows


def _resolution_view(outcome: TranslationRiskOutcome) -> dict[str, Any]:
    resolution = outcome.resolution
    return {
        "raw_response": outcome.raw_response,
        "rewrite_needed": resolution.rewrite_needed if resolution else None,
        "confidence": resolution.confidence if resolution else None,
        "risk_types": list(resolution.risk_types) if resolution else [],
        "semantic_notes": (
            [asdict(note) for note in resolution.semantic_notes] if resolution else []
        ),
        "question_type": resolution.question_type if resolution else None,
        "tense_aspect": resolution.tense_aspect if resolution else None,
        "referents": list(resolution.referents) if resolution else [],
        "clarified_target": resolution.clarified_target if resolution else None,
        "selected_target": outcome.decision.selected_target,
        "validation_passed": bool(
            resolution is not None and not outcome.decision.validation_failures
        ),
        "validation_reasons": list(outcome.decision.validation_failures),
        "rewrite_used": outcome.decision.rewrite_used,
        "rejection_reason": outcome.decision.rejection_reason,
        "resolver_failed": outcome.resolver_failed,
        "malformed_json": outcome.malformed_json,
    }


def _validation_metrics(
    dataset: list[ContextualTestItem],
    outcomes: dict[int, TranslationRiskOutcome],
) -> tuple[dict[str, int], dict[str, list[int]], list[int]]:
    category_by_reason = {
        "named_term_loss": "named_term_rejection",
        "number_changed": "number_rejection",
        "polarity_changed": "polarity_rejection",
        "question_type_changed": "question_type_rejection",
        "terminal_punctuation_changed": "question_type_rejection",
        "declared_question_type_mismatch": "question_type_rejection",
        "tense_aspect_changed": "tense_aspect_rejection",
        "clarified_target_not_english": "non_english_rejection",
        "clarified_target_too_long": "excessive_length_rejection",
        "context_sentence_copied": "context_copy_rejection",
    }
    categories = [
        "named_term_rejection",
        "number_rejection",
        "polarity_rejection",
        "question_type_rejection",
        "tense_aspect_rejection",
        "non_english_rejection",
        "excessive_length_rejection",
        "context_copy_rejection",
        "other_rejection",
    ]
    ids_by_category: dict[str, list[int]] = {key: [] for key in categories}
    unsafe_selected_ids: list[int] = []
    by_id = {item.id: item for item in dataset}
    for item_id, outcome in outcomes.items():
        for reason in outcome.decision.validation_failures:
            category = category_by_reason.get(reason, "other_rejection")
            ids_by_category[category].append(item_id)
        item = by_id[item_id]
        if validate_clarified_target(
            item.target_source,
            outcome.decision.selected_target,
            named_terms=item.named_terms,
            context_sources=(*item.previous_context, *item.next_context),
        ):
            unsafe_selected_ids.append(item_id)
    ids_by_category = {
        key: sorted(set(value)) for key, value in ids_by_category.items()
    }
    return (
        {key: len(ids_by_category[key]) for key in categories},
        ids_by_category,
        sorted(unsafe_selected_ids),
    )


def _comparison_text(rows: list[dict[str, Any]]) -> str:
    separator = "=" * 60
    lines = [
        "SEMANTIC CONTEXT V2 — V1 VS V2 HUMAN QUALITY REVIEW",
        "No automatic quality winner is declared.",
        "",
    ]
    for row in rows:
        v1_notes = row["v1"]["semantic_notes"]
        v2_notes = row["v2_resolver"]["semantic_notes"]
        lines.extend(
            [
                separator,
                f"[{row['id']:03d}]",
                "",
                "PREVIOUS:",
                "\n".join(row["previous_context"]) or "<NONE>",
                "",
                "TARGET:",
                row["target_source"],
                "",
                "NEXT:",
                "\n".join(row["next_context"]) or "<NONE>",
                "",
                "V1 RESOLVER:",
                f"rewrite used: {'yes' if row['v1']['rewrite_used'] else 'no'}",
                "semantic notes:",
                *(
                    [
                        f"- {note['span']} -> {note['intended_sense']}"
                        for note in v1_notes
                    ]
                    or ["<NONE>"]
                ),
                "",
                "V2 RESOLVER:",
                f"rewrite_needed: {row['v2_resolver']['rewrite_needed']}",
                f"confidence: {row['v2_resolver']['confidence']}",
                "risk_types:",
                *(
                    [f"- {value}" for value in row["v2_resolver"]["risk_types"]]
                    or ["<NONE>"]
                ),
                "semantic notes:",
                *(
                    [
                        f"- {note['span']} -> {note['resolved_meaning']}"
                        for note in v2_notes
                    ]
                    or ["<NONE>"]
                ),
                "",
                "CLARIFIED TARGET:",
                row["v2_resolver"]["clarified_target"] or "<NONE>",
                "",
                "SELECTED TARGET:",
                row["selected_target"],
                "",
                "VALIDATION:",
                "PASS" if row["v2_resolver"]["validation_passed"] else "FALLBACK",
                *(
                    [f"- {reason}" for reason in row["v2_resolver"]["validation_reasons"]]
                    or ["- no deterministic validation failures"]
                ),
                "",
                "V1 / BASELINE TRANSLATION:",
                row["v1"]["translation"] or "<FAILED>",
                "",
                "V2 CONTEXT TRANSLATION:",
                row["v2_translation"] or "<FAILED>",
                "",
                "WARNINGS:",
                *(row["warnings"] or ["<NONE>"]),
                separator,
                "",
            ]
        )
    return "\n".join(lines)


def _default_translation_provider() -> _CapturingTranslateGemmaProvider:
    return _CapturingTranslateGemmaProvider(
        managed=True,
        micro_batch_enabled=False,
        prompt_variant="minimal_faithful",
    )


def run_semantic_context_v2(
    *,
    output_dir: Path = OUTPUT_DIR,
    qwen_provider: Any | None = None,
    translation_provider_factory: Callable[
        [], _CapturingTranslateGemmaProvider
    ] = _default_translation_provider,
) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError(f"Fresh-run guard: output directory already exists: {output_dir}")

    (
        dataset_bytes,
        dataset,
        baseline_rows,
        v1_comparison_rows,
        v1_resolver_rows,
    ) = _preflight_v1_alignment()
    expected_ids = [item.id for item in dataset]
    by_id_item = {item.id: item for item in dataset}
    by_id_baseline = {row["id"]: row for row in baseline_rows}
    by_id_v1_comparison = {row["id"]: row for row in v1_comparison_rows}
    by_id_v1_resolver = {row["id"]: row for row in v1_resolver_rows}

    qwen = qwen_provider or QwenSemanticResolverProvider(
        prompt_renderer=render_translation_risk_resolver_prompt,
    )
    if qwen_provider is None and qwen.model_path != EXPECTED_QWEN_MODEL_PATH:
        raise RuntimeError(f"Unexpected Qwen model path: {qwen.model_path}")
    if (
        qwen_provider is None
        and qwen.executable_path != EXPECTED_QWEN_LLAMA_EXE_PATH
    ):
        raise RuntimeError(
            f"Unexpected Qwen llama.cpp executable: {qwen.executable_path}"
        )

    outcomes: dict[int, TranslationRiskOutcome] = {}
    qwen.load()
    try:
        for index, item in enumerate(dataset, start=1):
            request = SemanticContextRequest(
                previous_context=item.previous_context,
                target_source=item.target_source,
                next_context=item.next_context,
                named_terms=item.named_terms,
            )
            outcomes[item.id] = resolve_translation_risk_with_fallback(
                request,
                qwen.resolve,
                min_confidence=MIN_CONFIDENCE,
            )
            print(
                f"Qwen V2 resolved {index}/{len(dataset)} (ID {item.id:03d})",
                flush=True,
            )
    finally:
        qwen.unload()

    selected_targets = {
        item_id: outcome.decision.selected_target
        for item_id, outcome in outcomes.items()
    }
    accepted_items = [
        item for item in dataset if outcomes[item.id].decision.rewrite_used
    ]

    translated_selected_rows: list[dict[str, Any]] = []
    selected_translation_summary: dict[str, Any] = {
        "selected_items": 0,
        "experimental_model_calls": 0,
        "system_ui_bypass": 0,
        "term_only_bypass": 0,
        "selected_requires_review": 0,
        "selected_guard_counts": {},
        "generation_seconds": 0.0,
        "wall_time": 0.0,
        "input_tokens": 0,
        "generated_tokens": 0,
        "tokens_per_second": 0.0,
        "retries": 0,
        "micro_batch_requests": 0,
    }
    if accepted_items:
        print(
            f"TranslateGemma V2 selected targets: {len(accepted_items)}",
            flush=True,
        )
        translation_provider = translation_provider_factory()
        if (
            translation_provider_factory is _default_translation_provider
            and translation_provider.model_path != EXPECTED_TRANSLATEGEMMA_MODEL_PATH
        ):
            raise RuntimeError(
                f"Unexpected TranslateGemma model path: {translation_provider.model_path}"
            )
        translation_input = _make_translation_input(accepted_items, selected_targets)
        translation_provider.load()
        try:
            translation_started = time.perf_counter()
            translation_output = translation_provider.translate(translation_input)
            translation_wall = time.perf_counter() - translation_started
        finally:
            translation_provider.unload()
        translated_selected_rows, selected_translation_summary = _collect_translation(
            translation_provider,
            translation_output,
            translation_wall,
        )
        if translation_provider.metrics.micro_batch_requests:
            raise RuntimeError("Semantic Context V2 issued a forbidden micro-batch request")

    by_id_selected_translation = {
        row["id"]: row for row in translated_selected_rows
    }
    missing_selected = sorted(
        {item.id for item in accepted_items} - set(by_id_selected_translation)
    )
    if missing_selected:
        raise RuntimeError(
            f"Missing TranslateGemma results for accepted rewrites: {missing_selected}"
        )

    context_rows: list[dict[str, Any]] = []
    for item in dataset:
        rewrite_used = outcomes[item.id].decision.rewrite_used
        source_row = (
            by_id_selected_translation[item.id]
            if rewrite_used
            else by_id_baseline[item.id]
        )
        context_rows.append(
            {
                **source_row,
                "source": selected_targets[item.id],
                "original_target_source": item.target_source,
                "selected_target": selected_targets[item.id],
                "rewrite_used": rewrite_used,
                "reused_v1_baseline": not rewrite_used,
                "rejection_reason": outcomes[item.id].decision.rejection_reason,
            }
        )

    resolver_rows = [
        {
            "id": item.id,
            "context_source": "SYNTHETIC" if item.synthetic_context else "REAL",
            "synthetic_context": item.synthetic_context,
            "source_origin": item.source_origin,
            "previous_context": list(item.previous_context),
            "target_source": item.target_source,
            "next_context": list(item.next_context),
            "named_terms": list(item.named_terms),
            **_resolution_view(outcomes[item.id]),
        }
        for item in dataset
    ]
    by_id_resolver = {row["id"]: row for row in resolver_rows}
    by_id_context = {row["id"]: row for row in context_rows}

    comparison_rows: list[dict[str, Any]] = []
    for item in dataset:
        v1_comparison = by_id_v1_comparison[item.id]
        v1_resolver = by_id_v1_resolver[item.id]
        resolver = by_id_resolver[item.id]
        context = by_id_context[item.id]
        comparison_rows.append(
            {
                "id": item.id,
                "context_source": "SYNTHETIC" if item.synthetic_context else "REAL",
                "source_origin": item.source_origin,
                "previous_context": list(item.previous_context),
                "target_source": item.target_source,
                "next_context": list(item.next_context),
                "v1": {
                    "rewrite_used": bool(v1_resolver["clarification_used"]),
                    "semantic_notes": v1_resolver["semantic_notes"],
                    "malformed_json": bool(v1_resolver["malformed_json"]),
                    "selected_target": v1_resolver["selected_target"],
                    "translation": v1_comparison["contextual"]["translation"],
                    "warnings": v1_comparison["contextual"]["warnings"],
                },
                "v2_resolver": {
                    key: resolver[key]
                    for key in (
                        "rewrite_needed",
                        "confidence",
                        "risk_types",
                        "semantic_notes",
                        "question_type",
                        "tense_aspect",
                        "referents",
                        "clarified_target",
                        "validation_passed",
                        "validation_reasons",
                        "rewrite_used",
                        "rejection_reason",
                        "resolver_failed",
                        "malformed_json",
                    )
                },
                "selected_target": selected_targets[item.id],
                "v2_translation": context["translation"],
                "warnings": context["warnings"],
                "requires_review": context["requires_review"],
                "reused_v1_baseline": context["reused_v1_baseline"],
            }
        )

    validation_counts, validation_ids, unsafe_selected_ids = _validation_metrics(
        dataset,
        outcomes,
    )
    resolution_values = [
        outcome.resolution for outcome in outcomes.values() if outcome.resolution
    ]
    rejection_counts = Counter(
        outcome.decision.rejection_reason
        for outcome in outcomes.values()
        if outcome.decision.rejection_reason
    )
    accepted_rewrites = [
        {
            "id": item.id,
            "original_target": item.target_source,
            "clarified_target": selected_targets[item.id],
            "confidence": outcomes[item.id].resolution.confidence,
            "risk_types": list(outcomes[item.id].resolution.risk_types),
        }
        for item in accepted_items
        if outcomes[item.id].resolution is not None
    ]
    rejected_rewrites = [
        {
            "id": item.id,
            "original_target": item.target_source,
            "proposed_target": outcomes[item.id].resolution.clarified_target,
            "confidence": outcomes[item.id].resolution.confidence,
            "rejection_reason": outcomes[item.id].decision.rejection_reason,
            "validation_reasons": list(
                outcomes[item.id].decision.validation_failures
            ),
        }
        for item in dataset
        if outcomes[item.id].resolution is not None
        and outcomes[item.id].resolution.rewrite_needed
        and not outcomes[item.id].decision.rewrite_used
    ]

    warning_counts = Counter(
        warning for row in context_rows for warning in row["warnings"]
    )
    qwen_summary = {
        "model": qwen.name,
        "model_path": getattr(qwen, "model_path", DEFAULT_QWEN_SEMANTIC_MODEL_PATH),
        "resolver_backend": getattr(
            qwen,
            "backend",
            "qwen3.5-9b-gguf-llamacpp",
        ),
        "quantization": getattr(qwen, "quantization", "Q5_K_M"),
        "llama_executable": getattr(
            qwen,
            "executable_path",
            DEFAULT_QWEN_SEMANTIC_LLAMA_EXE_PATH,
        ),
        "server_url": getattr(
            qwen,
            "server_url",
            DEFAULT_QWEN_SEMANTIC_SERVER_URL,
        ),
        "gpu_layers": getattr(
            qwen,
            "gpu_layers",
            DEFAULT_QWEN_SEMANTIC_GPU_LAYERS,
        ),
        "context_size": getattr(
            qwen,
            "max_context_length",
            DEFAULT_QWEN_SEMANTIC_CONTEXT_SIZE,
        ),
        "temperature": getattr(
            qwen,
            "temperature",
            DEFAULT_QWEN_SEMANTIC_TEMPERATURE,
        ),
        "reasoning": getattr(qwen, "reasoning_mode", "off"),
        "chat_template_strategy": getattr(
            qwen,
            "chat_template_strategy",
            "embedded GGUF template via /v1/chat/completions",
        ),
        "chat_template_sha256": getattr(qwen, "chat_template_sha256", None),
        "chat_template_preview": getattr(qwen, "chat_template_preview", None),
        "chat_template_caps": getattr(qwen, "last_props", {}).get(
            "chat_template_caps",
            {},
        ),
        "llama_server_command": list(
            getattr(qwen, "last_server_command", ())
        ),
        "port_closed_after_unload": getattr(
            qwen,
            "port_closed_after_unload",
            None,
        ),
        "calls": qwen.metrics.resolver_calls,
        "failures": sum(outcome.resolver_failed for outcome in outcomes.values()),
        "provider_failures": qwen.metrics.resolver_failures,
        "reasoning_contamination": getattr(
            qwen.metrics,
            "reasoning_contamination_count",
            0,
        ),
        "malformed_json": sum(outcome.malformed_json for outcome in outcomes.values()),
        "rewrite_needed_true": sum(
            value.rewrite_needed for value in resolution_values
        ),
        "rewrite_needed_false": sum(
            not value.rewrite_needed for value in resolution_values
        ),
        "high_confidence_rewrites": sum(
            value.rewrite_needed and value.confidence >= MIN_CONFIDENCE
            for value in resolution_values
        ),
        "accepted_rewrites": len(accepted_rewrites),
        "rejected_rewrites": len(rejected_rewrites),
        "low_confidence_rewrites": rejection_counts["low_confidence"],
        "unchanged_targets": sum(
            outcome.decision.selected_target == outcome.request.target_source
            for outcome in outcomes.values()
        ),
        "fallback_reason_counts": dict(sorted(rejection_counts.items())),
        "average_resolver_seconds": round(qwen.metrics.average_resolver_seconds, 4),
        "generation_seconds": round(qwen.metrics.generation_seconds, 4),
        "server_generation_seconds": round(
            getattr(qwen.metrics, "server_generation_seconds", 0.0),
            4,
        ),
        "input_tokens": qwen.metrics.input_token_count,
        "generated_tokens": qwen.metrics.generated_token_count,
        "tokens_per_second": round(
            getattr(qwen.metrics, "tokens_per_second", 0.0),
            4,
        ),
        "model_load_seconds": qwen.metrics.model_load_seconds,
        "model_load_vram_gb": qwen.metrics.model_load_vram_gb,
        "peak_vram_gb": round(qwen.metrics.peak_vram_gb, 4),
        "transformers_v1_historical_average_seconds": 57.436,
    }
    translation_summary = {
        **selected_translation_summary,
        "items": len(context_rows),
        "reused_v1_baseline_count": sum(
            row["reused_v1_baseline"] for row in context_rows
        ),
        "requires_review": sum(row["requires_review"] for row in context_rows),
        "guard_counts": dict(sorted(warning_counts.items())),
    }

    def leak_ids(predicate: Callable[[str], bool]) -> list[int]:
        return [
            row["id"]
            for row in context_rows
            if row["translation"] and predicate(row["translation"])
        ]

    target_mismatch_ids = [
        item.id
        for item in dataset
        if by_id_context[item.id]["source"] != selected_targets[item.id]
        or by_id_context[item.id]["original_target_source"] != item.target_source
    ]
    context_mismatch_ids = [
        item.id
        for item in dataset
        if tuple(by_id_v1_comparison[item.id]["previous_context"])
        != item.previous_context
        or tuple(by_id_v1_comparison[item.id]["next_context"])
        != item.next_context
    ]
    structural = {
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "dataset_hash_match": True,
        "missing_ids": sorted(set(expected_ids) - set(by_id_context)),
        "duplicate_ids": _duplicates(context_rows),
        "target_mismatch_ids": target_mismatch_ids,
        "context_mismatch_ids": context_mismatch_ids,
        "unsafe_rewrite_selected_ids": unsafe_selected_ids,
        "sentinel_leak_ids": leak_ids(contains_unrestored_protected_term),
        "segment_marker_leak_ids": leak_ids(contains_segment_marker),
        "server_error_ids": [
            row["id"]
            for row in context_rows
            if "translation_server_error" in row["warnings"]
        ],
        "micro_batch_requests": translation_summary["micro_batch_requests"],
    }
    structural["final_path_clean"] = not any(
        value
        for key, value in structural.items()
        if key not in {"dataset_sha256", "dataset_hash_match", "final_path_clean"}
    ) and structural["dataset_hash_match"]
    if unsafe_selected_ids:
        raise RuntimeError(
            "Unsafe selected targets reached TranslateGemma boundary: "
            f"{unsafe_selected_ids}"
        )
    if not structural["final_path_clean"]:
        raise RuntimeError(f"Semantic Context V2 structural guard failed: {structural}")

    summary = {
        "dataset": {
            "items": len(dataset),
            "real_items": sum(not item.synthetic_context for item in dataset),
            "synthetic_items": sum(item.synthetic_context for item in dataset),
            "source": "exact frozen Semantic Context V1 dataset",
            "sha256": structural["dataset_sha256"],
            "code_matches_artifact": True,
        },
        "semantic_resolver_min_confidence": MIN_CONFIDENCE,
        "model_sequence": [
            "validate_and_reuse_v1_dataset_and_baseline",
            "load_qwen_v2_gguf_llamacpp_resolver",
            "resolve_all_24_targets_once",
            "unload_qwen_v2_resolver",
            "load_translategemma_only_if_rewrites_are_accepted",
            "translate_accepted_targets_single_item_only",
            "unload_translategemma",
            "reuse_v1_baseline_for_byte_identical_original_targets",
        ],
        "qwen_v2": qwen_summary,
        "rewrite_validation": validation_counts,
        "rewrite_validation_ids": validation_ids,
        "accepted_rewrites": accepted_rewrites,
        "rejected_rewrites": rejected_rewrites,
        "translategemma_context_c": translation_summary,
        "structural": structural,
        "special_human_review_ids": list(SPECIAL_REVIEW_IDS),
        "quality_winner": None,
        "quality_winner_note": (
            "Manual human review required; V2 is not automatically ranked over V1."
        ),
        "production_default": "legacy",
        "semantic_context_v2_status": "experimental",
    }
    selected_target_rows = [
        {
            "id": item.id,
            "original_target": item.target_source,
            "selected_target": selected_targets[item.id],
            "rewrite_used": outcomes[item.id].decision.rewrite_used,
            "reused_v1_baseline": not outcomes[item.id].decision.rewrite_used,
            "rejection_reason": outcomes[item.id].decision.rejection_reason,
        }
        for item in dataset
    ]
    serialized: dict[str, bytes] = {
        "dataset.json": dataset_bytes,
        "resolver_outputs.json": json.dumps(
            resolver_rows,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        "selected_targets.json": json.dumps(
            selected_target_rows,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        "context_c_results.json": json.dumps(
            context_rows,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        "comparison_v1_v2.txt": _comparison_text(comparison_rows).encode("utf-8"),
        "comparison_v1_v2.json": json.dumps(
            comparison_rows,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        "summary.json": json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
    }
    output_dir.mkdir(parents=True)
    for filename, content in serialized.items():
        (output_dir / filename).write_bytes(content)

    print("=== SEMANTIC CONTEXT V2 COMPLETED ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Human comparison: {output_dir / 'comparison_v1_v2.txt'}")
    return summary


if __name__ == "__main__":
    if DEFAULT_GEMMA_MODEL_PATH != EXPECTED_TRANSLATEGEMMA_MODEL_PATH:
        raise RuntimeError(
            f"Unexpected TranslateGemma default model path: {DEFAULT_GEMMA_MODEL_PATH}"
        )
    run_semantic_context_v2()
