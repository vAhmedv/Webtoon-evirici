"""One-shot TranslateGemma Quality Gate v4 (32 fresh items).

This benchmark performs structural validation only.  Linguistic grades are
assigned manually after the single successful run.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from core.translation.profile_discovery import (
    CandidateStore,
    contains_candidate_phrase,
    get_relevant_terms_for_item,
)
from core.translation.protection import contains_unrestored_protected_term
from core.translation.series_profile import SeriesProfile
from providers.translation import TranslationInput, TranslationItem, get_translation_provider
from providers.translation.translategemma_gguf_translation import (
    TranslateGemmaGGUFTranslationProvider,
)


QUALITY_GATE_V4_ITEMS = [
    (1, "I don't use this ability to attack."),
    (2, "Then what do you use it for?"),
    (3, "It's called Frost Chain."),
    (4, "Frost Chain?"),
    (5, "Activate Frost Chain."),
    (6, "Don't use Frost Chain until I give the signal."),
    (7, "Frost Chain can hold three targets at once."),
    (8, "I learned Frost Chain from my sister."),
    (9, "These Spirit Stones belong to the Guild Master."),
    (10, "Leave the Spirit Stones beside the door."),
    (11, "They want twenty Spirit Stones for one bottle."),
    (12, "A Spirit Stone's value depends on its purity."),
    (13, "PASSIVE SKILL ACQUIRED: ECHO VEIL"),
    (14, "TITLE ACQUIRED: BONE WARDEN"),
    (15, "CLASS ADVANCEMENT AVAILABLE"),
    (16, "ABILITY COOLDOWN: 19 SECONDS"),
    (17, "Leave him out of this."),
    (18, "Leave the key where you found it."),
    (19, "Who charged the meal to my room?"),
    (20, "The boar charged before I could draw my sword."),
    (21, "Not everyone agreed with the Guild Master."),
    (22, "Almost everyone stayed silent."),
    (23, "We need to leave before midnight."),
    (24, "Don't move until the door closes."),
    (25, "Wonderful. Now the bridge is on fire."),
    (26, "You don't sound very impressed."),
    (27, "That's because I'm trying not to scream."),
    (28, "Very reassuring."),
    (29, "The chamber had been empty only a moment earlier."),
    (30, "Now a line of wet footprints crossed the floor."),
    (31, "There was nobody at the other end of the corridor."),
    (32, "Then one of the footprints moved."),
]


def run_quality_gate_v4() -> None:
    output_dir = Path("benchmark_results/translategemma_quality_gate_v4")
    if output_dir.exists():
        raise RuntimeError(f"Fresh-run guard: output directory already exists: {output_dir}")

    profile = SeriesProfile(
        series_id="quality_gate_v4",
        glossary={
            "SPIRIT STONE": "Ruh Taşı",
            "GUILD MASTER": "Lonca Lideri",
        },
    )
    store = CandidateStore(series_id="quality_gate_v4")
    items = [
        TranslationItem(region_id=region_id, source=source, reading_order=region_id)
        for region_id, source in QUALITY_GATE_V4_ITEMS
    ]
    inp = TranslationInput(
        items=items,
        profile=profile,
        candidate_store=store,
        chapter_id="quality_gate_v4_ch1",
    )

    provider = get_translation_provider()
    if not isinstance(provider, TranslateGemmaGGUFTranslationProvider):
        raise RuntimeError(f"Quality Gate v4 requires TranslateGemma, got: {provider.name}")
    load_started = time.perf_counter()
    provider.load()
    load_time = time.perf_counter() - load_started

    try:
        wall_started = time.perf_counter()
        out = provider.translate(inp)
        wall_time = time.perf_counter() - wall_started
    finally:
        provider.unload()

    metrics = provider.metrics
    returned_ids = [result.region_id for result in out.results]
    expected_ids = [region_id for region_id, _ in QUALITY_GATE_V4_ITEMS]
    missing_ids = sorted(set(expected_ids) - set(returned_ids))
    duplicate_ids = sorted({region_id for region_id in returned_ids if returned_ids.count(region_id) > 1})

    results_data = []
    for result in out.results:
        approved_terms, _ = get_relevant_terms_for_item(result.source, profile, store)
        results_data.append(
            {
                "id": result.region_id,
                "source": result.source,
                "translation": result.translation,
                "approved_terms": approved_terms,
                "validation_warnings": result.validation_warnings,
                "requires_review": result.requires_review,
                "raw_model_response": result.raw_model_response,
            }
        )

    chatbot_ids = [
        row["id"]
        for row in results_data
        if "chatbot_or_explanation_output" in row["validation_warnings"]
    ]
    sentinel_leak_ids = [
        row["id"]
        for row in results_data
        if row["translation"] and contains_unrestored_protected_term(row["translation"])
    ]
    server_error_ids = [
        row["id"]
        for row in results_data
        if "translation_server_error" in row["validation_warnings"]
    ]
    approved_english_leak_ids = []
    for row in results_data:
        translation = row["translation"] or ""
        for source_term, target_term in row["approved_terms"].items():
            if (
                source_term.casefold() != target_term.casefold()
                and contains_candidate_phrase(source_term, translation)
            ):
                approved_english_leak_ids.append(row["id"])
                break

    expected_initial_generation_calls = (
        len(QUALITY_GATE_V4_ITEMS)
        - metrics.system_ui_bypass_count
        - metrics.term_only_bypass_count
    )
    truthful_generation_calls = (
        metrics.generation_call_count
        == expected_initial_generation_calls + metrics.retries
    )

    structural_validation = {
        "result_object_count": len(out.results),
        "expected_result_object_count": len(QUALITY_GATE_V4_ITEMS),
        "missing_ids": missing_ids,
        "duplicate_ids": duplicate_ids,
        "chatbot_or_explanation_ids": chatbot_ids,
        "sentinel_leak_ids": sentinel_leak_ids,
        "server_error_ids": server_error_ids,
        "approved_glossary_english_leak_ids": approved_english_leak_ids,
        "system_ui_bypass_count": metrics.system_ui_bypass_count,
        "term_only_bypass_count": metrics.term_only_bypass_count,
        "generation_calls_truthful": truthful_generation_calls,
        "retries_truthful": truthful_generation_calls,
        "requires_review_ids": [row["id"] for row in results_data if row["requires_review"]],
    }
    structural_validation["clean"] = all(
        [
            len(out.results) == len(QUALITY_GATE_V4_ITEMS),
            not missing_ids,
            not duplicate_ids,
            not chatbot_ids,
            not sentinel_leak_ids,
            not server_error_ids,
            not approved_english_leak_ids,
            metrics.system_ui_bypass_count == 4,
            metrics.term_only_bypass_count == 1,
            truthful_generation_calls,
        ]
    )

    summary_data = {
        "execution": {
            "items": len(out.results),
            "generation_calls": metrics.generation_call_count,
            "system_ui_bypass_count": metrics.system_ui_bypass_count,
            "term_only_bypass_count": metrics.term_only_bypass_count,
            "retries": metrics.retries,
            "requires_review": sum(1 for row in results_data if row["requires_review"]),
            "input_tokens": metrics.input_token_count,
            "generated_tokens": metrics.generated_token_count,
            "generation_seconds": round(metrics.generation_seconds, 4),
            "wall_time_seconds": round(wall_time, 4),
            "tokens_per_second": metrics.tokens_per_sec,
            "load_time_seconds": round(load_time, 4),
            "backend": provider.name,
            "model": metrics.translation_model,
        },
        "structural_validation": structural_validation,
    }

    output_dir.mkdir(parents=True)
    (output_dir / "results.json").write_text(
        json.dumps(results_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    text_lines = []
    for row in results_data:
        text_lines.extend(
            [
                f"[{row['id']:03d}]",
                f"EN: {row['source']}",
                f"TR: {row['translation'] or '<FAILED>'}",
                f"WARNINGS: {', '.join(row['validation_warnings']) or '-'}",
                "",
            ]
        )
    (output_dir / "results.txt").write_text("\n".join(text_lines), encoding="utf-8")

    print("=== TRANSLATEGEMMA QUALITY GATE V4 COMPLETED ===")
    print(json.dumps(summary_data, ensure_ascii=False, indent=2))
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    run_quality_gate_v4()
