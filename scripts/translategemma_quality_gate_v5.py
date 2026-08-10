"""One-shot TranslateGemma Quality Gate v5 (40 unseen contextual items).

The script performs production translation and deterministic structural checks.
Linguistic grades are assigned manually by the coding agent after this single
successful run; no model-based judge is used.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from core.translation.profile_discovery import (
    CandidateStore,
    ProfileCandidate,
    get_relevant_terms_for_item,
)
from core.translation.protection import (
    contains_unrestored_protected_term,
    protect_source_text,
    validate_protected_terms,
)
from core.translation.series_profile import SeriesProfile
from providers.translation import TranslationInput, TranslationItem, get_translation_provider
from providers.translation.translategemma_gguf_translation import (
    TranslateGemmaGGUFTranslationProvider,
    contains_segment_marker,
    is_explanation_like_output,
)


QUALITY_GATE_V5_ITEMS = [
    (1, "I told you not to touch that."),
    (2, "You told me not to open it."),
    (3, "That's not the same thing."),
    (4, "It is when the box explodes."),
    (5, "It's called Silent Chain."),
    (6, "Silent Chain?"),
    (7, "Activate Silent Chain."),
    (8, "It can hold two enemies at the same time."),
    (9, "Don't use Silent Chain again until I give the signal."),
    (10, "I learned it from my father."),
    (11, "He never taught you when to stop?"),
    (12, "Apparently not."),
    (13, "These Spirit Stones are for the Guild Master."),
    (14, "Put the Spirit Stones on his desk."),
    (15, "They paid thirty Spirit Stones for this map."),
    (16, "One Spirit Stone's value can change with its purity."),
    (17, "PASSIVE SKILL ACQUIRED: NIGHT SENSE"),
    (18, "TITLE ACQUIRED: WHITE WARDEN"),
    (19, "CLASS ADVANCEMENT AVAILABLE"),
    (20, "ABILITY COOLDOWN: 21 SECONDS"),
    (21, "Who charged the medicine to my room?"),
    (22, "The healer did."),
    (23, "The wolf charged before I could stand up."),
    (24, "That's a different kind of charge."),
    (25, "Leave him out of this."),
    (26, "Leave the documents on the table."),
    (27, "We need to leave before sunrise."),
    (28, "Leave the rest to me."),
    (29, "Not everyone believed the Guild Master."),
    (30, "Almost everyone stayed quiet."),
    (31, "Only one person objected."),
    (32, "That doesn't mean the others agreed."),
    (33, "Wonderful. Now the door is on fire."),
    (34, "You sound surprisingly calm."),
    (35, "I'm deciding whether to scream or run."),
    (36, "You can do both."),
    (37, "The chamber had been empty a few seconds earlier."),
    (38, "Now three wet footprints crossed the stone floor."),
    (39, "There was nobody near the far wall."),
    (40, "Then one of the footprints turned toward us."),
]

# These are real upstream scene boundaries. Within each call, the production
# provider alone decides which adjacent ordinary items form a micro-batch.
SCENE_CALL_REGION_IDS = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
    [21, 22, 23, 24],
    [25, 26, 27, 28],
    [29, 30, 31, 32],
    [33, 34, 35, 36],
    [37, 38, 39, 40],
]


def _make_profile_and_candidates() -> tuple[SeriesProfile, CandidateStore]:
    profile = SeriesProfile(
        series_id="quality_gate_v5",
        glossary={
            "SPIRIT STONE": "Ruh Taşı",
            "GUILD MASTER": "Lonca Lideri",
        },
    )
    candidates = {
        term.upper(): ProfileCandidate(source=term, kind="named_term", status="discovered")
        for term in ("Silent Chain", "Night Sense", "White Warden")
    }
    return profile, CandidateStore(series_id="quality_gate_v5", candidates=candidates)


def run_quality_gate_v5() -> None:
    output_dir = Path("benchmark_results/translategemma_quality_gate_v5")
    if output_dir.exists():
        raise RuntimeError(f"Fresh-run guard: output directory already exists: {output_dir}")

    profile, store = _make_profile_and_candidates()
    item_by_id = {
        region_id: TranslationItem(
            region_id=region_id,
            source=source,
            reading_order=region_id,
        )
        for region_id, source in QUALITY_GATE_V5_ITEMS
    }

    provider = get_translation_provider()
    if not isinstance(provider, TranslateGemmaGGUFTranslationProvider):
        raise RuntimeError(f"Quality Gate v5 requires TranslateGemma, got: {provider.name}")

    load_started = time.perf_counter()
    provider.load()
    load_seconds = time.perf_counter() - load_started

    all_results = []
    all_raw_responses: list[str] = []
    wall_started = time.perf_counter()
    try:
        for scene_index, region_ids in enumerate(SCENE_CALL_REGION_IDS, start=1):
            scene_input = TranslationInput(
                items=[item_by_id[region_id] for region_id in region_ids],
                profile=profile,
                candidate_store=store,
                chapter_id=f"quality_gate_v5_scene_{scene_index:02d}",
            )
            scene_output = provider.translate(scene_input)
            all_results.extend(scene_output.results)
            all_raw_responses.append(scene_output.raw_response)
    finally:
        wall_seconds = time.perf_counter() - wall_started
        provider.unload()

    metrics = provider.metrics
    expected_ids = [region_id for region_id, _ in QUALITY_GATE_V5_ITEMS]
    returned_ids = [result.region_id for result in all_results]
    missing_ids = sorted(set(expected_ids) - set(returned_ids))
    duplicate_ids = sorted(
        {region_id for region_id in returned_ids if returned_ids.count(region_id) > 1}
    )

    result_rows = []
    for result in all_results:
        approved_terms, _ = get_relevant_terms_for_item(result.source, profile, store)
        bypass_type = None
        if result.raw_model_response == "[System UI Lexicon]":
            bypass_type = "SYSTEM_UI"
        elif result.raw_model_response == "[Term-Only Bypass]":
            bypass_type = "TERM_ONLY"
        result_rows.append(
            {
                "id": result.region_id,
                "source": result.source,
                "translation": result.translation,
                "approved_terms": approved_terms,
                "validation_warnings": list(result.validation_warnings),
                "requires_review": result.requires_review,
                "raw_model_response": result.raw_model_response,
                "micro_batch_id": result.micro_batch_id,
                "micro_batch_region_ids": list(result.micro_batch_region_ids),
                "bypass_type": bypass_type,
            }
        )

    segment_leak_ids = [
        row["id"]
        for row in result_rows
        if row["translation"] and contains_segment_marker(row["translation"])
    ]
    term_sentinel_leak_ids = [
        row["id"]
        for row in result_rows
        if row["translation"] and contains_unrestored_protected_term(row["translation"])
    ]
    accepted_explanation_ids = [
        row["id"]
        for row in result_rows
        if row["translation"]
        and is_explanation_like_output(row["translation"], row["source"])
    ]
    accepted_wrapper_ids = [
        row["id"]
        for row in result_rows
        if row["translation"] and "source_translation_wrapper" in row["validation_warnings"]
    ]

    approved_violation_ids: list[int] = []
    unflagged_approved_violation_ids: list[int] = []
    for row in result_rows:
        if not row["approved_terms"] or not row["translation"]:
            continue
        _, placeholder_map = protect_source_text(
            row["source"],
            row["approved_terms"],
            set(),
        )
        violations = validate_protected_terms(row["translation"], placeholder_map)
        if violations:
            approved_violation_ids.append(row["id"])
            if not row["requires_review"]:
                unflagged_approved_violation_ids.append(row["id"])

    history_by_id = {
        entry["micro_batch_id"]: entry for entry in provider.micro_batch_history
    }
    metadata_mismatch_ids: list[int] = []
    bypass_metadata_ids: list[int] = []
    for row in result_rows:
        batch_id = row["micro_batch_id"]
        if row["bypass_type"] and (batch_id is not None or row["micro_batch_region_ids"]):
            bypass_metadata_ids.append(row["id"])
        if batch_id is None:
            if row["micro_batch_region_ids"]:
                metadata_mismatch_ids.append(row["id"])
            continue
        history = history_by_id.get(batch_id)
        if not history or row["micro_batch_region_ids"] != history["region_ids"]:
            metadata_mismatch_ids.append(row["id"])

    initial_single_item_calls = sum(
        1
        for row in result_rows
        if row["bypass_type"] is None and row["micro_batch_id"] is None
    )
    expected_generation_calls = (
        metrics.micro_batch_requests
        + initial_single_item_calls
        + metrics.single_item_fallback_calls
        + metrics.retries
    )

    structural_validation = {
        "result_object_count": len(result_rows),
        "expected_result_object_count": len(QUALITY_GATE_V5_ITEMS),
        "missing_ids": missing_ids,
        "duplicate_ids": duplicate_ids,
        "segment_order_preserved": returned_ids == expected_ids,
        "segment_marker_leak_ids": segment_leak_ids,
        "term_sentinel_leak_ids": term_sentinel_leak_ids,
        "accepted_chatbot_dictionary_explanation_ids": accepted_explanation_ids,
        "accepted_source_translation_wrapper_ids": accepted_wrapper_ids,
        "approved_glossary_violation_ids": approved_violation_ids,
        "unflagged_approved_glossary_violation_ids": unflagged_approved_violation_ids,
        "micro_batch_metadata_mismatch_ids": metadata_mismatch_ids,
        "bypass_with_micro_batch_metadata_ids": bypass_metadata_ids,
        "system_ui_bypass_count_truthful": metrics.system_ui_bypass_count == 4,
        "term_only_bypass_count_truthful": metrics.term_only_bypass_count == 1,
        "generation_calls_truthful": metrics.generation_call_count == expected_generation_calls,
        "requires_review_ids": [row["id"] for row in result_rows if row["requires_review"]],
    }
    structural_validation["clean"] = all(
        [
            len(result_rows) == len(QUALITY_GATE_V5_ITEMS),
            not missing_ids,
            not duplicate_ids,
            returned_ids == expected_ids,
            not segment_leak_ids,
            not term_sentinel_leak_ids,
            not accepted_explanation_ids,
            not accepted_wrapper_ids,
            not unflagged_approved_violation_ids,
            not metadata_mismatch_ids,
            not bypass_metadata_ids,
            metrics.system_ui_bypass_count == 4,
            metrics.term_only_bypass_count == 1,
            metrics.generation_call_count == expected_generation_calls,
        ]
    )

    execution = {
        "total_items": len(result_rows),
        "model_generation_calls": metrics.generation_call_count,
        "micro_batch_requests": metrics.micro_batch_requests,
        "micro_batch_successes": metrics.micro_batch_successes,
        "micro_batch_fallbacks": metrics.micro_batch_fallbacks,
        "single_item_fallback_calls": metrics.single_item_fallback_calls,
        "initial_single_item_calls": initial_single_item_calls,
        "system_ui_bypass": metrics.system_ui_bypass_count,
        "term_only_bypass": metrics.term_only_bypass_count,
        "retries": metrics.retries,
        "requires_review": sum(1 for row in result_rows if row["requires_review"]),
        "input_tokens": metrics.input_token_count,
        "generated_tokens": metrics.generated_token_count,
        "generation_seconds": round(metrics.generation_seconds, 4),
        "wall_seconds": round(wall_seconds, 4),
        "tokens_per_second": metrics.tokens_per_sec,
        "load_seconds": round(load_seconds, 4),
        "backend": provider.name,
        "model": metrics.translation_model,
    }
    summary = {
        "execution": execution,
        "micro_batches": provider.micro_batch_history,
        "bypasses": [
            {"id": row["id"], "type": row["bypass_type"]}
            for row in result_rows
            if row["bypass_type"]
        ],
        "structural_validation": structural_validation,
    }

    output_dir.mkdir(parents=True)
    (output_dir / "results.json").write_text(
        json.dumps(result_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    text_lines = []
    for row in result_rows:
        text_lines.extend(
            [
                f"[{row['id']:03d}]",
                f"EN: {row['source']}",
                f"TR: {row['translation'] or '<FAILED>'}",
                f"WARNINGS: {', '.join(row['validation_warnings']) or '-'}",
                f"MICRO_BATCH_ID: {row['micro_batch_id'] or '-'}",
                "MICRO_BATCH_REGION_IDS: "
                + (",".join(str(value) for value in row["micro_batch_region_ids"]) or "-"),
                f"BYPASS: {row['bypass_type'] or '-'}",
                "",
            ]
        )
    (output_dir / "results.txt").write_text("\n".join(text_lines), encoding="utf-8")

    print("=== TRANSLATEGEMMA QUALITY GATE V5 COMPLETED ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    run_quality_gate_v5()
