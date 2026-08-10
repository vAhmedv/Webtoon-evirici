"""One-shot single-item TranslateGemma legacy/canonical prompt A/B test.

Only the raw prompt renderer variant changes between A and B. Model, decoding,
preprocessing, terminology protection, cleaners, guards, and retries remain the
same. This script records outputs for manual review and never auto-ranks Turkish
translation quality.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from core.translation.profile_discovery import CandidateStore
from core.translation.protection import contains_unrestored_protected_term
from core.translation.series_profile import SeriesProfile
from providers.translation.base import TranslationInput, TranslationItem, TranslationOutput
from providers.translation.translategemma_gguf_translation import (
    TranslateGemmaGGUFTranslationProvider,
    _PreparedTranslationItem,
    contains_segment_marker,
)
OUTPUT_DIR = Path("benchmark_results/translategemma_prompt_ab")

# Verbatim source set from
# scripts/translategemma_quality_gate_v4.py::QUALITY_GATE_V4_ITEMS.
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


class _CapturingSingleItemProvider(TranslateGemmaGGUFTranslationProvider):
    """Capture the exact prepared source that reaches each single-item path."""

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


def _make_input() -> TranslationInput:
    profile = SeriesProfile(
        series_id="translategemma_prompt_ab_v4",
        glossary={
            "SPIRIT STONE": "Ruh Taşı",
            "GUILD MASTER": "Lonca Lideri",
        },
    )
    store = CandidateStore(series_id="translategemma_prompt_ab_v4")
    items = [
        TranslationItem(region_id=region_id, source=source, reading_order=region_id)
        for region_id, source in QUALITY_GATE_V4_ITEMS
    ]
    return TranslationInput(
        items=items,
        profile=profile,
        candidate_store=store,
        chapter_id="translategemma_prompt_ab_v4",
    )


def _bypass_type(raw_model_response: str) -> str | None:
    if raw_model_response == "[System UI Lexicon]":
        return "SYSTEM_UI"
    if raw_model_response == "[Term-Only Bypass]":
        return "TERM_ONLY"
    return None


def _collect_variant(
    provider: _CapturingSingleItemProvider,
    output: TranslationOutput,
    wall_time: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    warning_counts: Counter[str] = Counter()
    for result in output.results:
        bypass = _bypass_type(result.raw_model_response)
        warnings = list(result.validation_warnings)
        warning_counts.update(warnings)
        rows.append(
            {
                "id": result.region_id,
                "source": result.source,
                "prepared_source": provider.prepared_sources.get(result.region_id),
                "translation": result.translation,
                "raw_model_response": result.raw_model_response,
                "validation_warnings": warnings,
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
        "model_call_items": sum(row["model_called"] for row in rows),
        "system_ui_bypass": metrics.system_ui_bypass_count,
        "term_only_bypass": metrics.term_only_bypass_count,
        "generation_calls": metrics.generation_call_count,
        "retries": metrics.retries,
        "empty_outputs": sum(
            "empty_translation" in row["validation_warnings"] for row in rows
        ),
        "requires_review": sum(row["requires_review"] for row in rows),
        "guard_trigger_counts": dict(sorted(warning_counts.items())),
        "input_tokens": metrics.input_token_count,
        "generated_tokens": metrics.generated_token_count,
        "generation_seconds": round(metrics.generation_seconds, 4),
        "tokens_per_second": metrics.tokens_per_sec,
        "wall_time": round(wall_time, 4),
        "micro_batch_requests": metrics.micro_batch_requests,
    }
    return rows, summary


def _duplicate_ids(rows: list[dict[str, Any]]) -> list[int]:
    ids = [row["id"] for row in rows]
    return sorted({region_id for region_id in ids if ids.count(region_id) > 1})


def _format_warnings(warnings: list[str]) -> str:
    return ", ".join(warnings) if warnings else "none"


def run_prompt_ab_test() -> None:
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"Fresh-run guard: output directory already exists: {OUTPUT_DIR}")
    if len(QUALITY_GATE_V4_ITEMS) != 32:
        raise RuntimeError("Expected the exact 32-item V4 quality-gate dataset")

    variant_a = _CapturingSingleItemProvider(
        managed=True,
        micro_batch_enabled=False,
        prompt_variant="legacy",
    )
    variant_b = _CapturingSingleItemProvider(
        managed=False,
        micro_batch_enabled=False,
        prompt_variant="canonical",
    )

    variant_a.load()
    try:
        a_started = time.perf_counter()
        output_a = variant_a.translate(_make_input())
        a_wall_time = time.perf_counter() - a_started

        b_started = time.perf_counter()
        output_b = variant_b.translate(_make_input())
        b_wall_time = time.perf_counter() - b_started
    finally:
        variant_a.unload()

    rows_a, summary_a = _collect_variant(variant_a, output_a, a_wall_time)
    rows_b, summary_b = _collect_variant(variant_b, output_b, b_wall_time)
    by_id_a = {row["id"]: row for row in rows_a}
    by_id_b = {row["id"]: row for row in rows_b}
    expected_source_by_id = dict(QUALITY_GATE_V4_ITEMS)
    expected_ids = list(expected_source_by_id)

    comparison_rows: list[dict[str, Any]] = []
    for region_id in expected_ids:
        row_a = by_id_a.get(region_id)
        row_b = by_id_b.get(region_id)
        if row_a is None or row_b is None:
            continue
        same_prepared = row_a["prepared_source"] == row_b["prepared_source"]
        comparison_rows.append(
            {
                "id": region_id,
                "source": expected_source_by_id[region_id],
                "prepared_source": row_a["prepared_source"],
                "variant_a": {
                    "translation": row_a["translation"],
                    "raw_model_response": row_a["raw_model_response"],
                    "validation_warnings": row_a["validation_warnings"],
                    "requires_review": row_a["requires_review"],
                },
                "variant_b": {
                    "translation": row_b["translation"],
                    "raw_model_response": row_b["raw_model_response"],
                    "validation_warnings": row_b["validation_warnings"],
                    "requires_review": row_b["requires_review"],
                },
                "same_prepared_source": same_prepared,
                "model_called": row_a["model_called"] and row_b["model_called"],
                "bypass_type": row_a["bypass_type"],
            }
        )

    prepared_source_mismatch_ids = [
        row["id"] for row in comparison_rows if not row["same_prepared_source"]
    ]
    model_call_mismatch_ids = [
        region_id
        for region_id in expected_ids
        if region_id in by_id_a
        and region_id in by_id_b
        and by_id_a[region_id]["model_called"] != by_id_b[region_id]["model_called"]
    ]
    source_mismatch_ids = [
        region_id
        for region_id, expected_source in expected_source_by_id.items()
        if (
            region_id not in by_id_a
            or region_id not in by_id_b
            or by_id_a[region_id]["source"] != expected_source
            or by_id_b[region_id]["source"] != expected_source
        )
    ]

    def leak_ids(rows: list[dict[str, Any]], predicate) -> list[int]:
        return [
            row["id"]
            for row in rows
            if row["translation"] and predicate(row["translation"])
        ]

    structural = {
        "variant_a_missing_ids": sorted(set(expected_ids) - set(by_id_a)),
        "variant_b_missing_ids": sorted(set(expected_ids) - set(by_id_b)),
        "variant_a_duplicate_ids": _duplicate_ids(rows_a),
        "variant_b_duplicate_ids": _duplicate_ids(rows_b),
        "source_mismatch_ids": source_mismatch_ids,
        "prepared_source_mismatch_ids": prepared_source_mismatch_ids,
        "model_call_mismatch_ids": model_call_mismatch_ids,
        "variant_a_sentinel_leak_ids": leak_ids(
            rows_a,
            contains_unrestored_protected_term,
        ),
        "variant_b_sentinel_leak_ids": leak_ids(
            rows_b,
            contains_unrestored_protected_term,
        ),
        "variant_a_segment_marker_leak_ids": leak_ids(rows_a, contains_segment_marker),
        "variant_b_segment_marker_leak_ids": leak_ids(rows_b, contains_segment_marker),
        "variant_a_server_error_ids": [
            row["id"]
            for row in rows_a
            if "translation_server_error" in row["validation_warnings"]
        ],
        "variant_b_server_error_ids": [
            row["id"]
            for row in rows_b
            if "translation_server_error" in row["validation_warnings"]
        ],
        "variant_a_micro_batch_requests": variant_a.metrics.micro_batch_requests,
        "variant_b_micro_batch_requests": variant_b.metrics.micro_batch_requests,
    }
    structural["clean"] = not any(
        value
        for key, value in structural.items()
        if key != "clean"
    )

    if prepared_source_mismatch_ids:
        raise RuntimeError(
            f"A/B prepared-source mismatch for IDs: {prepared_source_mismatch_ids}"
        )
    if variant_a.metrics.micro_batch_requests or variant_b.metrics.micro_batch_requests:
        raise RuntimeError("Prompt A/B benchmark must not issue micro-batch requests")

    OUTPUT_DIR.mkdir(parents=True)
    (OUTPUT_DIR / "variant_a_results.json").write_text(
        json.dumps(rows_a, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "variant_b_results.json").write_text(
        json.dumps(rows_b, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "comparison.json").write_text(
        json.dumps(comparison_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model_rows = [row for row in comparison_rows if row["model_called"]]
    bypass_rows = [row for row in comparison_rows if not row["model_called"]]
    text_lines: list[str] = []
    separator = "=" * 60
    for row in model_rows:
        text_lines.extend(
            [
                separator,
                f"[{row['id']:03d}]",
                "",
                "SOURCE:",
                row["source"],
                "",
                "PREPARED SOURCE:",
                row["prepared_source"] or "<none>",
                "",
                "A — LEGACY:",
                row["variant_a"]["translation"] or "<FAILED>",
                "",
                "B — CANONICAL:",
                row["variant_b"]["translation"] or "<FAILED>",
                "",
                "A WARNINGS:",
                _format_warnings(row["variant_a"]["validation_warnings"]),
                "",
                "B WARNINGS:",
                _format_warnings(row["variant_b"]["validation_warnings"]),
                separator,
                "",
            ]
        )

    text_lines.extend(["BYPASS ITEMS", separator, ""])
    for row in bypass_rows:
        text_lines.extend(
            [
                f"[{row['id']:03d}] {row['bypass_type']}",
                f"SOURCE: {row['source']}",
                f"A: {row['variant_a']['translation'] or '<FAILED>'}",
                f"B: {row['variant_b']['translation'] or '<FAILED>'}",
                "",
            ]
        )
    (OUTPUT_DIR / "comparison.txt").write_text(
        "\n".join(text_lines),
        encoding="utf-8",
    )

    summary = {
        "dataset": {
            "source": "scripts/translategemma_quality_gate_v4.py::QUALITY_GATE_V4_ITEMS",
            "items": len(QUALITY_GATE_V4_ITEMS),
        },
        "variant_a_legacy": summary_a,
        "variant_b_canonical": summary_b,
        "structural_comparison": structural,
        "quality_winner": None,
        "quality_winner_note": "Manual human review required; no semantic auto-ranking performed.",
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== TRANSLATEGEMMA PROMPT A/B COMPLETED ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Human comparison: {OUTPUT_DIR / 'comparison.txt'}")


if __name__ == "__main__":
    run_prompt_ab_test()
