"""One-shot TranslateGemma minimal-faithful prompt C test and A/B/C report.

The existing A/B artifacts are read-only inputs. Only variant C performs real-model
inference. Translation quality is left entirely to human review.
"""
from __future__ import annotations

import json
import runpy
import time
from collections import Counter
from pathlib import Path
from typing import Any

from core.translation.protection import contains_unrestored_protected_term
from providers.translation.translategemma_gguf_translation import contains_segment_marker

_AB_HELPERS = runpy.run_path(
    str(Path(__file__).with_name("translategemma_prompt_ab_test.py"))
)
QUALITY_GATE_V4_ITEMS: list[tuple[int, str]] = _AB_HELPERS["QUALITY_GATE_V4_ITEMS"]
_CapturingSingleItemProvider = _AB_HELPERS["_CapturingSingleItemProvider"]
_collect_variant = _AB_HELPERS["_collect_variant"]
_duplicate_ids = _AB_HELPERS["_duplicate_ids"]
_format_warnings = _AB_HELPERS["_format_warnings"]
_make_input = _AB_HELPERS["_make_input"]

AB_OUTPUT_DIR = Path("benchmark_results/translategemma_prompt_ab")
OUTPUT_DIR = Path("benchmark_results/translategemma_prompt_abc")
CRITICAL_REVIEW_IDS = (3, 6, 7, 9, 12, 19, 25, 29)
EXPECTED_MODEL_PATH = r"C:\AI\Models\translategemma-12b-it-q5_k_m.gguf"

_EXPLICIT_C_GUARDS = {
    "chatbot_or_explanation_output": "chatbot_or_explanation_output",
    "source_translation_wrapper": "source_translation_wrapper",
    "untranslated_source_prose": "untranslated_source_prose",
    "unrestored_protected_term": "unrestored_protected_term",
    "empty_translation": "empty_translation",
    "server_error": "translation_server_error",
    "segment_marker_leak": "segment_marker_leak",
}


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"Required A/B artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_existing_variant(
    rows: Any,
    label: str,
    expected_source_by_id: dict[int, str],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{label} results must be a JSON list of objects")
    typed_rows: list[dict[str, Any]] = rows
    invalid_ids = [row.get("id") for row in typed_rows if not isinstance(row.get("id"), int)]
    if invalid_ids:
        raise RuntimeError(f"{label} contains invalid IDs: {invalid_ids}")
    by_id: dict[int, dict[str, Any]] = {row["id"]: row for row in typed_rows}
    expected_ids = set(expected_source_by_id)
    actual_ids = set(by_id)
    duplicate_ids = _duplicate_ids(typed_rows)
    if len(typed_rows) != len(expected_ids) or actual_ids != expected_ids or duplicate_ids:
        raise RuntimeError(
            f"{label} artifact is not the exact 32-item V4 result set: "
            f"count={len(typed_rows)}, missing={sorted(expected_ids - actual_ids)}, "
            f"extra={sorted(actual_ids - expected_ids)}, duplicates={duplicate_ids}"
        )
    source_mismatch_ids = [
        region_id
        for region_id, source in expected_source_by_id.items()
        if by_id[region_id].get("source") != source
    ]
    if source_mismatch_ids:
        raise RuntimeError(f"{label} source mismatch for IDs: {source_mismatch_ids}")
    return typed_rows


def _variant_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "translation": row["translation"],
        "raw_model_response": row["raw_model_response"],
        "validation_warnings": row["validation_warnings"],
        "requires_review": row["requires_review"],
        "model_called": row["model_called"],
        "bypass_type": row["bypass_type"],
    }


def _leak_ids(rows: list[dict[str, Any]], predicate) -> list[int]:
    return [
        row["id"]
        for row in rows
        if row["translation"] and predicate(row["translation"])
    ]


def _server_error_ids(rows: list[dict[str, Any]]) -> list[int]:
    return [
        row["id"]
        for row in rows
        if "translation_server_error" in row["validation_warnings"]
    ]


def _comparison_text(comparison_rows: list[dict[str, Any]]) -> str:
    model_rows = [row for row in comparison_rows if row["model_called"]]
    bypass_rows = [row for row in comparison_rows if not row["model_called"]]
    separator = "=" * 60
    lines: list[str] = []

    for row in model_rows:
        lines.extend(
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
                "C — MINIMAL FAITHFUL:",
                row["variant_c"]["translation"] or "<FAILED>",
                "",
                "A WARNINGS:",
                _format_warnings(row["variant_a"]["validation_warnings"]),
                "",
                "B WARNINGS:",
                _format_warnings(row["variant_b"]["validation_warnings"]),
                "",
                "C WARNINGS:",
                _format_warnings(row["variant_c"]["validation_warnings"]),
                separator,
                "",
            ]
        )

    lines.extend(["BYPASS ITEMS", separator, ""])
    for row in bypass_rows:
        lines.extend(
            [
                f"[{row['id']:03d}] {row['bypass_type']}",
                f"SOURCE: {row['source']}",
                f"A: {row['variant_a']['translation'] or '<FAILED>'}",
                f"B: {row['variant_b']['translation'] or '<FAILED>'}",
                f"C: {row['variant_c']['translation'] or '<FAILED>'}",
                f"A WARNINGS: {_format_warnings(row['variant_a']['validation_warnings'])}",
                f"B WARNINGS: {_format_warnings(row['variant_b']['validation_warnings'])}",
                f"C WARNINGS: {_format_warnings(row['variant_c']['validation_warnings'])}",
                "",
            ]
        )

    by_id = {row["id"]: row for row in comparison_rows}
    lines.extend(["CRITICAL MANUAL REVIEW TARGETS", separator, ""])
    for region_id in CRITICAL_REVIEW_IDS:
        row = by_id[region_id]
        lines.extend(
            [
                f"[{region_id:03d}]",
                f"SOURCE: {row['source']}",
                f"A: {row['variant_a']['translation'] or '<FAILED>'}",
                f"B: {row['variant_b']['translation'] or '<FAILED>'}",
                f"C: {row['variant_c']['translation'] or '<FAILED>'}",
                "",
            ]
        )
    return "\n".join(lines)


def run_prompt_c_test() -> None:
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"Fresh-run guard: output directory already exists: {OUTPUT_DIR}")
    if len(QUALITY_GATE_V4_ITEMS) != 32:
        raise RuntimeError("Expected the exact 32-item V4 quality-gate dataset")

    expected_source_by_id = dict(QUALITY_GATE_V4_ITEMS)
    expected_ids = list(expected_source_by_id)
    if len(expected_source_by_id) != 32:
        raise RuntimeError("V4 quality-gate IDs must be unique")

    rows_a = _validate_existing_variant(
        _load_json(AB_OUTPUT_DIR / "variant_a_results.json"),
        "Variant A",
        expected_source_by_id,
    )
    rows_b = _validate_existing_variant(
        _load_json(AB_OUTPUT_DIR / "variant_b_results.json"),
        "Variant B",
        expected_source_by_id,
    )
    ab_summary = _load_json(AB_OUTPUT_DIR / "summary.json")
    by_id_a = {row["id"]: row for row in rows_a}
    by_id_b = {row["id"]: row for row in rows_b}

    existing_prepared_mismatch_ids = [
        region_id
        for region_id in expected_ids
        if by_id_a[region_id]["prepared_source"]
        != by_id_b[region_id]["prepared_source"]
    ]
    existing_execution_mismatch_ids = [
        region_id
        for region_id in expected_ids
        if (
            by_id_a[region_id]["model_called"] != by_id_b[region_id]["model_called"]
            or by_id_a[region_id]["bypass_type"] != by_id_b[region_id]["bypass_type"]
        )
    ]
    if existing_prepared_mismatch_ids or existing_execution_mismatch_ids:
        raise RuntimeError(
            "Existing A/B structural mismatch: "
            f"prepared={existing_prepared_mismatch_ids}, "
            f"execution={existing_execution_mismatch_ids}"
        )

    variant_c = _CapturingSingleItemProvider(
        managed=True,
        micro_batch_enabled=False,
        prompt_variant="minimal_faithful",
    )
    if variant_c.model_path != EXPECTED_MODEL_PATH:
        raise RuntimeError(f"Unexpected TranslateGemma model path: {variant_c.model_path}")

    variant_c.load()
    try:
        c_started = time.perf_counter()
        output_c = variant_c.translate(_make_input())
        c_wall_time = time.perf_counter() - c_started
    finally:
        variant_c.unload()

    rows_c, summary_c = _collect_variant(variant_c, output_c, c_wall_time)
    rows_c = _validate_existing_variant(
        rows_c,
        "Variant C",
        expected_source_by_id,
    )
    by_id_c = {row["id"]: row for row in rows_c}

    comparison_rows: list[dict[str, Any]] = []
    for region_id in expected_ids:
        row_a = by_id_a[region_id]
        row_b = by_id_b[region_id]
        row_c = by_id_c.get(region_id)
        if row_c is None:
            continue
        same_prepared_source = (
            row_a["prepared_source"]
            == row_b["prepared_source"]
            == row_c["prepared_source"]
        )
        comparison_rows.append(
            {
                "id": region_id,
                "source": expected_source_by_id[region_id],
                "prepared_source": row_c["prepared_source"],
                "variant_a": _variant_view(row_a),
                "variant_b": _variant_view(row_b),
                "variant_c": _variant_view(row_c),
                "same_prepared_source": same_prepared_source,
                "model_called": (
                    row_a["model_called"]
                    and row_b["model_called"]
                    and row_c["model_called"]
                ),
                "bypass_type": row_c["bypass_type"],
            }
        )

    source_mismatch_ids = [
        region_id
        for region_id, source in expected_source_by_id.items()
        if (
            region_id not in by_id_c
            or by_id_c[region_id]["source"] != source
            or by_id_a[region_id]["source"] != source
            or by_id_b[region_id]["source"] != source
        )
    ]
    prepared_source_mismatch_ids = [
        row["id"] for row in comparison_rows if not row["same_prepared_source"]
    ]
    model_call_mismatch_ids = [
        region_id
        for region_id in expected_ids
        if region_id in by_id_c
        and not (
            by_id_a[region_id]["model_called"]
            == by_id_b[region_id]["model_called"]
            == by_id_c[region_id]["model_called"]
        )
    ]
    bypass_type_mismatch_ids = [
        region_id
        for region_id in expected_ids
        if region_id in by_id_c
        and not (
            by_id_a[region_id]["bypass_type"]
            == by_id_b[region_id]["bypass_type"]
            == by_id_c[region_id]["bypass_type"]
        )
    ]

    structural = {
        "variant_a_missing_ids": sorted(set(expected_ids) - set(by_id_a)),
        "variant_b_missing_ids": sorted(set(expected_ids) - set(by_id_b)),
        "variant_c_missing_ids": sorted(set(expected_ids) - set(by_id_c)),
        "variant_a_duplicate_ids": _duplicate_ids(rows_a),
        "variant_b_duplicate_ids": _duplicate_ids(rows_b),
        "variant_c_duplicate_ids": _duplicate_ids(rows_c),
        "source_mismatch_ids": source_mismatch_ids,
        "prepared_source_mismatch_ids": prepared_source_mismatch_ids,
        "model_call_mismatch_ids": model_call_mismatch_ids,
        "bypass_type_mismatch_ids": bypass_type_mismatch_ids,
        "variant_a_sentinel_leak_ids": _leak_ids(
            rows_a, contains_unrestored_protected_term
        ),
        "variant_b_sentinel_leak_ids": _leak_ids(
            rows_b, contains_unrestored_protected_term
        ),
        "variant_c_sentinel_leak_ids": _leak_ids(
            rows_c, contains_unrestored_protected_term
        ),
        "variant_a_segment_marker_leak_ids": _leak_ids(rows_a, contains_segment_marker),
        "variant_b_segment_marker_leak_ids": _leak_ids(rows_b, contains_segment_marker),
        "variant_c_segment_marker_leak_ids": _leak_ids(rows_c, contains_segment_marker),
        "variant_a_server_error_ids": _server_error_ids(rows_a),
        "variant_b_server_error_ids": _server_error_ids(rows_b),
        "variant_c_server_error_ids": _server_error_ids(rows_c),
        "variant_c_micro_batch_requests": variant_c.metrics.micro_batch_requests,
    }
    structural["clean"] = not any(structural.values())

    if prepared_source_mismatch_ids:
        raise RuntimeError(
            f"A/B/C prepared-source mismatch for IDs: {prepared_source_mismatch_ids}"
        )
    if model_call_mismatch_ids or bypass_type_mismatch_ids:
        raise RuntimeError(
            "A/B/C execution path mismatch: "
            f"model={model_call_mismatch_ids}, bypass={bypass_type_mismatch_ids}"
        )
    if variant_c.metrics.micro_batch_requests:
        raise RuntimeError("Variant C benchmark must not issue micro-batch requests")

    c_warning_counts = Counter(
        warning for row in rows_c for warning in row["validation_warnings"]
    )
    summary_c["guard_results"] = {
        report_name: c_warning_counts[warning_name]
        for report_name, warning_name in _EXPLICIT_C_GUARDS.items()
    }

    critical_rows = [
        {
            "id": region_id,
            "source": expected_source_by_id[region_id],
            "variant_a": _variant_view(by_id_a[region_id]),
            "variant_b": _variant_view(by_id_b[region_id]),
            "variant_c": _variant_view(by_id_c[region_id]),
        }
        for region_id in CRITICAL_REVIEW_IDS
    ]
    summary = {
        "dataset": {
            "source": "scripts/translategemma_quality_gate_v4.py::QUALITY_GATE_V4_ITEMS",
            "items": len(QUALITY_GATE_V4_ITEMS),
        },
        "variant_a_legacy": ab_summary["variant_a_legacy"],
        "variant_b_canonical": ab_summary["variant_b_canonical"],
        "variant_c_minimal_faithful": summary_c,
        "structural_comparison": structural,
        "critical_manual_review_targets": critical_rows,
        "quality_winner": None,
        "quality_winner_note": (
            "Manual human review required; no semantic heuristic or auto-ranking performed."
        ),
        "production_default": "legacy",
    }

    serialized_outputs = {
        "variant_c_results.json": json.dumps(rows_c, ensure_ascii=False, indent=2),
        "comparison_abc.json": json.dumps(
            comparison_rows, ensure_ascii=False, indent=2
        ),
        "comparison_abc.txt": _comparison_text(comparison_rows),
        "summary_abc.json": json.dumps(summary, ensure_ascii=False, indent=2),
    }
    OUTPUT_DIR.mkdir(parents=True)
    for filename, content in serialized_outputs.items():
        (OUTPUT_DIR / filename).write_text(content, encoding="utf-8")

    print("=== TRANSLATEGEMMA PROMPT VARIANT C COMPLETED ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Human comparison: {OUTPUT_DIR / 'comparison_abc.txt'}")


if __name__ == "__main__":
    run_prompt_c_test()
