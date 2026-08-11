from __future__ import annotations

import json
from pathlib import Path

from scripts.qwen_translation_pipeline_fix_v1 import (
    EXPECTED_OUTPUT_FILES,
    MANDATORY_AXE_IDS,
    _structural_metrics,
    select_gate_items,
    validate_clean_input,
)


INPUT = Path("benchmark_results/real_chapter_translation_gate_v1_clean/valid_story_items.json")


def test_gate_selection_is_30_unique_balanced_and_covers_all_chapters() -> None:
    items = json.loads(INPUT.read_text(encoding="utf-8"))
    selected = select_gate_items(items)
    assert len(selected) == len({item["id"] for item in selected}) == 30
    assert sum(item["series"].startswith("Axe God") for item in selected) == 15
    assert sum(item["series"].startswith("Reincarnated") for item in selected) == 15
    assert len({(item["series"], item["chapter"]) for item in selected}) == 10
    assert set(MANDATORY_AXE_IDS) <= {item["id"] for item in selected}
    validate_clean_input(items, selected)


def test_gate_output_contract_is_exactly_eight_files() -> None:
    assert len(EXPECTED_OUTPUT_FILES) == 8
    assert "summary.json" in EXPECTED_OUTPUT_FILES
    assert "before_after_comparison.txt" in EXPECTED_OUTPUT_FILES


def test_morphology_guard_distinguishes_repaired_lowercase_suffix() -> None:
    base = {
        "id": "x",
        "old_stored_qwen": {"requires_review": False, "warnings": []},
        "requires_review": False,
        "warnings": [],
        "detected_named_terms": [],
    }
    repaired = _structural_metrics([{**base, "final_restored": "yetenek kullanıcısıdır"}])
    broken = _structural_metrics([{**base, "final_restored": "yetenek kullanıcısıDIR"}])
    assert repaired["morphology_artifact_ids"] == []
    assert broken["morphology_artifact_ids"] == ["x"]
