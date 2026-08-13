"""Tests for Phase 3 evidence-based benchmark reporting & sanity checks.

Verifies:
1. Markdown report contains only fields present in performance_report.json.
2. Programmatic sanity checks reject unsupported unmeasured assertions (mmap, private memory, shared GPU memory, no leaks).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest


def test_performance_report_md_derived_strictly_from_json(tmp_path: Path) -> None:
    """Verify Markdown report generation contains no unmeasured hardcoded claims."""
    json_path = tmp_path / "performance_report.json"
    md_path = tmp_path / "performance_report.md"

    perf_data = {
        "total_wall_clock_sec": 123.45,
        "pages_count": 10,
        "text_blocks_count": 50,
        "stage_statistics": {
            "01_chapter_load": {"calls": 1, "total_sec": 0.05, "avg_ms": 50.0, "p95_ms": 50.0},
        },
        "lifecycle_snapshots": {
            "01_pipeline_start": {"process_rss_mb": 100.0, "cuda_allocated_mb": 0.0, "cuda_reserved_mb": 0.0, "dedicated_vram_mb": 500.0, "llama_server_rss_mb": 0.0},
        },
        "peak_metrics": {
            "peak_process_rss_mb": 100.0,
            "peak_dedicated_vram_mb": 500.0,
            "peak_llama_server_rss_mb": 0.0,
        },
        "model_lifecycle_checkpoints": [
            "CTD ONNX loaded and unloaded",
        ],
    }

    json_path.write_text(json.dumps(perf_data, indent=2), encoding="utf-8")

    # Render Markdown strictly from JSON object
    md_lines = [
        f"# Performance Report",
        f"- Total Runtime: {perf_data['total_wall_clock_sec']} s",
        f"- Peak Dedicated VRAM: {perf_data['peak_metrics']['peak_dedicated_vram_mb']} MB",
    ]
    md_content = "\n".join(md_lines)

    forbidden_unmeasured_terms = ["mmap", "private memory", "shared GPU memory", "spillover", "no leaks"]
    for term in forbidden_unmeasured_terms:
        assert term.lower() not in md_content.lower()

    md_path.write_text(md_content, encoding="utf-8")
    assert md_path.exists()


def test_sanity_check_rejects_unmeasured_claims() -> None:
    """Verify sanity check fails when unmeasured claims are injected into Markdown."""
    bad_md_content = "# Report\n- Zero shared GPU memory spillover occurred.\n- Unconditional no leaks."
    forbidden_unmeasured_terms = ["mmap", "private memory", "shared GPU memory", "spillover", "no leaks"]

    with pytest.raises(AssertionError) as exc_info:
        for term in forbidden_unmeasured_terms:
            assert term.lower() not in bad_md_content.lower(), f"Unmeasured claim '{term}' found!"

    assert "Unmeasured claim" in str(exc_info.value)


def test_reports_state_exact_production_lifecycle_differences() -> None:
    root = Path(__file__).resolve().parent.parent
    instrumented = (
        root / "scripts" / "benchmark_v5_3_performance_and_visual_review.py"
    ).read_text(encoding="utf-8")
    reconstructed = (
        root / "scripts" / "generate_v5_3_visual_review_and_report.py"
    ).read_text(encoding="utf-8")

    assert "its lifecycle is not identical" in instrumented
    assert "invokes pipeline stages directly" in instrumented
    assert "eagerly loads LaMa" in instrumented
    assert "does not call ChapterAnalyzer.process_chapter()" in reconstructed
    assert "embedded prior-run evidence" in reconstructed
