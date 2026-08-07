"""Benchmark raporu üretici.

JSON ve human-readable raporlar.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

from benchmark.metrics import DetectorBenchmarkResult


def generate_comparison_md(results: Sequence[DetectorBenchmarkResult]) -> str:
    """Markdown karşılaştırma tablosu üretir."""
    lines = [
        "# Detector Benchmark Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "| Detector | Precision | Recall | F1 | FP | FN | VRAM (MB) | Time (s) |",
        "| -------- | --------: | -----: | -: | -: | -: | --------: | -------: |",
    ]
    for r in results:
        vram = f"{r.peak_vram_mb:.0f}" if r.peak_vram_mb is not None else "N/A"
        lines.append(
            f"| {r.detector_name} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} "
            f"| {r.false_positives} | {r.false_negatives} | {vram} | {r.inference_time:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def save_results(
    results: Sequence[DetectorBenchmarkResult],
    output_dir: str | Path,
) -> Path:
    """Benchmark sonuçlarını JSON ve markdown olarak kaydeder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "detectors": [],
    }
    for r in results:
        summary["detectors"].append(
            {
                "detector_name": r.detector_name,
                "model_version": r.model_version,
                "device": r.device,
                "chapter_path": r.chapter_path,
                "pages": r.pages,
                "windows": r.windows,
                "detections_before_merge": r.detections_before_merge,
                "regions_after_merge": r.regions_after_merge,
                "true_positives": r.true_positives,
                "false_positives": r.false_positives,
                "false_negatives": r.false_negatives,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "boundary_success": r.boundary_success,
                "boundary_total": r.boundary_total,
                "no_text_fp": r.no_text_fp,
                "load_time": r.load_time,
                "inference_time": r.inference_time,
                "avg_window_time": r.avg_window_time,
                "peak_vram_mb": r.peak_vram_mb,
                "warnings": r.warnings,
            }
        )

    json_path = output_dir / "comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md_path = output_dir / "comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_comparison_md(results))

    return md_path
