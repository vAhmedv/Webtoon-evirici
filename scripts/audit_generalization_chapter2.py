"""Generalization Audit Script for Real Chapter 2.

Runs Chapter 2 through the real ChapterAnalyzer.process_chapter() pipeline,
verifies source immutability, tests multi-signal classification generalizability,
and checks full lifecycle reconciliation equations.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from application.chapter_analyzer import ChapterAnalyzer, load_config
from core.detection import RegionStatus, RegionType

SOURCE_CHAPTER = Path(
    r"C:\Users\Ahmed\AppData\Local\Tachidesk\downloads\mangas\Asmodeus Scans (EN)\Reincarnated as a God-Tier Crafter\Chapter 2"
)
OUTPUT_DIR = Path(r"audit_output/generalization_test")


def compute_file_hash_and_mtime(p: Path) -> tuple[str, float]:
    content = p.read_bytes()
    h = hashlib.sha256(content).hexdigest()
    mtime = p.stat().st_mtime
    return h, mtime


def snapshot_directory(d: Path) -> dict[str, tuple[str, float]]:
    snapshots = {}
    for f in sorted(list(d.glob("*"))):
        if f.is_file():
            snapshots[f.name] = compute_file_hash_and_mtime(f)
    return snapshots


def main() -> None:
    print("=" * 80)
    print("RUNNING REAL CHAPTER 2 GENERALIZATION AUDIT")
    print("=" * 80)

    assert SOURCE_CHAPTER.exists(), f"Source chapter directory not found: {SOURCE_CHAPTER}"
    assert SOURCE_CHAPTER.resolve() != OUTPUT_DIR.resolve(), "Output directory must not equal source directory!"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Snapshot source file integrity before E2E
    pre_snapshots = snapshot_directory(SOURCE_CHAPTER)
    print(f"[OK] Pre-E2E snapshot recorded for {len(pre_snapshots)} source files in Chapter 2.")

    from providers.detector.ctd import ComicTextDetector
    from providers.ocr.paddleocr import PaddleOCRProvider
    from providers.ocr.paddleocr_vl import PaddleOCRVLOcrProvider
    from providers.ocr.qwen_repair import QwenRepairProvider
    from providers.translation.hy_mt2_gguf_translation import HyMT2GGUFTranslationProvider

    detector = ComicTextDetector()
    primary_ocr = PaddleOCRProvider("PP-OCRv6_medium_rec")
    verifier_ocr = PaddleOCRVLOcrProvider()
    qwen_repair = QwenRepairProvider()
    translator = HyMT2GGUFTranslationProvider()

    analyzer = ChapterAnalyzer()
    analyzer._cache.enabled = False

    t_start = time.time()
    pipeline_result = analyzer.process_chapter(
        SOURCE_CHAPTER,
        OUTPUT_DIR,
        detector=detector,
        primary_ocr=primary_ocr,
        verifier_ocr=verifier_ocr,
        qwen_repair=qwen_repair,
        translator=translator,
    )
    elapsed = time.time() - t_start
    print(f"[OK] Production pipeline finished in {elapsed:.2f} s.")

    # 2. Snapshot source file integrity after E2E and verify 0 mutations
    post_snapshots = snapshot_directory(SOURCE_CHAPTER)
    assert len(pre_snapshots) == len(post_snapshots), "File count mismatch after E2E!"

    mutated_files = []
    for fname, (pre_h, pre_m) in pre_snapshots.items():
        assert fname in post_snapshots, f"Source file missing after E2E: {fname}"
        post_h, post_m = post_snapshots[fname]
        if pre_h != post_h or pre_m != post_m:
            mutated_files.append(fname)

    assert not mutated_files, f"Source file mutation detected! Mutated files: {mutated_files}"
    print("[OK] Source immutability verified: 0 source files mutated (SHA256 & mtimes identical).")

    # 3. Extract and verify metrics
    regions_json_path = OUTPUT_DIR / "analysis" / "regions.json"
    summary_json_path = OUTPUT_DIR / "analysis" / "summary.json"

    with open(regions_json_path, "r", encoding="utf-8") as f:
        regions_data = json.load(f)

    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    regions = regions_data["regions"]

    final_auto = sum(1 for r in regions if r.get("status") == "auto")
    final_review = sum(1 for r in regions if r.get("status") == "review")
    final_skip = sum(1 for r in regions if r.get("status") == "skip")

    # Multi-signal filter breakdown
    noise_skip_count = sum(1 for r in regions if r.get("review_reason") == "non_text_noise_skip")
    sfx_skip_count = sum(1 for r in regions if r.get("review_reason") == "sfx_skip")
    credit_skip_count = sum(1 for r in regions if r.get("review_reason") == "credit_metadata_skip")
    ambig_review_count = sum(1 for r in regions if r.get("review_reason") == "ambiguous_unknown_review")

    metrics = {
        "source_page_count": len(pipeline_result.pages),
        "output_page_count": len(pipeline_result.exported_page_paths),
        "total_regions": len(regions),
        "final_auto_regions": final_auto,
        "final_review_regions": final_review,
        "final_skip_regions": final_skip,
        "non_text_noise_skip_count": noise_skip_count,
        "sfx_skip_count": sfx_skip_count,
        "credit_metadata_skip_count": credit_skip_count,
        "ambiguous_unknown_review_count": ambig_review_count,
        "text_block_count": regions_data["text_blocks_count"],
        "translation_eligible_blocks_count": regions_data["translation_eligible_blocks_count"],
        "translated_blocks_count": summary_data["translated_blocks_count"],
        "successfully_inpainted_blocks_count": summary_data["inpainted_blocks_count"],
        "review_inpaint_blocks_count": summary_data["review_inpaint_blocks_count"],
        "actually_rendered_blocks_count": summary_data["rendered_blocks_count"],
        "overflow_blocks_count": summary_data["overflow_blocks_count"],
        "ocr_elapsed_seconds": round(pipeline_result.ocr_elapsed_time, 2),
        "translation_elapsed_seconds": round(pipeline_result.translation_elapsed_time, 2),
        "inpainting_rendering_elapsed_seconds": round(
            pipeline_result.inpainting_rendering_elapsed_time, 2
        ),
        "elapsed_seconds": round(elapsed, 2),
    }

    assert metrics["output_page_count"] == metrics["source_page_count"], "Output page count mismatch!"

    # Lifecycle equations verification
    tb_total = regions_data["text_blocks_count"]
    pre_inpaint_skipped = regions_data["pre_inpaint_skipped_blocks_count"]
    trans_eligible = regions_data["translation_eligible_blocks_count"]
    assert tb_total == pre_inpaint_skipped + trans_eligible, "Eq 1 mismatch!"

    translated = regions_data["translated_blocks_count"]
    trans_failed = regions_data["translation_failed_blocks_count"]
    assert trans_eligible == translated + trans_failed, "Eq 2 mismatch!"

    inp_success = regions_data["inpainted_blocks_count"]
    inp_review = regions_data["review_inpaint_blocks_count"]
    assert translated == inp_success + inp_review, "Eq 3 mismatch!"

    rendered = regions_data["rendered_blocks_count"]
    overflow = regions_data["overflow_blocks_count"]
    assert rendered == inp_success - overflow, "Eq 4 mismatch!"
    assert overflow == 0, "Overflow must be 0!"

    metrics_out = OUTPUT_DIR / "generalization_metrics.json"
    metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("CHAPTER 2 GENERALIZATION AUDIT METRICS SUMMARY:")
    print("=" * 80)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\n[OK] Metrics saved to {metrics_out}")


if __name__ == "__main__":
    main()
