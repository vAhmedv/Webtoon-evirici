"""Real Chapter 1 Production E2E Audit Script.

Runs Chapter 1 through the real ChapterAnalyzer.process_chapter() pipeline,
snapshots source file integrity before/after, and collects exact operational metrics.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hashlib
import json
import time
from typing import Any

from application.chapter_analyzer import ChapterAnalyzer, load_config
from core.detection import RegionStatus, RegionType

SOURCE_CHAPTER = Path(
    r"C:\Users\Ahmed\AppData\Local\Tachidesk\downloads\mangas\Asmodeus Scans (EN)\Reincarnated as a God-Tier Crafter\Chapter 1"
)
OUTPUT_DIR = Path(r"audit_output/real_chapter1_e2e")


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


def summarize_final_region_states(regions: list[dict[str, Any]]) -> dict[str, int]:
    """Count states in the final, post-classification regions.json payload.

    These are snapshot counts, not UNKNOWN-to-status transition counters.  CTD
    emits UNKNOWN detections, but classification may assign a final story-text
    type before this payload is serialized.
    """
    return {
        "final_serialized_region_count": len(regions),
        "final_unknown_skip_regions": sum(
            1 for region in regions
            if region.get("type") == RegionType.UNKNOWN.value
            and region.get("status") == RegionStatus.SKIP.value
        ),
        "final_unknown_review_regions": sum(
            1 for region in regions
            if region.get("type") == RegionType.UNKNOWN.value
            and region.get("status") == RegionStatus.REVIEW.value
        ),
        "final_unknown_auto_regions": sum(
            1 for region in regions
            if region.get("type") == RegionType.UNKNOWN.value
            and region.get("status") == RegionStatus.AUTO.value
        ),
    }


def main() -> None:
    print("==================================================================")
    print("RUNNING REAL CHAPTER 1 PRODUCTION E2E AUDIT")
    print("==================================================================")

    assert SOURCE_CHAPTER.exists(), f"Source chapter directory not found: {SOURCE_CHAPTER}"
    assert SOURCE_CHAPTER.resolve() != OUTPUT_DIR.resolve(), "Output directory must not equal source directory!"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Snapshot source file integrity before E2E
    pre_snapshots = snapshot_directory(SOURCE_CHAPTER)
    print(f"[OK] Pre-E2E snapshot recorded for {len(pre_snapshots)} source files.")

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

    # 3. Snapshot source file integrity after E2E and verify 0 mutations
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

    # 4. Extract E2E metrics from pipeline result and analysis metadata
    regions_json_path = OUTPUT_DIR / "analysis" / "regions.json"
    summary_json_path = OUTPUT_DIR / "analysis" / "summary.json"

    with open(regions_json_path, "r", encoding="utf-8") as f:
        regions_data = json.load(f)

    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    raw_regions = regions_data["regions"]

    final_state_metrics = summarize_final_region_states(raw_regions)

    final_auto = sum(1 for r in raw_regions if r.get("status") == "auto")
    final_review = sum(1 for r in raw_regions if r.get("status") == "review")
    final_skip = sum(1 for r in raw_regions if r.get("status") == "skip")

    verifier_calls = sum(
        1 for r in raw_regions
        if isinstance(r.get("metadata"), dict) and r["metadata"].get("ocr_verdict", {}).get("second_pass_invoked")
    )

    qwen_repair_candidates = sum(
        1 for r in raw_regions
        if isinstance(r.get("metadata"), dict) and r["metadata"].get("ocr_verdict", {}).get("needs_repair")
    )

    actual_qwen_calls = sum(
        1 for r in raw_regions
        if isinstance(r.get("metadata"), dict) and r["metadata"].get("repaired")
    )

    metrics = {
        "source_page_count": len(pipeline_result.pages),
        "output_page_count": len(pipeline_result.exported_page_paths),
        **final_state_metrics,
        "verifier_attempted_calls": verifier_calls,
        "qwen_repair_candidates": qwen_repair_candidates,
        "actual_qwen_model_calls": actual_qwen_calls,
        "final_auto_regions": final_auto,
        "final_review_regions": final_review,
        "final_skip_regions": final_skip,
        "text_block_count": regions_data["text_blocks_count"],
        "translated_blocks_count": summary_data["translated_blocks_count"],
        "successfully_inpainted_blocks_count": summary_data["inpainted_blocks_count"],
        "review_inpaint_blocks_count": summary_data["review_inpaint_blocks_count"],
        "actually_rendered_blocks_count": summary_data["rendered_blocks_count"],
        "overflow_blocks_count": summary_data["overflow_blocks_count"],
        "elapsed_seconds": round(elapsed, 2),
    }

    assert metrics["output_page_count"] == metrics["source_page_count"], "Output page count mismatch!"

    audit_json_path = OUTPUT_DIR / "e2e_audit_metrics.json"
    audit_json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n==================================================================")
    print("REAL CHAPTER 1 E2E METRICS SUMMARY:")
    print("==================================================================")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\n[OK] Metrics saved to {audit_json_path}")


if __name__ == "__main__":
    main()
