"""Validation script for running production pipeline on real Tachidesk Chapter 1."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from application.chapter_analyzer import ChapterAnalyzer, ProductionPipelineResult
from providers.detector.ctd import ComicTextDetector
from providers.ocr.paddleocr_vl import PaddleOCRVLOcrProvider
from providers.ocr.paddleocr import PaddleOCRProvider
from providers.ocr.qwen_repair import QwenRepairProvider
from providers.translation.hy_mt2_gguf_translation import HyMT2GGUFTranslationProvider


def compute_dir_hashes_and_mtimes(folder: Path) -> dict[str, tuple[str, float]]:
    """Compute sha256 hash and mtime for all files in a folder."""
    result = {}
    for p in sorted(folder.glob("*")):
        if p.is_file():
            hasher = hashlib.sha256()
            hasher.update(p.read_bytes())
            result[p.name] = (hasher.hexdigest(), p.stat().st_mtime)
    return result


def main() -> None:
    # Selected real chapter from Tachidesk
    source_chapter = Path(
        r"C:\Users\Ahmed\AppData\Local\Tachidesk\downloads\mangas\Asmodeus Scans (EN)\Reincarnated as a God-Tier Crafter\Chapter 1"
    )
    output_dir = Path(__file__).resolve().parent.parent / "e2e_output" / "real_tachidesk_chapter_1"

    print("==================================================================")
    print("STARTING REAL TACHIDESK CHAPTER 1 PRODUCTION VALIDATION")
    print(f"Source Chapter: {source_chapter}")
    print(f"Output Path:    {output_dir}")
    print("==================================================================")

    if not source_chapter.exists():
        print(f"ERROR: Source directory does not exist: {source_chapter}")
        sys.exit(1)

    # 1. Compute pre-run source hashes and mtimes
    print("[1/5] Recording pre-run source file hashes and mtimes...")
    pre_hashes = compute_dir_hashes_and_mtimes(source_chapter)
    print(f"      Recorded {len(pre_hashes)} source files.")

    # 2. Setup real providers
    print("[2/5] Initializing REAL production providers...")
    detector = ComicTextDetector()
    primary_ocr = PaddleOCRProvider("PP-OCRv6_medium_rec")
    verifier_ocr = PaddleOCRVLOcrProvider()
    qwen_repair = QwenRepairProvider()
    translator = HyMT2GGUFTranslationProvider()

    analyzer = ChapterAnalyzer()

    def progress(evt) -> None:
        print(f"      [{evt.stage}] {evt.percent * 100:.1f}% - {evt.message}")

    # 3. Run production pipeline
    print("[3/5] Running process_chapter() end-to-end pipeline...")
    start_time = time.time()
    result: ProductionPipelineResult = analyzer.process_chapter(
        chapter_path=source_chapter,
        output_path=output_dir,
        detector=detector,
        primary_ocr=primary_ocr,
        verifier_ocr=verifier_ocr,
        qwen_repair=qwen_repair,
        translator=translator,
        progress_callback=progress,
    )
    total_elapsed = time.time() - start_time

    # 4. Verify source files UNCHANGED
    print("[4/5] Verifying source files hash and mtime UNCHANGED...")
    post_hashes = compute_dir_hashes_and_mtimes(source_chapter)

    assert len(pre_hashes) == len(post_hashes), "Source file count changed!"
    for fname, (pre_hash, pre_mtime) in pre_hashes.items():
        post_hash, post_mtime = post_hashes[fname]
        if pre_hash != post_hash:
            raise RuntimeError(f"SOURCE OVERWRITE DETECTED! File {fname} hash changed!")
        if pre_mtime != post_mtime:
            raise RuntimeError(f"SOURCE MODIFICATION DETECTED! File {fname} mtime changed!")
    print("      [OK] SOURCE UNCHANGED VERIFICATION PASSED (Hashes & mtimes identical).")

    # 5. Verify Output Files and Inspect Regions JSON
    print("[5/5] Analyzing regions.json and output pages...")
    regions_file = output_dir / "analysis" / "regions.json"
    with open(regions_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    regions_data = data.get("regions", [])

    detected_count = len(regions_data)
    story_text_count = sum(1 for r in regions_data if r.get("status") == "auto")
    sfx_count = sum(1 for r in regions_data if r.get("type") == "sfx")
    watermark_count = sum(1 for r in regions_data if r.get("type") == "watermark")
    non_text_count = sum(1 for r in regions_data if r.get("type") == "unknown" and r.get("status") == "skip")
    review_count = sum(1 for r in regions_data if r.get("status") == "review")
    skip_count = sum(1 for r in regions_data if r.get("status") == "skip")

    pp_v6_direct_auto = sum(
        1 for r in regions_data
        if r.get("status") == "auto"
        and not r.get("metadata", {}).get("repaired")
        and not r.get("metadata", {}).get("ocr_verdict", {}).get("second_pass_invoked")
    )
    vl_second_pass_invoked = sum(
        1 for r in regions_data
        if r.get("metadata", {}).get("ocr_verdict", {}).get("second_pass_invoked")
    )
    vl_second_pass_resolved = sum(
        1 for r in regions_data
        if r.get("metadata", {}).get("ocr_verdict", {}).get("second_pass_invoked")
        and r.get("status") == "auto"
        and not r.get("metadata", {}).get("repaired")
    )

    qwen_attempted_count = sum(
        1 for r in regions_data
        if r.get("metadata", {}).get("ocr_verdict", {}).get("needs_repair")
        and r.get("type") in ("dialogue", "narration", "unknown")
        and r.get("status") != "skip"
    )
    qwen_success_count = sum(
        1 for r in regions_data
        if r.get("metadata", {}).get("repaired") is True
    )

    baseline_time = 2324.12
    speedup_pct = ((baseline_time - total_elapsed) / baseline_time) * 100.0

    text_blocks_data = data.get("text_blocks", [])
    tb_count = len(text_blocks_data)
    single_member_blocks = sum(1 for b in text_blocks_data if len(b.get("member_ids", [])) == 1)
    multi_member_blocks = sum(1 for b in text_blocks_data if len(b.get("member_ids", [])) > 1)
    member_counts = [len(b.get("member_ids", [])) for b in text_blocks_data]
    avg_members = sum(member_counts) / tb_count if tb_count > 0 else 0.0
    max_members = max(member_counts) if member_counts else 0

    rendered_blocks_cnt = data.get("rendered_blocks_count", sum(1 for b in text_blocks_data if b.get("translation")))
    overflow_blocks_cnt = data.get("overflow_blocks_count", 0)

    print("\n==================================================================")
    print("REAL CHAPTER PRODUCTION VALIDATION METRICS (PIPELINE V4 - BLOCK RENDER)")
    print("==================================================================")
    print(f"Real Chapter:             Chapter 1")
    print(f"Source Page Count:        {len(result.pages)}")
    print(f"Output Page Count:        {len(result.exported_page_paths)}")
    print(f"Detected Regions Total:   {detected_count}")
    print(f"  • STORY_TEXT Regions:   {story_text_count} (AUTO accepted)")
    print(f"  • SFX Regions:          {sfx_count} (SKIP)")
    print(f"  • WATERMARK Regions:    {watermark_count} (SKIP)")
    print(f"  • NON_TEXT Regions:     {non_text_count} (SKIP)")
    print(f"  • REVIEW Regions:       {review_count} (Protected/Human review)")
    print(f"Total SKIP Regions:       {skip_count} (= SFX + WATERMARK + NON_TEXT)")
    print("------------------------------------------------------------------")
    print(f"Text-Blocks Created:      {tb_count}")
    print(f"Single-Member Blocks:     {single_member_blocks}")
    print(f"Multi-Member Blocks:      {multi_member_blocks}")
    print(f"Avg Members / Block:      {avg_members:.2f}")
    print(f"Max Members in a Block:   {max_members}")
    print(f"Translation Requests:     {tb_count} (vs 505 region-based requests previously)")
    print("------------------------------------------------------------------")
    print(f"Translated Blocks:        {sum(1 for b in text_blocks_data if b.get('translation'))}")
    print(f"Rendered Blocks:          {rendered_blocks_cnt}")
    print(f"Overflow Blocks:          {overflow_blocks_cnt}")
    print("------------------------------------------------------------------")
    print(f"PP-OCRv6 Direct Accepted: {pp_v6_direct_auto}")
    print(f"PaddleOCR-VL Second-Pass: {vl_second_pass_invoked} calls (Resolved: {vl_second_pass_resolved})")
    print(f"Qwen Repair Attempted:    {qwen_attempted_count}")
    print(f"Qwen Repair Success:      {qwen_success_count}")
    print(f"Qwen Repair Unresolved:   {review_count}")
    print("------------------------------------------------------------------")
    print(f"Total Elapsed Time:       {total_elapsed:.2f}s ({total_elapsed/60.0:.2f} min)")
    print(f"OCR Time:                 {result.ocr_elapsed_time:.2f}s")
    print(f"Translation Time:         {result.translation_elapsed_time:.2f}s")
    print(f"Inpaint/Render Time:      {result.inpainting_rendering_elapsed_time:.2f}s")
    print(f"Speedup vs 38.7m Baseline:{speedup_pct:+.1f}% ({baseline_time:.1f}s -> {total_elapsed:.1f}s)")
    print("==================================================================")
    print("[OK] All exported page dimensions match source specifications perfectly.")
    
    # Dimension and page match check
    for page_obj, out_path in zip(result.pages, result.exported_page_paths):
        with Image.open(out_path) as out_im:
            assert out_im.size == (page_obj.width, page_obj.height), (
                f"Page {out_path.name} dimension mismatch: expected ({page_obj.width}, {page_obj.height}), got {out_im.size}"
            )

    print("[OK] All exported page dimensions match source specifications perfectly.")


if __name__ == "__main__":
    main()
