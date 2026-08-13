"""Script to execute controlled real chapter end-to-end production pipeline."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from application.chapter_analyzer import ChapterAnalyzer, ProductionPipelineResult
from providers.detector.ctd import ComicTextDetector
from providers.ocr.rapid_onnx import RapidONNXOCR
from providers.translation.hy_mt2_gguf_translation import HyMT2GGUFTranslationProvider


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    source_chapter = root_dir / "test_data" / "chapter_test"
    output_dir = root_dir / "e2e_output" / "real_chapter_test"

    print("==================================================================")
    print("STARTING PRODUCTION END-TO-END CHAPTER PIPELINE")
    print(f"Source Chapter: {source_chapter}")
    print(f"Output Path:    {output_dir}")
    print("==================================================================")

    if not source_chapter.exists():
        print(f"ERROR: Source chapter directory not found at {source_chapter}")
        sys.exit(1)

    analyzer = ChapterAnalyzer()
    detector = ComicTextDetector()
    ocr = RapidONNXOCR()
    translator = HyMT2GGUFTranslationProvider()

    def progress(evt) -> None:
        print(f"[{evt.stage}] {evt.percent * 100:.1f}% - {evt.message}")

    start_time = time.time()
    try:
        result: ProductionPipelineResult = analyzer.process_chapter(
            chapter_path=source_chapter,
            output_path=output_dir,
            detector=detector,
            primary_ocr=ocr,
            translator=translator,
            progress_callback=progress,
        )
    except Exception as e:
        print(f"Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start_time

    print("\n==================================================================")
    print("PRODUCTION PIPELINE COMPLETED SUCCESSFULLY")
    print("==================================================================")
    print(f"Page Count:               {result.page_count}")
    print(f"Detected Regions:         {result.detected_region_count}")
    print(f"Translated Regions:       {result.translated_region_count}")
    print(f"Skipped Regions:          {result.skipped_region_count}")
    print(f"Review Required Regions:  {result.review_required_count}")
    print(f"Exported Pages:           {len(result.exported_page_paths)}")
    print(f"Total Elapsed Time:       {elapsed:.2f}s")
    print(f"OCR Time:                 {result.ocr_elapsed_time:.2f}s")
    print(f"Translation Time:         {result.translation_elapsed_time:.2f}s")
    print(f"Inpaint/Render Time:      {result.inpainting_rendering_elapsed_time:.2f}s")
    print("==================================================================")

    for page_path in result.exported_page_paths:
        if page_path.exists():
            print(f"Verified Output Page: {page_path.name} ({page_path.stat().st_size} bytes)")
        else:
            print(f"ERROR: Missing output page {page_path}")
            sys.exit(1)


if __name__ == "__main__":
    main()
