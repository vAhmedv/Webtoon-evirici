"""Bölüm analiz ve uçtan uca üretim (production) pipeline hizmeti.

Core pipeline'ı UI'dan bağımsız olarak orchestrate eder.
"""

from __future__ import annotations

import gc
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from loguru import logger

from application.cancellation import CancellationToken, CancelledError
from application.progress import ProgressEvent
from core.config import Config, load_config
from core.coordinate.global_coords import GlobalCoordinateSystem
from core.coordinate.sliding_window import generate_windows_for_pages
from core.detection import Detection, DetectionCache, Region, RegionStatus, RegionType
from core.detection.cache import CACHE_PATH
from core.detection.coordinate import (
    window_bbox_to_global,
    global_bbox_to_window,
    window_polygon_to_global,
    global_polygon_to_window,
)
from core.detection.merge import merge_duplicates
from core.detection.region_validity import evaluate_region_validity
from core.detection.repair_eligibility import evaluate_repair_eligibility
from core.imaging.inpainter import Inpainter
from core.imaging.region_cropper import RegionCropper
from core.imaging.renderer import TextRenderer
from core.imaging.window_extractor import extract_window_image, WindowImage
from core.io.input_loader import load_chapter
from core.io.output_exporter import export_chapter_pages
from core.models import Page, Window
from core.serialization.serializer import region_to_dict
from core.visualization.draw import draw_detections, draw_regions
from providers.detector.base import DetectorProvider
from providers.ocr.agreement import decide_ocr_agreement, should_run_verifier
from providers.ocr.base import OCRProvider
from providers.ocr.repair import OCRRepairInput, OCRRepairProvider
from providers.translation.base import TranslationInput, TranslationItem, TranslationProvider


ProgressCallback = Callable[[ProgressEvent], None]


class AnalysisResult:
    """Bölüm analizi sonucu.

    Attributes:
        pages: Yüklenen sayfalar.
        windows: Üretilen window'lar.
        regions: Canonical region listesi.
        auto_count: AUTO durumundaki region sayısı.
        review_count: REVIEW durumundaki region sayısı.
        skip_count: SKIP durumundaki region sayısı.
        elapsed_time: Analiz süresi (saniye).
        visualization_paths: Her window için görselleştirme yolları.
        warnings: Oluşan uyarılar.
        ocr_elapsed_time: OCR süresi (saniye).
    """

    def __init__(
        self,
        pages: list[Page],
        windows: list[Window],
        regions: list[Region],
        elapsed_time: float,
        visualization_paths: list[Path] | None = None,
        warnings: list[str] | None = None,
        ocr_elapsed_time: float = 0.0,
        stage_timings: dict[str, float] | None = None,
    ) -> None:
        self.pages = pages
        self.windows = windows
        self.regions = regions
        self.auto_count = sum(1 for r in regions if r.status == RegionStatus.AUTO)
        self.review_count = sum(1 for r in regions if r.status == RegionStatus.REVIEW)
        self.skip_count = sum(1 for r in regions if r.status == RegionStatus.SKIP)
        self.elapsed_time = elapsed_time
        self.visualization_paths = visualization_paths or []
        self.warnings = warnings or []
        self.ocr_elapsed_time = ocr_elapsed_time
        self.stage_timings = stage_timings or {}


class ProductionPipelineResult:
    """Uçtan uca üretim (production) pipeline sonucu."""

    def __init__(
        self,
        source_chapter: Path,
        output_directory: Path,
        pages: list[Page],
        windows: list[Window],
        regions: list[Region],
        exported_page_paths: list[Path],
        elapsed_time: float,
        ocr_elapsed_time: float = 0.0,
        translation_elapsed_time: float = 0.0,
        inpainting_rendering_elapsed_time: float = 0.0,
        warnings: list[str] | None = None,
        stage_timings: dict[str, float] | None = None,
    ) -> None:
        self.source_chapter = source_chapter
        self.output_directory = output_directory
        self.pages = pages
        self.windows = windows
        self.regions = regions
        self.exported_page_paths = exported_page_paths
        self.page_count = len(pages)
        self.detected_region_count = len(regions)
        self.translated_region_count = sum(1 for r in regions if r.translation is not None)
        self.skipped_region_count = sum(1 for r in regions if r.status == RegionStatus.SKIP)
        self.review_required_count = sum(1 for r in regions if r.status == RegionStatus.REVIEW)
        self.elapsed_time = elapsed_time
        self.ocr_elapsed_time = ocr_elapsed_time
        self.translation_elapsed_time = translation_elapsed_time
        self.inpainting_rendering_elapsed_time = inpainting_rendering_elapsed_time
        self.warnings = warnings or []
        self.stage_timings = stage_timings or {}


class ChapterAnalyzer:
    """Bölüm analiz ve üretim pipeline hizmeti.

    Pipeline'ı UI'dan bağımsız olarak çalıştırır.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config if config is not None else load_config()
        self._cache = DetectionCache(
            cache_path=CACHE_PATH,
            max_entries=self.config.detection.max_cache_entries,
            enabled=self.config.detection.enabled,
        )

    def process_chapter(
        self,
        chapter_path: str | Path,
        output_path: str | Path,
        detector: DetectorProvider,
        primary_ocr: OCRProvider | None = None,
        verifier_ocr: OCRProvider | None = None,
        qwen_repair: OCRRepairProvider | None = None,
        translator: TranslationProvider | None = None,
        window_height: int | None = None,
        window_overlap: int | None = None,
        min_confidence: float | None = None,
        progress_callback: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ProductionPipelineResult:
        """Uçtan uca production pipeline'ı çalıştırır.

        Kaynak görselleri korur, tespit, OCR, visual repair, çeviri,
        inpainting ve Türkçe metin rendering uygulayarak çıktıyı ayrı
        bir dizine aktarır.
        """
        start_time = time.time()
        t_total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}
        chapter_path = Path(chapter_path).resolve()
        output_path = Path(output_path).resolve()
        warnings: list[str] = []

        # Phase 3 Guard: Source safety check
        if (
            chapter_path == output_path
            or chapter_path in output_path.parents
            or output_path in chapter_path.parents
        ):
            raise ValueError(
                f"SOURCE OVERWRITE GUARD: Output path '{output_path}' conflicts with "
                f"source '{chapter_path}'. Output must not be identical to, inside, "
                f"or a parent of the source directory."
            )

        cfg = self.config
        if window_height is not None:
            cfg = _replace(cfg, window_height=window_height)
        if window_overlap is not None:
            cfg = _replace(cfg, window_overlap=window_overlap)
        conf = min_confidence if min_confidence is not None else cfg.min_confidence

        def _progress(stage: str, current: int = 0, total: int = 0, message: str = "") -> None:
            if progress_callback is None:
                return
            pct = 0.0
            if total > 0:
                pct = max(pct, current / total)
            progress_callback(ProgressEvent(stage=stage, current=current, total=total, message=message, percent=pct))

        # 1. Load Chapter
        t_load_start = time.perf_counter()
        _progress("Loading chapter", message=str(chapter_path))
        pages = load_chapter(chapter_path, cfg, allow_non_uniform_widths=True)
        _progress("Loading chapter", current=1, total=1, message=f"{len(pages)} pages loaded")
        stage_timings["load_chapter"] = round(time.perf_counter() - t_load_start, 3)

        if cancellation_token and cancellation_token.is_cancelled:
            raise CancelledError()

        # 2. Global Coordinate System
        coords = GlobalCoordinateSystem(tuple(pages))

        # 3. Generate Windows
        windows = generate_windows_for_pages(
            pages,
            window_height=cfg.window_height,
            overlap=cfg.window_overlap,
        )

        # 4. Detection Stage
        t_det_start = time.perf_counter()
        _progress("Loading detector")
        if hasattr(detector, "confidence_threshold"):
            detector.confidence_threshold = conf

        all_detections: list[Detection] = []
        try:
            detector.load()
            model_id, model_mtime = _get_model_identity(detector)
            self._cache.load()

            # Batch sliding window detection (batch_size=12)
            batch_size = 12
            for b_start in range(0, len(windows), batch_size):
                if cancellation_token and cancellation_token.is_cancelled:
                    raise CancelledError()

                chunk = windows[b_start : b_start + batch_size]
                uncached_jobs: list[tuple[int, Any, Window, bytes, str]] = []  # (chunk_idx, w_img, window, bytes, hash)

                for c_idx, window in enumerate(chunk):
                    w_img = extract_window_image(tuple(pages), window, coords)
                    img_bytes = _image_to_bytes(w_img.image)
                    p_hash = DetectionCache.compute_hash(img_bytes)
                    cached = self._cache.get(p_hash, model_id, model_mtime)
                    if cached is not None:
                        all_detections.extend(cached)
                    else:
                        uncached_jobs.append((c_idx, w_img, window, img_bytes, p_hash))

                if uncached_jobs:
                    items_to_detect = [(job[1].image, job[2].id) for job in uncached_jobs]
                    if hasattr(detector, "detect_batch"):
                        batch_results = detector.detect_batch(items_to_detect)
                    else:
                        batch_results = [detector.detect(img, wid) for img, wid in items_to_detect]

                    for job, detections in zip(uncached_jobs, batch_results):
                        _, _, window, img_bytes, p_hash = job
                        global_detections = []
                        for det in detections:
                            global_bbox = window_bbox_to_global(det.bbox, window.y_start)
                            metadata = _offset_geometry_metadata(det.metadata, window.y_start)
                            global_det = Detection(
                                bbox=global_bbox,
                                confidence=det.confidence,
                                type=det.type,
                                source_window_id=det.source_window_id,
                                mask=det.mask,
                                metadata=metadata,
                            )
                            global_detections.append(global_det)
                        self._cache.put(p_hash, model_id, model_mtime, global_detections)
                        all_detections.extend(global_detections)

                processed_so_far = min(b_start + len(chunk), len(windows))
                _progress(
                    "Sliding window detection",
                    current=processed_so_far,
                    total=len(windows),
                    message=f"Window {processed_so_far}/{len(windows)}",
                )

            self._cache.save()
        finally:
            detector.unload()

        # Merge duplicates
        regions = merge_duplicates(all_detections, min_confidence=conf)
        stage_timings["detection"] = round(time.perf_counter() - t_det_start, 3)

        if cancellation_token and cancellation_token.is_cancelled:
            raise CancelledError()

        # 5. Dual OCR Stage with Selective Second-Pass & Region Classification
        ocr_start = time.time()
        t_ocr_start = time.perf_counter()
        total_ocr_boxes = 0
        gated_verified_boxes = 0
        gated_passed_boxes = 0
        repair_candidates: list[tuple[Region, OCRRepairInput, Any]] = []

        if primary_ocr is not None:
            _progress("Loading Primary OCR")
            cropper = RegionCropper(pages, coords, padding=20)
            ocr_regions: list[Region] = []
            try:
                primary_ocr.load()
                if verifier_ocr is not None:
                    try:
                        verifier_ocr.load()
                    except Exception as e:
                        warnings.append(f"Verifier OCR load failed: {e}")
                        verifier_ocr = None

                batch_size = 32
                total_regions = len(regions)

                for chunk_start in range(0, total_regions, batch_size):
                    if cancellation_token and cancellation_token.is_cancelled:
                        raise CancelledError()

                    chunk_regions = regions[chunk_start : chunk_start + batch_size]
                    processed_so_far = min(chunk_start + len(chunk_regions), total_regions)
                    _progress(
                        "OCR Recognition",
                        current=processed_so_far,
                        total=total_regions,
                        message=f"OCR {processed_so_far}/{total_regions}",
                    )

                    # 5a. Early Classification & Filtering
                    active_items: list[tuple[int, Region, Any]] = []
                    chunk_results: list[Region | None] = [None] * len(chunk_regions)

                    for rel_idx, region in enumerate(chunk_regions):
                        bbox = region.global_bbox
                        if region.type in (RegionType.SFX, RegionType.WATERMARK) or bbox.height < 10 or bbox.width < 10:
                            skipped_region = _replace_region(
                                region,
                                status=RegionStatus.SKIP,
                                review_reason="sfx_or_non_text_skip",
                            )
                            chunk_results[rel_idx] = skipped_region
                        else:
                            crop = cropper.crop_region(region, adaptive_padding=True)
                            active_items.append((rel_idx, region, crop))

                    if not active_items:
                        for item in chunk_results:
                            if item is not None:
                                ocr_regions.append(item)
                        continue

                    total_ocr_boxes += len(active_items)

                    # 5b. Primary OCR Batch Recognition
                    p_crops = [item[2].to_pil() for item in active_items]
                    p_bboxes = [item[1].global_bbox for item in active_items]
                    primary_results = primary_ocr.recognize_batch(p_crops, p_bboxes)

                    # 5c. Validity Check, Geometric Recovery & Gated Verifier Evaluation
                    verifier_queue: list[tuple[int, Region, Any, OCRResult, dict[str, object], OCRVerdict]] = []
                    processed_items: dict[int, tuple[Region, OCRResult, dict[str, object], OCRVerdict, OCRResult | None, Any]] = {}

                    for (rel_idx, region, crop), p_res in zip(active_items, primary_results):
                        validity = evaluate_region_validity(region, p_res.text)
                        recovery_metadata: dict[str, object] = {}
                        if validity.recovered_bbox is not None:
                            original_bbox = region.global_bbox
                            region = _replace_region(region, global_bbox=validity.recovered_bbox)
                            crop = cropper.crop_region(region, adaptive_padding=True)
                            p_res = primary_ocr.recognize(crop.to_pil(), region_bbox=region.global_bbox)
                            validity = evaluate_region_validity(region, p_res.text)
                            recovery_metadata = {
                                "geometry_recovered": True,
                                "original_bbox": list(original_bbox.to_tuple()),
                                "recovered_bbox": list(region.global_bbox.to_tuple()),
                            }

                        validity_metadata = {
                            "valid": validity.is_valid,
                            "reason": validity.reason,
                            **validity.evidence,
                            **recovery_metadata,
                        }

                        if not validity.is_valid:
                            skipped_region = _replace_region(
                                region,
                                text=p_res.text or "",
                                ocr_confidence=p_res.confidence,
                                status=RegionStatus.SKIP,
                                review_reason=validity.reason,
                                metadata={
                                    **region.metadata,
                                    "region_validity": validity_metadata,
                                    "ocr_verdict": {
                                        "source": "primary",
                                        "requires_review": False,
                                        "needs_repair": False,
                                        "reason": validity.reason,
                                        "second_pass_invoked": False,
                                    },
                                },
                            )
                            chunk_results[rel_idx] = skipped_region
                            continue

                        single_verdict = decide_ocr_agreement(p_res, verifier=None)
                        needs_verifier, _ = should_run_verifier(p_res, region, min_confidence=0.85)

                        if (single_verdict.requires_review or needs_verifier) and verifier_ocr is not None:
                            verifier_queue.append((rel_idx, region, crop, p_res, validity_metadata, single_verdict))
                        else:
                            processed_items[rel_idx] = (region, p_res, validity_metadata, single_verdict, None, crop)

                    gated_verified_boxes += len(verifier_queue)
                    gated_passed_boxes += len(active_items) - len(verifier_queue)

                    # 5d. Secondary OCR Batch Execution (PaddleOCR-VL GPU Batch with In-GPU Tensor Cropping)
                    if verifier_queue and verifier_ocr is not None:
                        from core.system.adaptive_batcher import get_batch_config
                        v_crops = [item[2] for item in verifier_queue]
                        v_bboxes = [item[1].global_bbox for item in verifier_queue]
                        v_batch_size = get_batch_config().ocr_vl_batch
                        verifier_results = verifier_ocr.recognize_batch(v_crops, v_bboxes, batch_size=v_batch_size)

                        for item, v_res in zip(verifier_queue, verifier_results):
                            rel_idx, region, crop, p_res, validity_metadata, single_verdict = item
                            final_verdict = decide_ocr_agreement(p_res, v_res)
                            processed_items[rel_idx] = (region, p_res, validity_metadata, final_verdict, v_res, crop)

                    # 5e. Form Final Regions & Repair Eligibility
                    for rel_idx, (region, p_res, validity_metadata, verdict, v_res, crop) in processed_items.items():
                        if region.status == RegionStatus.SKIP:
                            status = RegionStatus.SKIP
                        else:
                            status = RegionStatus.REVIEW if verdict.requires_review else RegionStatus.AUTO

                        accepted = verdict.accepted_text or verdict.provisional_text or p_res.text or ""
                        has_text_content = bool(accepted and accepted.strip() and any(c.isalnum() for c in accepted))
                        if region.type == RegionType.UNKNOWN and not has_text_content:
                            status = RegionStatus.SKIP
                            verdict_reason = "unknown_non_text_skip"
                        else:
                            verdict_reason = verdict.reason

                        updated_region = _replace_region(
                            region,
                            text=accepted,
                            ocr_confidence=p_res.confidence,
                            status=status,
                            review_reason=verdict_reason,
                            metadata={
                                **region.metadata,
                                "region_validity": validity_metadata,
                                "ocr_verdict": {
                                    "source": verdict.source,
                                    "requires_review": verdict.requires_review,
                                    "needs_repair": verdict.needs_repair,
                                    "reason": verdict_reason,
                                    "second_pass_invoked": v_res is not None,
                                },
                            },
                        )

                        repair_eligibility = evaluate_repair_eligibility(updated_region, verdict)
                        eligibility_metadata = {
                            "eligible": repair_eligibility.eligible,
                            "reason": repair_eligibility.reason,
                            **repair_eligibility.evidence,
                        }
                        updated_region = _replace_region(
                            updated_region,
                            metadata={
                                **updated_region.metadata,
                                "repair_eligibility": eligibility_metadata,
                            },
                        )

                        if repair_eligibility.eligible and qwen_repair is not None:
                            repair_inp = OCRRepairInput(
                                region_id=region.id,
                                primary_text=p_res.text or "",
                                primary_confidence=p_res.confidence,
                                verifier_text=v_res.text if v_res else None,
                                verifier_confidence=v_res.confidence if v_res else None,
                                agreement_verdict=verdict.reason,
                            )
                            repair_candidates.append((updated_region, repair_inp, crop.to_pil()))

                        chunk_results[rel_idx] = updated_region

                    for item in chunk_results:
                        if item is not None:
                            ocr_regions.append(item)

                regions = ocr_regions
            finally:
                cropper.clear_gpu_cache()
                primary_ocr.unload()
                if verifier_ocr is not None:
                    verifier_ocr.unload()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()

        ocr_elapsed = time.time() - ocr_start
        stage_timings["ocr"] = round(time.perf_counter() - t_ocr_start, 3)

        # 6. Visual OCR Repair Stage (Sequential VRAM lifecycle)
        if repair_candidates and qwen_repair is not None:
            _progress("Loading Visual OCR Repair Model")
            try:
                qwen_repair.load()
            except Exception as e:
                logger.warning(f"Visual OCR repair model load failed: {e}")
                warnings.append(f"Visual OCR repair model load skipped: {e}")
                qwen_repair = None

            if qwen_repair is not None:
                try:
                    repaired_regions: list[Region] = []
                    for region, repair_inp, crop_img in repair_candidates:
                        if cancellation_token and cancellation_token.is_cancelled:
                            raise CancelledError()
                        try:
                            rep_res = qwen_repair.repair(repair_inp, crop_img)
                            region = _replace_region(
                                region,
                                metadata={
                                    **region.metadata,
                                    "qwen_repair": dict(rep_res.metadata),
                                },
                            )
                            if rep_res.repaired_text and not rep_res.unresolved:
                                region = _replace_region(
                                    region,
                                    text=rep_res.repaired_text,
                                    status=RegionStatus.AUTO,
                                    metadata={**region.metadata, "repaired": True},
                                )
                        except Exception as e:
                            warnings.append(f"Visual repair failed for region {region.id}: {e}")
                        repaired_regions.append(region)

                    # Update main region list
                    repaired_map = {r.id: r for r in repaired_regions}
                    regions = [repaired_map.get(r.id, r) for r in regions]
                finally:
                    qwen_repair.unload()
                    # Clear VRAM cache
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass

        # 7. Multi-Feature Classification & Generic Watermark Filtering
        from core.detection.classification import classify_regions
        from core.detection.text_block import group_text_blocks, TextBlock
        from core.detection.translation_eligibility import evaluate_translation_eligibility

        regions = classify_regions(regions, coords)

        # 8. Text-Block Grouping (Translation Units)
        text_blocks = group_text_blocks(regions, coords)
        translation_eligibility = {
            block.id: evaluate_translation_eligibility(block)
            for block in text_blocks
        }
        translation_eligible_blocks = [
            block for block in text_blocks
            if translation_eligibility[block.id].eligible
        ]
        block_map: dict[int, TextBlock] = {}
        region_to_block: dict[int, int] = {}
        for b in text_blocks:
            block_map[b.id] = b
            for m_id in b.member_ids:
                region_to_block[m_id] = b.id

        # Update region metadata with text block references
        regions_with_block: list[Region] = []
        for r in regions:
            if r.id in region_to_block:
                b_id = region_to_block[r.id]
                b_obj = block_map[b_id]
                meta = dict(r.metadata)
                meta["text_block"] = {
                    "block_id": b_id,
                    "member_ids": list(b_obj.member_ids),
                    "source_text": b_obj.source_text,
                }
                r = _replace_region(r, metadata=meta)
            regions_with_block.append(r)
        regions = regions_with_block

        # 9. Block-Level Translation Stage (Hy-MT2 GGUF)
        trans_start = time.time()
        t_trans_start = time.perf_counter()
        translated_block_pairs: list[tuple[TextBlock, str]] = []

        if translation_eligible_blocks and translator is not None:
            _progress("Loading Translation Model (Hy-MT2)")
            try:
                translator.load()
                items = [
                    TranslationItem(region_id=b.id, source=b.source_text)
                    for b in translation_eligible_blocks
                ]
                trans_inp = TranslationInput(items=items)
                trans_out = translator.translate(trans_inp)

                out_map = {item.region_id: item.translation for item in trans_out.results if item.translation}

                for b in translation_eligible_blocks:
                    if b.id in out_map:
                        translated_block_pairs.append((b, out_map[b.id]))

                updated_regions: list[Region] = []
                for r in regions:
                    b_id = region_to_block.get(r.id)
                    if b_id and b_id in out_map:
                        tr_text = out_map[b_id]
                        meta = dict(r.metadata)
                        if "text_block" in meta:
                            meta["text_block"]["translation"] = tr_text
                        r_updated = _replace_region(r, translation=tr_text, metadata=meta)
                        updated_regions.append(r_updated)
                    else:
                        updated_regions.append(r)
                regions = updated_regions
            finally:
                translator.unload()

        trans_elapsed = time.time() - trans_start
        stage_timings["translation"] = round(time.perf_counter() - t_trans_start, 3)
        translation_failed_count = len(translation_eligible_blocks) - len(translated_block_pairs)
        pre_inpaint_skipped_count = len(text_blocks) - len(translated_block_pairs)

        # 10. Block-Level Inpainting & Rendering Stage
        inp_render_start = time.time()
        t_inp_render_start = time.perf_counter()
        _progress("Inpainting and Rendering Turkish Text Blocks")

        from PIL import Image as PILImage
        canvas_w = pages[0].width
        canvas_h = sum(p.height for p in pages)

        global_canvas = PILImage.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        y_cursor = 0
        for page in pages:
            with PILImage.open(page.path) as p_img:
                p_img_rgb = p_img.convert("RGB")
                if p_img_rgb.width != page.width or p_img_rgb.height != page.height:
                    p_img_rgb = p_img_rgb.resize((page.width, page.height), PILImage.Resampling.LANCZOS)
                global_canvas.paste(p_img_rgb, (0, y_cursor))
                y_cursor += page.height

        # Clean speech bubble text for translated blocks
        inpainter_kwargs: dict[str, Any] = {
            "debug_dir": output_path / "analysis" / "inpainting_debug"
        }
        if cfg.inpainter.model:
            inpainter_kwargs["lama_checkpoint"] = cfg.inpainter.model
        inpainter = Inpainter(**inpainter_kwargs)
        t_inp_start = time.perf_counter()
        try:
            cleaned_canvas = inpainter.inpaint_blocks(global_canvas, [b for b, _ in translated_block_pairs])
        finally:
            # Translation models are already unloaded above; release LaMa before rendering/export.
            inpainter.unload()
        stage_timings["inpainting"] = round(time.perf_counter() - t_inp_start, 3)

        # Flag regions belonging to inpainting review blocks
        if inpainter.review_block_ids:
            updated_regions = []
            for r in regions:
                b_id = region_to_block.get(r.id)
                if b_id in inpainter.review_block_ids:
                    r_updated = _replace_region(
                        r,
                        status=RegionStatus.REVIEW,
                        review_reason="inpaint_boundary_residual_review",
                    )
                    updated_regions.append(r_updated)
                else:
                    updated_regions.append(r)
            regions = updated_regions

        # Render Turkish text into merged block bounding boxes (excluding review blocks)
        t_render_start = time.perf_counter()
        renderer = TextRenderer()
        renderable_pairs = [
            pair for pair in translated_block_pairs
            if pair[0].id in inpainter.processed_block_ids
            and pair[0].id not in inpainter.review_block_ids
        ]
        rendered_canvas, actual_rendered_count, overflow_count = renderer.render_blocks(cleaned_canvas, renderable_pairs)

        inp_render_elapsed = time.time() - inp_render_start

        # 11. Output Export & Analysis Metadata
        _progress("Exporting final output pages")
        exported_page_paths = export_chapter_pages(pages, rendered_canvas, output_path)
        stage_timings["render_and_save"] = round(time.perf_counter() - t_render_start, 3)

        analysis_dir = output_path / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        successful_inpainting_count = len(inpainter.processed_block_ids - inpainter.review_block_ids)
        review_inpainting_count = len(inpainter.review_block_ids)
        render_eligible_count = len(renderable_pairs)

        if len(text_blocks) != pre_inpaint_skipped_count + len(translated_block_pairs):
            raise RuntimeError("TextBlock lifecycle metrics do not reconcile before inpainting")
        if len(translated_block_pairs) != successful_inpainting_count + review_inpainting_count:
            raise RuntimeError("Translated block lifecycle metrics do not reconcile after inpainting")

        regions_json = analysis_dir / "regions.json"
        with open(regions_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pages": len(pages),
                    "windows": len(windows),
                    "text_blocks_count": len(text_blocks),
                    "translation_eligible_blocks_count": len(translation_eligible_blocks),
                    "translated_blocks_count": len(translated_block_pairs),
                    "translation_failed_blocks_count": translation_failed_count,
                    "pre_inpaint_skipped_blocks_count": pre_inpaint_skipped_count,
                    "inpainted_blocks_count": successful_inpainting_count,
                    "inpaint_review_blocks_count": review_inpainting_count,
                    "review_inpaint_blocks_count": review_inpainting_count,
                    "render_eligible_blocks_count": render_eligible_count,
                    "rendered_blocks_count": actual_rendered_count,
                    "overflow_blocks_count": overflow_count,
                    "review_block_ids": sorted(list(inpainter.review_block_ids)),
                    "text_blocks": [
                        {
                            "id": b.id,
                            "member_ids": list(b.member_ids),
                            "source_text": b.source_text,
                            "translation": r.translation if (r := next((r for r in regions if r.id in b.member_ids), None)) else None,
                            "merged_bbox": [b.merged_bbox.x1, b.merged_bbox.y1, b.merged_bbox.x2, b.merged_bbox.y2],
                        }
                        for b in text_blocks
                    ],
                    "regions": [region_to_dict(r) for r in regions],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        elapsed = time.time() - start_time
        stage_timings["total"] = round(time.perf_counter() - t_total_start, 3)

        # Build and log telemetry stage timings banner
        trans_chunk_sz = getattr(translator, "chunk_size", 32)
        total_tr_items = len(translation_eligible_blocks)
        total_tr_chunks = (total_tr_items + trans_chunk_sz - 1) // trans_chunk_sz if total_tr_items > 0 else 0
        trans_fallbacks = getattr(translator, "fallback_count", 0)

        profile_banner = (
            "\n"
            + "=" * 80 + "\n"
            + "                    ⚡ WEBTOON PIPELINE PROFİLE RAPORU ⚡\n"
            + "=" * 80 + "\n"
            + "  Aşama                                    | Süre (sn) | Detay\n"
            + "-" * 80 + "\n"
            + f"  Detection (CTD Batch GPU + Post-Process) | {stage_timings.get('detection', 0.0):>7.2f} s | {len(pages)} Sayfa / {len(windows)} Pencere\n"
            + f"  OCR (PP-OCR CPU + PaddleOCR-VL GPU Gated)| {stage_timings.get('ocr', 0.0):>7.2f} s | Doğrulanan Kutu: {gated_verified_boxes} / Toplam: {total_ocr_boxes} (Gated: {gated_passed_boxes})\n"
            + f"  Translation (Hy-MT2 Chunks: {total_tr_chunks}, Fallback: {trans_fallbacks}) | {stage_timings.get('translation', 0.0):>7.2f} s | {len(translated_block_pairs)} Blok Çevrildi\n"
            + f"  Inpainting (LaMa GPU Batch)              | {stage_timings.get('inpainting', 0.0):>7.2f} s | {successful_inpainting_count} Blok\n"
            + f"  Render & Save                            | {stage_timings.get('render_and_save', 0.0):>7.2f} s | {len(exported_page_paths)} Sayfa Dışa Aktarıldı\n"
            + "-" * 80 + "\n"
            + f"  TOPLAM ÇALIŞMA SÜRESİ                    | {stage_timings.get('total', 0.0):>7.2f} s | 🚀 Hızlandırılmış Mod\n"
            + "=" * 80
        )
        logger.info(profile_banner)

        summary = {
            "chapter": str(chapter_path),
            "output": str(output_path),
            "pages": len(pages),
            "windows": len(windows),
            "regions": len(regions),
            "translated": sum(1 for r in regions if r.translation),
            "skipped": sum(1 for r in regions if r.status == RegionStatus.SKIP),
            "review": sum(1 for r in regions if r.status == RegionStatus.REVIEW),
            "translated_blocks_count": len(translated_block_pairs),
            "text_blocks_count": len(text_blocks),
            "translation_eligible_blocks_count": len(translation_eligible_blocks),
            "translation_failed_blocks_count": translation_failed_count,
            "pre_inpaint_skipped_blocks_count": pre_inpaint_skipped_count,
            "inpainted_blocks_count": successful_inpainting_count,
            "inpaint_review_blocks_count": review_inpainting_count,
            "review_inpaint_blocks_count": review_inpainting_count,
            "render_eligible_blocks_count": render_eligible_count,
            "rendered_blocks_count": actual_rendered_count,
            "overflow_blocks_count": overflow_count,
            "elapsed_time": round(elapsed, 2),
            "stage_timings": stage_timings,
            "warnings": warnings,
        }
        with open(analysis_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        _progress("Completed", current=1, total=1, message="Production pipeline complete")

        return ProductionPipelineResult(
            source_chapter=chapter_path,
            output_directory=output_path,
            pages=pages,
            windows=windows,
            regions=regions,
            exported_page_paths=exported_page_paths,
            elapsed_time=elapsed,
            ocr_elapsed_time=ocr_elapsed,
            translation_elapsed_time=trans_elapsed,
            inpainting_rendering_elapsed_time=inp_render_elapsed,
            warnings=warnings,
            stage_timings=stage_timings,
        )

    def analyze(
        self,
        chapter_path: str | Path,
        output_path: str | Path,
        detector: DetectorProvider,
        window_height: int | None = None,
        window_overlap: int | None = None,
        min_confidence: float | None = None,
        progress_callback: ProgressCallback | None = None,
        cancellation_token: CancellationToken | None = None,
        ocr_provider: OCRProvider | None = None,
    ) -> AnalysisResult:
        """Bölüm analizini çalıştırır (analiz modu)."""
        start_time = time.time()
        t_total_start = time.perf_counter()
        stage_timings: dict[str, float] = {}
        chapter_path = Path(chapter_path)
        output_path = Path(output_path)
        warnings: list[str] = []

        cfg = self.config
        if window_height is not None:
            cfg = _replace(cfg, window_height=window_height)
        if window_overlap is not None:
            cfg = _replace(cfg, window_overlap=window_overlap)
        conf = min_confidence if min_confidence is not None else cfg.min_confidence

        def _progress(stage: str, current: int = 0, total: int = 0, message: str = "") -> None:
            if progress_callback is None:
                return
            pct = 0.0
            if total > 0:
                pct = max(pct, current / total)
            progress_callback(ProgressEvent(stage=stage, current=current, total=total, message=message, percent=pct))

        # 1. Load chapter
        t_load_start = time.perf_counter()
        _progress("Loading chapter", message=str(chapter_path))
        pages = load_chapter(chapter_path, cfg)
        _progress("Loading chapter", current=1, total=1, message=f"{len(pages)} pages loaded")
        stage_timings["load_chapter"] = round(time.perf_counter() - t_load_start, 3)

        if cancellation_token and cancellation_token.is_cancelled:
            raise CancelledError()

        # 2. Global coordinate system
        _progress("Creating coordinate system")
        coords = GlobalCoordinateSystem(tuple(pages))

        # 3. Generate windows
        _progress("Creating windows")
        windows = generate_windows_for_pages(
            pages,
            window_height=cfg.window_height,
            overlap=cfg.window_overlap,
        )
        _progress("Creating windows", current=len(windows), total=len(windows))

        if cancellation_token and cancellation_token.is_cancelled:
            raise CancelledError()

        # 4. Load detector
        t_det_start = time.perf_counter()
        _progress("Loading detector")
        if hasattr(detector, "confidence_threshold"):
            detector.confidence_threshold = conf
        detector.load()

        model_id, model_mtime = _get_model_identity(detector)
        self._cache.load()

        # 5. Detect
        all_detections: list = []
        visualization_dir = output_path / "analysis" / "windows"
        visualization_dir.mkdir(parents=True, exist_ok=True)
        window_visualization_paths: list[Path] = []

        for idx, window in enumerate(windows, start=1):
            if cancellation_token and cancellation_token.is_cancelled:
                raise CancelledError()

            _progress("Detecting", current=idx, total=len(windows), message=f"Window {idx}/{len(windows)}")

            window_image = extract_window_image(tuple(pages), window, coords)

            image_bytes = _image_to_bytes(window_image.image)
            page_hash = DetectionCache.compute_hash(image_bytes)
            cached = self._cache.get(page_hash, model_id, model_mtime)

            if cached is not None:
                global_detections = cached
                detections = [_global_detection_to_window(d, window.y_start) for d in cached]
            else:
                detections = detector.detect(window_image.image, window.id)

                global_detections: list[Detection] = []
                for det in detections:
                    global_bbox = window_bbox_to_global(det.bbox, window.y_start)
                    metadata = _offset_geometry_metadata(det.metadata, window.y_start)
                    global_det = Detection(
                        bbox=global_bbox,
                        confidence=det.confidence,
                        type=det.type,
                        source_window_id=det.source_window_id,
                        mask=det.mask,
                        metadata=metadata,
                    )
                    global_detections.append(global_det)

                self._cache.put(page_hash, model_id, model_mtime, global_detections)

            all_detections.extend(global_detections)

            vis = draw_detections(window_image.image, detections, window_y_start=window.y_start)
            vis_path = visualization_dir / f"window_{window.id:03d}.png"
            vis.save(vis_path)
            window_visualization_paths.append(vis_path)

        self._cache.save()
        detector.unload()

        # 6. Merge duplicates
        _progress("Merging regions")
        regions = merge_duplicates(all_detections, min_confidence=conf)
        stage_timings["detection"] = round(time.perf_counter() - t_det_start, 3)

        regions = [
            _replace_status(reg, RegionStatus.REVIEW) if reg.status == RegionStatus.AUTO and reg.detection_confidence < conf else reg
            for reg in regions
        ]

        if cancellation_token and cancellation_token.is_cancelled:
            raise CancelledError()

        # 7. OCR (opsiyonel)
        ocr_start = 0.0
        ocr_elapsed = 0.0
        if ocr_provider is not None:
            t_ocr_start = time.perf_counter()
            _progress("Loading OCR")
            try:
                ocr_provider.load()
            except Exception as e:
                logger.error(f"OCR provider yüklenemedi: {e}")
                warnings.append(f"OCR load failed: {e}")
                ocr_provider = None

            if ocr_provider is not None:
                cropper = RegionCropper(pages, coords, padding=20)
                ocr_start = time.time()
                ocr_regions: list[Region] = []
                for idx, region in enumerate(regions, start=1):
                    if cancellation_token and cancellation_token.is_cancelled:
                        raise CancelledError()
                    _progress("OCR", current=idx, total=len(regions), message=f"OCR {idx}/{len(regions)}")
                    try:
                        crop = cropper.crop_region(region)
                        result = ocr_provider.recognize(crop.to_pil(), region_bbox=region.global_bbox)
                        if result.text:
                            ocr_regions.append(
                                _replace_region(
                                    region,
                                    text=result.text,
                                    ocr_confidence=result.confidence,
                                    metadata={**region.metadata, "ocr_warnings": result.warnings},
                                )
                            )
                        else:
                            warnings.extend(result.warnings)
                            ocr_regions.append(region)
                    except Exception as e:
                        logger.error(f"OCR failed for region {region.id}: {e}")
                        warnings.append(f"OCR region {region.id}: {e}")
                        ocr_regions.append(region)
                cropper.clear_gpu_cache()
                regions = ocr_regions
                ocr_elapsed = time.time() - ocr_start
                stage_timings["ocr"] = round(time.perf_counter() - t_ocr_start, 3)
                try:
                    ocr_provider.unload()
                except Exception:
                    pass

        if cancellation_token and cancellation_token.is_cancelled:
            raise CancelledError()

        # 8. Save outputs
        _progress("Saving results")
        analysis_dir = output_path / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        regions_json = analysis_dir / "regions.json"
        with open(regions_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pages": len(pages),
                    "windows": len(windows),
                    "regions": [region_to_dict(r) for r in regions],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        elapsed = time.time() - start_time
        stage_timings["total"] = round(time.perf_counter() - t_total_start, 3)
        summary = {
            "pages": len(pages),
            "windows": len(windows),
            "regions": len(regions),
            "auto": sum(1 for r in regions if r.status == RegionStatus.AUTO),
            "review": sum(1 for r in regions if r.status == RegionStatus.REVIEW),
            "skip": sum(1 for r in regions if r.status == RegionStatus.SKIP),
            "elapsed_time": round(elapsed, 2),
            "stage_timings": stage_timings,
            "warnings": warnings,
        }
        summary_path = analysis_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 9. Global preview
        _progress("Rendering preview")
        preview_path = self._render_global_preview(pages, regions, output_path)

        _progress("Completed", current=1, total=1, message="Analysis complete")

        return AnalysisResult(
            pages=pages,
            windows=windows,
            regions=regions,
            elapsed_time=elapsed,
            visualization_paths=window_visualization_paths + [preview_path],
            warnings=warnings,
            ocr_elapsed_time=ocr_elapsed,
            stage_timings=stage_timings,
        )

    def _render_global_preview(
        self,
        pages: list[Page],
        regions: list[Region],
        output_path: Path,
    ) -> Path:
        """Global preview görseli oluşturur."""
        from PIL import Image

        if not pages:
            preview_path = output_path / "analysis" / "preview.png"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (1, 1), (255, 255, 255)).save(preview_path)
            return preview_path

        first = pages[0]
        with Image.open(first.path) as sample:
            width, _ = sample.size

        total_height = sum(p.height for p in pages)

        full = Image.new("RGB", (width, total_height), (255, 255, 255))
        y_cursor = 0
        for page in pages:
            with Image.open(page.path) as img:
                full.paste(img, (0, y_cursor))
                y_cursor += page.height

        full = draw_regions(full, regions, window_y_start=0)
        preview_path = output_path / "analysis" / "preview.png"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        full.save(preview_path, "PNG")
        return preview_path


def _replace(config: Config, **kwargs) -> Config:
    """Config ile yeni bir Config oluşturur (override edilebilir alanlar için)."""
    from dataclasses import replace

    allowed = {
        "window_height",
        "window_overlap",
        "input_extensions",
        "output_format",
        "log_level",
        "log_file",
        "min_confidence",
    }
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return replace(config, **filtered)


def _replace_status(region: Region, new_status: RegionStatus) -> Region:
    """Region durumunu değiştirir (yeni Region döndürür)."""
    from dataclasses import replace

    return replace(
        region,
        status=new_status,
    )


def _replace_region(region: Region, **kwargs) -> Region:
    """Region alanlarını değiştirir (yeni Region döndürür)."""
    from dataclasses import replace

    allowed = {
        "id",
        "global_bbox",
        "type",
        "detection_confidence",
        "source_window_ids",
        "status",
        "text",
        "ocr_confidence",
        "translation",
        "review_reason",
        "metadata",
    }
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return replace(region, **filtered)


def _get_model_identity(detector: DetectorProvider) -> tuple[str, str | float]:
    """Detector'dan model_id ve model_mtime çıkarır."""
    model_path = getattr(detector, "_model_path", None)
    if model_path is not None and Path(model_path).exists():
        model_id = str(Path(model_path).resolve())
        try:
            model_mtime = os.path.getmtime(str(model_path))
        except OSError:
            model_mtime = "unknown"
    else:
        model_id = getattr(detector, "name", "unknown")
        model_mtime = "unknown"
    cache_schema = getattr(detector, "cache_schema_version", None)
    if cache_schema:
        model_id = f"{model_id}|{cache_schema}"
    return model_id, model_mtime


def _image_to_bytes(image) -> bytes:
    """PIL Image'ı deterministic byte string'e çevirir (PNG encode)."""
    if hasattr(image, "tobytes"):
        return image.tobytes()
    buf = io.BytesIO()
    if hasattr(image, "save"):
        image.save(buf, format="PNG")
    elif isinstance(image, bytes):
        return image
    else:
        buf.write(bytes(image))
    return buf.getvalue()


def _global_detection_to_window(det: Detection, window_y_start: int) -> Detection:
    """Global Detection'ı window-local koordinata çevirir (visualization için)."""
    local_bbox = global_bbox_to_window(det.bbox, window_y_start)
    metadata = _offset_geometry_metadata(det.metadata, -window_y_start)
    return Detection(
        bbox=local_bbox,
        confidence=det.confidence,
        type=det.type,
        source_window_id=det.source_window_id,
        mask=det.mask,
        metadata=metadata,
    )


def _offset_geometry_metadata(metadata: object, y_offset: int) -> dict:
    """Translate every compact CTD geometry field without expanding it to a pixel mask."""
    result = dict(metadata) if isinstance(metadata, dict) else {}
    for key in ("polygon",):
        polygon = result.get(key)
        if isinstance(polygon, list) and polygon:
            result[key] = [[float(p[0]), float(p[1]) + y_offset] for p in polygon]
    for key in ("line_polygons", "segmentation_polygons"):
        polygons = result.get(key)
        if isinstance(polygons, list):
            result[key] = [
                [[float(p[0]), float(p[1]) + y_offset] for p in polygon]
                for polygon in polygons
                if isinstance(polygon, list) and len(polygon) >= 3
            ]
    block_bbox = result.get("ctd_block_bbox")
    if isinstance(block_bbox, list) and len(block_bbox) == 4:
        result["ctd_block_bbox"] = [
            float(block_bbox[0]), float(block_bbox[1]) + y_offset,
            float(block_bbox[2]), float(block_bbox[3]) + y_offset,
        ]
    return result
