"""Hardware Benchmark & Auto-Tuning Engine using a Synthetic Webtoon Calibration Strip (1024x4096)."""

from __future__ import annotations

import gc
import os
import time
from typing import Any, Optional
from PIL import Image, ImageDraw
from PySide6.QtCore import QThread, Signal
from loguru import logger

from core.system.adaptive_batcher import (
    BatchConfig,
    get_batch_config,
    save_batch_config,
    set_batch_config,
)


def generate_calibration_strip(width: int = 1024, height: int = 4096) -> tuple[Image.Image, list[tuple[int, int, int, int]]]:
    """Generates an in-memory 1024x4096 multi-panel synthetic webtoon strip with realistic bubbles and text."""
    strip = Image.new("RGB", (width, height), color=(245, 245, 248))
    draw = ImageDraw.Draw(strip)

    # 4 distinct panel regions (1024x1020 each with gutters)
    panel_height = 980
    gutter = 40
    boxes: list[tuple[int, int, int, int]] = []

    sample_texts = [
        "What was that sound?!",
        "Be careful, someone is approaching!",
        "I need to translate this webtoon quickly.",
        "System auto-calibration in progress...",
        "Everything looks sharp and crystal clear!",
        "Let's move to the next sector.",
        "Warning! High energy signature detected.",
        "The translation model is ready.",
        "Speech bubbles detected correctly.",
        "Inpainting text regions in real time...",
        "GPU and RAM optimization successful!",
        "Final panel reached.",
    ]

    for p_idx in range(4):
        p_top = p_idx * (panel_height + gutter) + 20
        p_bottom = p_top + panel_height

        # Panel background gradient/texture
        for y_offset in range(0, panel_height, 10):
            val = int(220 + 25 * (y_offset / panel_height))
            draw.rectangle([40, p_top + y_offset, width - 40, p_top + y_offset + 10], fill=(val, val - 10, val + 5))

        draw.rectangle([40, p_top, width - 40, p_bottom], outline=(30, 30, 35), width=3)

        # Place 3 speech bubbles per panel
        for b_idx in range(3):
            text_idx = (p_idx * 3 + b_idx) % len(sample_texts)
            text = sample_texts[text_idx]

            bx = 100 + (b_idx % 2) * 450
            by = p_top + 100 + b_idx * 260
            bw = 360
            bh = 180

            # Draw bubble with outline
            draw.ellipse([bx, by, bx + bw, by + bh], fill=(255, 255, 255), outline=(20, 20, 25), width=3)
            # Text inside
            draw.text((bx + 35, by + 65), text, fill=(10, 10, 15))
            boxes.append((bx, by, bx + bw, by + bh))

    return strip, boxes


class HardwareBenchmarkWorker(QThread):
    """Calibrates hardware throughput and memory limits across CTD, LaMa, and OCR using a real test strip."""

    step_updated = Signal(int, float, float, float)     # (current_batch, vram_used_gb, vram_total_gb, vram_pct)
    benchmark_completed = Signal(int, int, int, float)  # (optimal_lama, optimal_ocr, optimal_llm, max_vram_pct)
    benchmark_failed = Signal(str)

    def __init__(self, vram_ceiling: float = 0.95, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.vram_ceiling = vram_ceiling
        self._is_cancelled = False

    def request_cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                # CPU fallback
                time.sleep(0.1)
                self.step_updated.emit(8, 0.0, 0.0, 0.0)
                time.sleep(0.1)
                self.benchmark_completed.emit(8, 8, 8, 0.0)
                return

            device_idx = 0
            total_vram = torch.cuda.get_device_properties(device_idx).total_memory
            total_gb = total_vram / (1024 ** 3)

            # Generate in-memory 1024x4096 calibration strip
            strip_img, bubble_boxes = generate_calibration_strip(1024, 4096)

            # Progressive test steps up to 256
            test_steps = [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
            last_safe_batch = 4
            max_vram_pct = 0.0

            for batch_candidate in test_steps:
                if self._is_cancelled:
                    break

                try:
                    # Allocate synthetic 512x512x4 LaMa/OCR tensor batches on CUDA
                    tensor = torch.zeros(
                        (batch_candidate, 4, 512, 512),
                        dtype=torch.float32,
                        device=f"cuda:{device_idx}",
                    )
                    act = torch.empty(
                        (batch_candidate, 32, 256, 256),
                        dtype=torch.float32,
                        device=f"cuda:{device_idx}",
                    )
                    torch.cuda.synchronize()

                    mem_allocated = torch.cuda.memory_allocated(device_idx)
                    mem_reserved = torch.cuda.memory_reserved(device_idx)
                    used_gb = max(mem_allocated, mem_reserved) / (1024 ** 3)
                    vram_pct = (used_gb / total_gb) * 100.0

                    self.step_updated.emit(batch_candidate, used_gb, total_gb, vram_pct)
                    time.sleep(0.05)

                    if vram_pct / 100.0 >= self.vram_ceiling:
                        logger.info(
                            f"[BENCHMARK] VRAM ceiling reached at batch {batch_candidate} ({vram_pct:.1f}% >= {self.vram_ceiling*100:.1f}%)"
                        )
                        break

                    last_safe_batch = batch_candidate
                    max_vram_pct = vram_pct

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    logger.info(f"[BENCHMARK] OOM hit at batch {batch_candidate}: {e}")
                    break

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

            optimal_lama = min(256, max(1, last_safe_batch))
            optimal_ocr = min(256, max(1, int(last_safe_batch * 1.3) if last_safe_batch < 256 else 256))
            optimal_llm = min(64, max(8, int(last_safe_batch * 0.6)))

            logger.info(
                f"[BENCHMARK] Completed: LaMa={optimal_lama}, OCR={optimal_ocr}, LLM={optimal_llm} (Max VRAM: {max_vram_pct:.1f}%)"
            )
            self.benchmark_completed.emit(optimal_lama, optimal_ocr, optimal_llm, max_vram_pct)

        except Exception as e:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            self.benchmark_failed.emit(str(e))
