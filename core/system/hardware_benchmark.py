"""Hardware Stress Test & Auto-Tuning Worker for GPU VRAM & Batch Calibration."""

from __future__ import annotations

import gc
import time
from typing import Optional
from PySide6.QtCore import QThread, Signal
from loguru import logger


class HardwareBenchmarkWorker(QThread):
    """QThread worker that stresses GPU VRAM in steps up to 95% ceiling to find optimal batch sizes."""

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
                time.sleep(0.4)
                self.step_updated.emit(8, 0.0, 0.0, 0.0)
                time.sleep(0.4)
                self.benchmark_completed.emit(8, 8, 8, 0.0)
                return

            device_idx = 0
            total_vram = torch.cuda.get_device_properties(device_idx).total_memory
            total_gb = total_vram / (1024 ** 3)

            # Progressive test steps up to 256
            test_steps = [8, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256]
            last_safe_batch = 8
            max_vram_pct = 0.0

            allocated_tensors = []

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
                    allocated_tensors.append(tensor)

                    # Simulate forward activation overhead
                    act = torch.empty(
                        (batch_candidate, 64, 256, 256),
                        dtype=torch.float32,
                        device=f"cuda:{device_idx}",
                    )
                    allocated_tensors.append(act)

                    torch.cuda.synchronize()

                    mem_allocated = torch.cuda.memory_allocated(device_idx)
                    mem_reserved = torch.cuda.memory_reserved(device_idx)
                    used_gb = max(mem_allocated, mem_reserved) / (1024 ** 3)
                    vram_pct = (used_gb / total_gb) * 100.0

                    self.step_updated.emit(batch_candidate, used_gb, total_gb, vram_pct)
                    time.sleep(0.25)  # Visual feedback duration

                    if vram_pct / 100.0 >= self.vram_ceiling:
                        logger.info(
                            f"[BENCHMARK] VRAM ceiling hit at batch {batch_candidate} ({vram_pct:.1f}% >= {self.vram_ceiling*100:.1f}%)"
                        )
                        break

                    last_safe_batch = batch_candidate
                    max_vram_pct = vram_pct

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    logger.info(f"[BENCHMARK] OOM hit at batch {batch_candidate}: {e}")
                    break

            # Clean up all allocated benchmark tensors
            allocated_tensors.clear()
            torch.cuda.empty_cache()
            gc.collect()

            # Calibrate modules based on safe ceiling (up to 256)
            optimal_lama = max(1, last_safe_batch)
            optimal_ocr = min(256, max(1, int(last_safe_batch * 1.3)))
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
