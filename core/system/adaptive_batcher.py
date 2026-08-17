from dataclasses import dataclass, asdict
import gc
from typing import Any, Callable, Sequence, TypeVar
from loguru import logger

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchConfig:
    """Global configuration for elastic/manual batch execution."""

    mode: str = "auto"  # "auto" or "manual"
    vram_ceiling: float = 0.95  # 0.70 to 0.98
    lama_batch: int = 24  # 1 to 256
    ocr_vl_batch: int = 32  # 1 to 256
    llm_chunk: int = 16  # 1 to 64
    cpu_ocr_workers: int = 10  # 1 to 16


_GLOBAL_BATCH_CONFIG = BatchConfig()


def get_batch_config() -> BatchConfig:
    return _GLOBAL_BATCH_CONFIG


def set_batch_config(config: BatchConfig) -> None:
    global _GLOBAL_BATCH_CONFIG
    _GLOBAL_BATCH_CONFIG = config


class ElasticAdaptiveBatcher:
    """Manages elastic batch execution with proactive 95% VRAM checks and N -> N-1 reactive OOM recovery."""

    def __init__(
        self,
        default_batch_size: int = 32,
        min_batch_size: int = 1,
        vram_ceiling: float = 0.95,
    ) -> None:
        cfg = get_batch_config()
        self.default_batch_size = default_batch_size
        self.min_batch_size = min_batch_size
        self.vram_ceiling = cfg.vram_ceiling if cfg.mode == "auto" else vram_ceiling
        self.current_optimal_batch = default_batch_size

    def check_vram_and_adjust(self) -> None:
        """Proactively checks VRAM usage and decrements batch if exceeding ceiling."""
        try:
            import torch

            if not torch.cuda.is_available():
                return

            total_vram = torch.cuda.get_device_properties(0).total_memory
            if total_vram <= 0:
                return

            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            ratio = max(allocated, reserved) / total_vram

            if ratio >= self.vram_ceiling:
                prev = self.current_optimal_batch
                self.current_optimal_batch = max(self.min_batch_size, self.current_optimal_batch - 1)
                logger.warning(
                    "[ELASTIC BATCH] VRAM ceiling reached ({:.1f}% >= {:.1f}%). Reducing batch: {} -> {}",
                    ratio * 100,
                    self.vram_ceiling * 100,
                    prev,
                    self.current_optimal_batch,
                )
                torch.cuda.empty_cache()
                gc.collect()
        except Exception:
            pass

    def execute(
        self,
        items: Sequence[T],
        process_fn: Callable[[Sequence[T]], Sequence[R]],
        batch_size: int | None = None,
    ) -> list[R]:
        """Executes items in chunks with step-by-step (N -> N-1) decay on OOM."""
        if not items:
            return []

        if batch_size is not None:
            self.current_optimal_batch = min(self.current_optimal_batch, batch_size)

        results: list[R] = []
        idx = 0
        total = len(items)

        while idx < total:
            self.check_vram_and_adjust()
            chunk_size = max(self.min_batch_size, self.current_optimal_batch)
            chunk = items[idx : idx + chunk_size]

            try:
                chunk_results = process_fn(chunk)
                results.extend(chunk_results)
                idx += len(chunk)
            except Exception as e:
                err_str = str(e).lower()
                is_oom = (
                    "out of memory" in err_str
                    or "cuda error: out of memory" in err_str
                    or "cuda out of memory" in err_str
                    or e.__class__.__name__ == "OutOfMemoryError"
                )

                if is_oom:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    gc.collect()

                    prev_batch = self.current_optimal_batch
                    self.current_optimal_batch = max(self.min_batch_size, self.current_optimal_batch - 1)

                    logger.warning(
                        "[ELASTIC BATCH OOM] Memory pressure caught! Decayed batch: {} -> {}. Retrying chunk.",
                        prev_batch,
                        self.current_optimal_batch,
                    )

                    if prev_batch <= self.min_batch_size:
                        logger.error("[ELASTIC BATCH] OOM at minimum batch size {}", self.min_batch_size)
                        raise
                else:
                    raise

        return results


def execute_with_elastic_batch(
    items: Sequence[T],
    process_func: Callable[[Sequence[T]], Sequence[R]],
    initial_batch: int = 32,
    vram_threshold: float = 0.95,
    min_batch: int = 1,
) -> list[R]:
    """Helper function executing items with elastic batching."""
    batcher = ElasticAdaptiveBatcher(
        default_batch_size=initial_batch,
        min_batch_size=min_batch,
        vram_ceiling=vram_threshold,
    )
    return batcher.execute(items, process_func)
